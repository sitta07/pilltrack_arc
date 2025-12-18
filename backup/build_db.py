import os

# 🚨 CRITICAL FIX FOR MAC M1/M2/M3 (MPS) 🚨
# สั่งให้ PyTorch สลับไปใช้ CPU ชั่วคราวถ้าเจอคำสั่งที่ MPS ทำไม่ได้ (เช่น Bicubic Interpolation)
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import cv2
import torch
import numpy as np
import pickle
import shutil
import gc
from ultralytics import YOLO
from torchvision import transforms
from PIL import Image
import kornia.augmentation as K
import torch.nn as nn

# ================= CONFIGURATION =================
DB_FOLDER = 'database_images'       

# Output Files (DINOv2)
OUT_PILLS = {'vec': 'db_pills_dino.pkl', 'col': 'colors_pills.pkl', 'ratio': 'ratios_pills.pkl'}
OUT_PACKS = {'vec': 'db_packs_dino.pkl', 'col': 'colors_packs.pkl', 'ratio': 'ratios_packs.pkl'}

MODEL_PILL_PATH = 'pills_seg.pt'   
MODEL_PACK_PATH = 'box_db_best_2.pt' 

# Settings
NUM_VARIATIONS = 50    
AUG_SIZE = 800          # เผื่อพื้นที่ให้ Augmentation
MODEL_INPUT_SIZE = 336  # DINOv2 ชอบ 14 หารลงตัว (448/14 = 32)
CROP_PADDING = 30

# ================= ⚡ GPU / DEVICE SETUP ⚡ =================
if torch.cuda.is_available():
    device = torch.device("cuda")
    aug_device = torch.device("cuda")
    print(f"🚀 POWERED BY NVIDIA CUDA ({torch.cuda.get_device_name(0)})")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    aug_device = torch.device("cpu") # Kornia บน Mac ใช้ CPU ชัวร์สุด
    print("🚀 POWERED BY APPLE SILICON (MPS) + Fallback Enabled")
else:
    device = torch.device("cpu")
    aug_device = torch.device("cpu")
    print("🐢 POWERED BY CPU")

# ================= 1. SETUP DINOv2 & PIPELINE =================

print("⏳ Loading DINOv2 (ViT-S/14) from Torch Hub...")
# ใช้ dinov2_vits14 (Small)
embedder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
embedder.eval().to(device)
print("✅ DINOv2 Loaded! Vector Size: 384 dimensions")

# --- Pipelines Setup ---
pre_aug_transform = transforms.Compose([
    transforms.Resize((AUG_SIZE, AUG_SIZE)), 
    transforms.ToTensor()
])

final_transform = transforms.Compose([
    transforms.Resize((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), antialias=True),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ================= 2. HELPER FUNCTIONS =================
def apply_clahe(img_bgr):
    try:
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl,a,b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    except: return img_bgr

def get_aspect_ratio(w, h):
    if w == 0 or h == 0: return 1.0
    return min(w, h) / max(w, h)

def get_smart_color(img_bgr, mask_binary=None):
    if img_bgr is None or img_bgr.size == 0: return np.array([0.0, 0.0, 0.0])
    try:
        img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        mask_colorful = cv2.inRange(img_hsv, np.array([0, 40, 40]), np.array([180, 255, 230]))
        if mask_binary is not None:
            if mask_binary.shape[:2] != img_hsv.shape[:2]:
                mask_binary = cv2.resize(mask_binary, (img_hsv.shape[1], img_hsv.shape[0]))
            final_mask = cv2.bitwise_and(mask_binary, mask_binary, mask=mask_colorful)
        else: final_mask = mask_colorful 
        pixels = img_hsv[final_mask > 0] if cv2.countNonZero(final_mask) > 50 else img_hsv.reshape(-1, 3)
        if len(pixels) == 0: return np.array([0.0, 0.0, 0.0])
        return np.mean(pixels, axis=0)
    except: return np.array([0.0, 0.0, 0.0])

# ================= 3. BUILD PROCESSOR =================
def process_build(mode='pill'):
    is_pill = (mode == 'pill')
    print(f"\n🔨 BUILDING DATABASE (DINOv2): {mode.upper()} ...")
    
    # 📌 AUGMENTATION PIPELINE (Optimized for Safety)
    if is_pill:
        model_path = MODEL_PILL_PATH
        class_suffix_base = "_pill"
        aug_pipeline = nn.Sequential(
            K.RandomPerspective(distortion_scale=0.5, p=0.7),
            K.RandomRotation(degrees=15.0, p=1.0),
            K.RandomMotionBlur(kernel_size=(3, 5), angle=(0., 360.), direction=(-1., 1.), p=0.3),
            K.RandomGaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0), p=0.3),
            # 🔥 Fine-tuned: ลด hue ลงเหลือ 0.01 ป้องกันสีเพี้ยนจนจำผิด
            K.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.01, p=0.8),
            K.RandomGaussianNoise(mean=0., std=0.05, p=0.2),
        ).to(aug_device)
    else:
        model_path = MODEL_PACK_PATH
        class_suffix_base = "_pack"
        aug_pipeline = nn.Sequential(
            K.RandomPerspective(distortion_scale=0.6, p=0.8),
            K.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=(-10, 10), p=0.5),
            K.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.0, hue=0.0, p=0.7),
            K.RandomGaussianNoise(mean=0., std=0.05, p=0.2), 
        ).to(aug_device)

    try: 
        model = YOLO(model_path)
    except: return print(f"❌ Error: Model {model_path} not found.")

    database = {}
    color_database = {}
    ratio_database = {}
    
    for folder_name in os.listdir(DB_FOLDER):
        folder_path = os.path.join(DB_FOLDER, folder_name)
        if not os.path.isdir(folder_path): continue
        
        print(f"   👉 Processing: {folder_name}")
        temp_colors = []
        temp_ratios = [] 
        
        files = [f for f in os.listdir(folder_path) if f.lower().endswith(('jpg', 'png', 'jpeg'))]
        
        for filename in files:
            img = cv2.imread(os.path.join(folder_path, filename))
            if img is None: continue
            h_img, w_img = img.shape[:2]
            
            # AI Inference
            results = model(img, verbose=False, conf=0.8 if is_pill else 0.8)
            if len(results[0].boxes) == 0: continue

            # Filter Top-2
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            classes = results[0].boxes.cls.cpu().numpy().astype(int)
            confs = results[0].boxes.conf.cpu().numpy()
            masks = results[0].masks
            has_mask = masks is not None and is_pill

            all_dets = []
            for i in range(len(boxes)):
                all_dets.append({'box': boxes[i], 'cls': classes[i], 'conf': confs[i], 'orig_idx': i})
            
            all_dets.sort(key=lambda x: x['conf'], reverse=True)
            target_dets = all_dets[:2] 
            if len(target_dets) == 1: target_dets.append(target_dets[0].copy())
            
            for item in target_dets:
                if item['cls'] != 0: continue
                box = item['box']
                x1, y1, x2, y2 = box
                temp_ratios.append(get_aspect_ratio(x2-x1, y2-y1))
                
                processed_img = img.copy()
                if has_mask:
                    mask_img = np.zeros((h_img, w_img), dtype=np.uint8)
                    contour = masks.xy[item['orig_idx']].astype(np.int32)
                    cv2.fillPoly(mask_img, [contour], 255)
                    noise_bg = np.full((h_img, w_img, 3), 128, dtype=np.uint8)
                    mask_bool = mask_img > 0
                    processed_img = noise_bg
                    processed_img[mask_bool] = img[mask_bool]

                cx1, cy1 = max(0, x1 - CROP_PADDING), max(0, y1 - CROP_PADDING)
                cx2, cy2 = min(w_img, x2 + CROP_PADDING), min(h_img, y2 + CROP_PADDING)
                final_crop = processed_img[cy1:cy2, cx1:cx2]
                if final_crop.size == 0: continue
                final_crop = apply_clahe(final_crop)

                rotations = [(0, "_rot0"), (90, "_rot90"), (180, "_rot180")]
                for angle, suffix in rotations:
                    rotated_crop = final_crop.copy()
                    if angle == 90: rotated_crop = cv2.rotate(rotated_crop, cv2.ROTATE_90_CLOCKWISE)
                    elif angle == 180: rotated_crop = cv2.rotate(rotated_crop, cv2.ROTATE_180)
                    elif angle == 270: rotated_crop = cv2.rotate(rotated_crop, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    
                    full_key = f"{folder_name}{class_suffix_base}{suffix}"
                    if full_key not in database: database[full_key] = []

                    if is_pill and angle == 0:
                        h_c, w_c = rotated_crop.shape[:2]
                        center = rotated_crop[int(h_c*0.25):int(h_c*0.75), int(w_c*0.25):int(w_c*0.75)]
                        if center.size > 0:
                            col = get_smart_color(center)
                            if col.shape == (3,): temp_colors.append(col)
                    
                    pil_img = Image.fromarray(cv2.cvtColor(rotated_crop, cv2.COLOR_BGR2RGB))
                    # Call Processing
                    process_vector(pil_img, aug_pipeline, database, full_key)

        base_key = f"{folder_name}{class_suffix_base}"
        if is_pill and temp_colors:
            color_database[base_key] = np.mean(np.array(temp_colors), axis=0)
        else: color_database[base_key] = np.array([0.0, 0.0, 0.0])
            
        if temp_ratios: ratio_database[base_key] = sum(temp_ratios) / len(temp_ratios)
        else: ratio_database[base_key] = 1.0

    # Save
    with open(OUT_PILLS['vec'] if is_pill else OUT_PACKS['vec'], 'wb') as f: pickle.dump(database, f)
    with open(OUT_PILLS['col'] if is_pill else OUT_PACKS['col'], 'wb') as f: pickle.dump(color_database, f)
    with open(OUT_PILLS['ratio'] if is_pill else OUT_PACKS['ratio'], 'wb') as f: pickle.dump(ratio_database, f)
    
    print(f"✅ Phase {mode.upper()} Complete!")
    del model
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

def process_vector(pil_img, aug_pipeline, db_dict, key):
    try:
        # 1. Image -> Tensor -> AugDevice
        input_tensor = pre_aug_transform(pil_img).to(aug_device)
        
        all_vectors = [] # List to hold all variations
        
        # ลด Batch Size ลงเพราะ 448x448 กิน VRAM เยอะขึ้น
        BATCH_SIZE = 32 
        for _ in range(0, NUM_VARIATIONS, BATCH_SIZE):
            current_batch_size = min(BATCH_SIZE, NUM_VARIATIONS - _) 
            if current_batch_size <= 0: break
            
            mini_batch = input_tensor.unsqueeze(0).repeat(current_batch_size, 1, 1, 1)
            
            with torch.no_grad():
                # 2. Augment
                aug_batch = aug_pipeline(mini_batch)
                
                # 3. Transform for DINO
                final_batch = final_transform(aug_batch).to(device)
                
                # 4. Embed using DINOv2
                outputs = embedder(final_batch)
                if isinstance(outputs, dict): outputs = outputs['x_norm_clstoken']
                
                # Normalize (สำคัญมาก)
                vectors = outputs / outputs.norm(dim=1, keepdim=True)
                
                all_vectors.append(vectors.cpu().numpy())
                
            del mini_batch, aug_batch, final_batch, vectors
        
        # 🔥 CENTROID LOGIC 🔥
        # ยุบ 300 Vectors เหลือ 1 ตัวแทน (Mean) เพื่อประหยัด RAM
        if len(all_vectors) > 0:
            stacked = np.vstack(all_vectors)
            centroid = np.mean(stacked, axis=0)
            centroid = centroid / np.linalg.norm(centroid) # Normalize again
            
            # เก็บค่า Centroid ลง Database
            db_dict[key].append(centroid)

    except Exception as e:
        print(f"⚠️ Error in process_vector: {e}")

if __name__ == "__main__":
    # Clear Cache
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    process_build(mode='pill') 
    process_build(mode='pack') 
    
    print("\n🎉 ALL SYSTEMS GO: DINOv2 Database Built Successfully!")