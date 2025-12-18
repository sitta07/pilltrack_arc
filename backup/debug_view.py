import os
import cv2
import numpy as np
from ultralytics import YOLO

# ================= CONFIGURATION =================
# 📁 แก้ Path ให้ตรงกับเครื่องคุณ
DB_FOLDER = 'database_images'       
MODEL_PILL_PATH = 'pills_seg.pt'        # Segmentation Model
MODEL_PACK_PATH = 'seg_best_process.pt'      # Detection Model

# ⚙️ DEBUG SETTINGS
DEBUG_MODE = 'pack'          # 'pill' หรือ 'pack'
TOP_K_FILTER = 2             # เอาเฉพาะ 2 ตัว (ใช้เฉพาะโหมด PILL)
USE_GREY_BACKGROUND = True   # True = พื้นหลังเทา, False = พื้นหลังดำ
CROP_PADDING = 30            # เผื่อขอบรอบๆ

# ================= MAIN LOGIC =================
def run_debug():
    print(f"🚀 STARTING DEBUGGER: Mode={DEBUG_MODE.upper()}")
    if DEBUG_MODE == 'pill':
        print(f"   👉 Logic: Top-{TOP_K_FILTER} Only + Auto-Duplicate Single Pill")
    else:
        print(f"   👉 Logic: ALL DETECTIONS (Pack Mode - No filtering)")
    
    # 1. Load Model
    model_path = MODEL_PILL_PATH if DEBUG_MODE == 'pill' else MODEL_PACK_PATH
    print(f"📦 Loading Model: {model_path}")
    
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # 2. Loop Folders
    if not os.path.exists(DB_FOLDER):
        print(f"❌ Database folder not found: {DB_FOLDER}")
        return

    for folder_name in os.listdir(DB_FOLDER):
        folder_path = os.path.join(DB_FOLDER, folder_name)
        if not os.path.isdir(folder_path): continue
        
        print(f"\n📂 Scanning Folder: {folder_name}...")
        
        files = [f for f in os.listdir(folder_path) if f.lower().endswith(('jpg', 'png', 'jpeg'))]
        if not files: continue

        for filename in files:
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            if img is None: continue
            
            h_img, w_img = img.shape[:2]
            
            # --- AI INFERENCE ---
            # Config ตามที่คุณขอมา (0.8 ทั้งคู่)
            results = model(img, verbose=False, conf=0.8 if DEBUG_MODE == 'pill' else 0.8)
            
            if len(results[0].boxes) == 0:
                print(f"   ⚠️ No detection in {filename}")
                continue

            # ================= [STEP 1: PREPARE DATA] =================
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            confs = results[0].boxes.conf.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy().astype(int)
            masks = results[0].masks
            
            # ใช้ Mask เฉพาะโหมด Pill เท่านั้น
            has_mask = masks is not None
            
            # เก็บข้อมูลดิบ
            all_detections = []
            for i in range(len(boxes)):
                all_detections.append({
                    'conf': confs[i],
                    'box': boxes[i],
                    'cls': classes[i],
                    'orig_idx': i,  
                    'is_clone': False, 
                    'note': 'Original'
                })
            
            # เรียงลำดับความมั่นใจ (มาก -> น้อย)
            all_detections.sort(key=lambda x: x['conf'], reverse=True)
            
            # ================= [STEP 2: SPLIT LOGIC PILL vs PACK] =================
            target_detections = []

            if DEBUG_MODE == 'pill':
                # --- PILL LOGIC: เอาแค่ Top-K และ Clone ถ้ามีเม็ดเดียว ---
                target_detections = all_detections[:TOP_K_FILTER]
                
                # Auto-Duplicate Logic
                if len(target_detections) == 1:
                    clone_item = target_detections[0].copy()
                    clone_item['is_clone'] = True
                    clone_item['note'] = 'Clone (Rotated 180)'
                    target_detections.append(clone_item)
            else:
                # --- PACK LOGIC: เอาหมดทุกอันที่เจอ (ไม่ตัด ไม่โคลน) ---
                target_detections = all_detections

            # ================= [STEP 3: DISPLAY LOOP] =================
            view_img = img.copy()

            for rank, item in enumerate(target_detections):
                box = item['box']
                conf = item['conf']
                orig_i = item['orig_idx']
                is_clone = item['is_clone']
                note = item['note']
                
                x1, y1, x2, y2 = box

                # วาดกรอบบนรูปใหญ่ (เฉพาะตัวจริง)
                if not is_clone:
                    color = (0, 255, 0)
                    cv2.rectangle(view_img, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(view_img, f"#{rank+1} {conf:.2f}", (x1, y1-5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # --- PROCESS CROP ---
                cx1 = max(0, x1 - CROP_PADDING)
                cy1 = max(0, y1 - CROP_PADDING)
                cx2 = min(w_img, x2 + CROP_PADDING)
                cy2 = min(h_img, y2 + CROP_PADDING)
                
                if cx2 <= cx1 or cy2 <= cy1: continue

                crop_raw = img[cy1:cy2, cx1:cx2].copy()
                final_crop = crop_raw.copy()
                
                bg_status = "Raw Box"
                
                # Pill Mode with Mask
                if has_mask:
                    mask_full = np.zeros((h_img, w_img), dtype=np.uint8)
                    contour = masks.xy[orig_i].astype(np.int32)
                    cv2.fillPoly(mask_full, [contour], 255)
                    mask_crop = mask_full[cy1:cy2, cx1:cx2]
                    
                    if USE_GREY_BACKGROUND:
                        bg_layer = np.full(crop_raw.shape, 128, dtype=np.uint8)
                        fg = cv2.bitwise_and(crop_raw, crop_raw, mask=mask_crop)
                        bg = cv2.bitwise_and(bg_layer, bg_layer, mask=cv2.bitwise_not(mask_crop))
                        final_crop = cv2.add(fg, bg)
                        bg_status = "Grey BG"
                    else:
                        final_crop = cv2.bitwise_and(crop_raw, crop_raw, mask=mask_crop)
                        bg_status = "Black BG"
                else:
                    # Pack Mode (หรือ Pill ที่ไม่มี Mask) -> ใช้ภาพเต็มกล่องเลย
                    bg_status = "Raw Crop (No Mask)"

                # --- ROTATION FOR CLONE (Pill Only) ---
                if is_clone:
                    final_crop = cv2.rotate(final_crop, cv2.ROTATE_180)

                # --- DISPLAY ---
                h, w = final_crop.shape[:2]
                if h > 0 and w > 0:
                    scale = 300 / max(h, w)
                    show_crop = cv2.resize(final_crop, (int(w*scale), int(h*scale)))
                else:
                    show_crop = np.zeros((200, 200, 3), dtype=np.uint8)

                cv2.imshow("1. Full Image", cv2.resize(view_img, (640, 480)))
                cv2.imshow(f"2. Input to AI ({note})", show_crop)

                print(f"   👉 {filename} | Obj #{rank+1} | {note} | {bg_status}")

                key = cv2.waitKey(0) & 0xFF
                if key == ord('q'):
                    print("👋 Exiting...")
                    cv2.destroyAllWindows()
                    return

    cv2.destroyAllWindows()
    print("✅ Finished checking all files.")

if __name__ == "__main__":
    run_debug()