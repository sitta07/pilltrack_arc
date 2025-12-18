import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO

class AIEngine:
    def __init__(self, model_path, dino_size):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. YOLO Segmentation
        self.yolo = YOLO(model_path)
        
        # 2. DINOv2 Model
        self.dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14').to(self.device).eval()
        self.dino_size = dino_size 
        
        # 3. SIFT Detector (เพิ่มใหม่)
        self.sift = cv2.SIFT_create()
        
        # Preprocessing มาตรฐาน
        self.preprocess = transforms.Compose([
            transforms.Resize((dino_size, dino_size), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    def _enhance_image(self, img_bgr):
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # ใส่ CLAHE เฉพาะช่องแสง (L) เพื่อให้ Detail เด้ง แต่สีไม่เพี้ยน
        l2 = self.clahe.apply(l)
        
        # รวมร่างกลับ
        lab = cv2.merge((l2, a, b))
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def extract_features(self, img_bgr):
        """
        Return: (dino_vector, sift_descriptors)
        """
        # 1. เพิ่มขั้นตอน Enhancement
        enhanced_img = self._enhance_image(img_bgr)
        
        # --- PART A: DINOv2 Extraction ---
        img_rgb = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2RGB)
        t = self.preprocess(Image.fromarray(img_rgb)).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            dino_vec = self.dino(t).flatten().cpu().numpy()
            
        # L2 Normalization
        dino_vec = dino_vec / (np.linalg.norm(dino_vec) + 1e-8)

        # --- PART B: SIFT Extraction (เพิ่มใหม่) ---
        # SIFT ต้องการภาพ Grayscale
        gray = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2GRAY)
        # return keypoints(ไม่ใช้), descriptors(ใช้)
        _, sift_des = self.sift.detectAndCompute(gray, None)

        # คืนค่ากลับไปทั้งคู่
        return dino_vec, sift_des

    def detect_and_crop(self, img_bgr, conf, padding_pct=0.05):
        # (ส่วนนี้เหมือนเดิม 100% ไม่ต้องแก้)
        results = self.yolo(img_bgr, verbose=False, conf=conf, task='segment')
        
        res = results[0]
        if res.masks is not None:
            idx = res.boxes.conf.argmax()
            
            mask = res.masks.data[idx].cpu().numpy()
            mask = cv2.resize(mask, (img_bgr.shape[1], img_bgr.shape[0]))
            
            masked_img = img_bgr.copy()
            masked_img[mask == 0] = 0 
            
            x1, y1, x2, y2 = res.boxes.xyxy[idx].cpu().numpy().astype(int)
            h, w = img_bgr.shape[:2]
            
            pad_w = int((x2 - x1) * padding_pct)
            pad_h = int((y2 - y1) * padding_pct)
            
            x1 = max(0, x1 - pad_w)
            y1 = max(0, y1 - pad_h)
            x2 = min(w, x2 + pad_w)
            y2 = min(h, y2 + pad_h)
            
            crop = masked_img[y1:y2, x1:x2]
            
            return crop if crop.size > 0 else img_bgr
            
        return img_bgr