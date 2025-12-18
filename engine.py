import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO

class AIEngine:
    def __init__(self, model_path, dino_size):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.yolo = YOLO(model_path) # โหลดโมเดล Segmentation [cite: 2025-11-11]
        self.dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(self.device).eval()
        
        # Senior Tip: dino_size ต้องเท่ากันทั้งตอน Build DB และ Inference [cite: 2025-12-18]
        self.dino_size = dino_size 
        self.preprocess = transforms.Compose([
            transforms.Resize((dino_size, dino_size), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def extract_vector(self, img_bgr):
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        t = self.preprocess(Image.fromarray(img_rgb)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            vec = self.dino(t).flatten().cpu().numpy()
        return vec / (np.linalg.norm(vec) + 1e-8)

    def detect_and_crop(self, img_bgr, conf):
        """ใช้ Segmentation Mask เพื่อตัดพื้นหลังทิ้ง (Senior Method) [cite: 2025-12-05]"""
        results = self.yolo(img_bgr, verbose=False, conf=conf, task='segment') # ระบุ task เป็น segment [cite: 2025-11-11]
        
        res = results[0]
        if res.masks is not None:
            # 1. เลือกตัวที่มีค่าความมั่นใจสูงสุด [cite: 2025-11-11]
            idx = res.boxes.conf.argmax()
            
            # 2. สร้าง Mask สีดำขนาดเท่าภาพจริง [cite: 2025-12-05]
            mask = res.masks.data[idx].cpu().numpy()
            mask = cv2.resize(mask, (img_bgr.shape[1], img_bgr.shape[0]))
            
            # 3. ลบพื้นหลัง (Bitwise AND) เหลือแต่แผงยา [cite: 2025-11-11, 2025-12-05]
            masked_img = img_bgr.copy()
            masked_img[mask == 0] = 0 # พื้นหลังเป็นดำ [cite: 2025-12-05]
            
            # 4. Crop ตาม Bounding Box [cite: 2025-11-11]
            x1, y1, x2, y2 = res.boxes.xyxy[idx].cpu().numpy().astype(int)
            crop = masked_img[y1:y2, x1:x2]
            
            return crop if crop.size > 0 else img_bgr
            
        return img_bgr