import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO

class AIEngine:
    def __init__(self, model_path, dino_size):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.yolo = YOLO(model_path)
        self.dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(self.device).eval()
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

    def detect_and_crop(self, img, conf):
        results = self.yolo(img, verbose=False, conf=conf)
        if len(results[0].boxes) > 0:
            b = sorted(results[0].boxes, key=lambda x: x.conf, reverse=True)[0]
            x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().astype(int)
            return img[max(0, y1-20):min(img.shape[0], y2+20), max(0, x1-20):min(img.shape[1], x2+20)]
        return img