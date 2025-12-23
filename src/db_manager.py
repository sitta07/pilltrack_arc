import pickle
import json
import os
from datetime import datetime

class DBManager:
    @staticmethod
    def load_pkl(path):
        """โหลดไฟล์ Vector Database (.pkl)"""
        if os.path.exists(path):
            with open(path, 'rb') as f: return pickle.load(f)
        return {}

    @staticmethod
    def save_pkl(data, path):
        """บันทึกไฟล์ Vector Database (.pkl)"""
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    # ✅ [NEW] ฟังก์ชันพระเอก: จัดการโครงสร้างข้อมูลแบบ Hybrid
    @staticmethod
    def insert_data(db, name, dino_vec, sift_des):
        """
        แทรกข้อมูลลง Database โดยอัตโนมัติ (จัดการ Schema ให้อัตโนมัติ)
        Structure ใหม่: 
        {
            "Para_pack_1": {
                "dino": [vec1, vec2, ...],
                "sift": [des1, des2, ...]
            }
        }
        """
        # 1. ถ้ายังไม่มีชื่อนี้ ให้สร้างโครงสร้างรอ
        if name not in db:
            db[name] = {"dino": [], "sift": []}
        
        # 2. (Migration Support) เผื่อไปเจอไฟล์เก่าที่เป็น List ล้วน ให้แปลงร่างก่อน
        if isinstance(db[name], list):
            print(f"📦 Converting legacy format for {name}...")
            db[name] = {"dino": db[name], "sift": []}

        # 3. ยัดข้อมูลลงถัง
        db[name]["dino"].append(dino_vec)
        
        # SIFT descriptors อาจเป็น None ได้ (ถ้าภาพไม่มีจุดเด่น)
        if sift_des is not None:
            db[name]["sift"].append(sift_des)

    @staticmethod
    def get_unique_drugs(db_dict):
        """สกัดรายชื่อยาจากกุญแจใน pkl"""
        names = set()
        for k in db_dict.keys():
            # Logic ตัดคำว่า _pack ออก เพื่อให้ได้ชื่อยาเพียวๆ
            if "_pack" in k: 
                names.add(k.split('_pack')[0])
            else:
                names.add(k) # กรณีชื่อไม่มี _pack
        return sorted(list(names))

    @staticmethod
    def generate_metadata(drug_names, out_path):
        """สร้างไฟล์ Metadata JSON สำหรับ Raspberry Pi"""
        metadata = {
            "drugs": drug_names,
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total": len(drug_names)
        }
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=4, ensure_ascii=False)

    @staticmethod
    def add_log(event_type, drug_name="-", count=0, details="-"):
        """บันทึกประวัติการทำงาน (Audit Trail)"""
        log_path = "database/activity_log.json"
        
        # สร้างโฟลเดอร์ database ถ้ายังไม่มี
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        
        log_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "event": event_type,
            "drug": drug_name,
            "images": count,
            "details": details
        }
        
        logs = []
        if os.path.exists(log_path):
            try:
                with open(log_path, 'r', encoding='utf-8') as f:
                    logs = json.load(f)
            except: logs = []
        
        logs.insert(0, log_entry)
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(logs[:100], f, indent=4, ensure_ascii=False)

    @staticmethod
    def get_logs():
        """ดึงรายการ Log ทั้งหมดออกมาแสดงผล"""
        log_path = "database/activity_log.json"
        if os.path.exists(log_path):
            try:
                with open(log_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except: return []
        return []