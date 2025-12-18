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

    @staticmethod
    def get_unique_drugs(db_dict):
        """สกัดรายชื่อยาจากกุญแจใน pkl (Hardmode Logic)"""
        names = set()
        for k in db_dict.keys():
            if "_pack" in k: names.add(k.split('_pack')[0])
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
        
        logs.insert(0, log_entry) # เอาเหตุการณ์ล่าสุดไว้บนสุด
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(logs[:100], f, indent=4, ensure_ascii=False) # เก็บแค่ 100 รายการล่าสุด

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