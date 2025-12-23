import os
import yaml
from dotenv import load_dotenv

# โหลด .env ตั้งแต่เริ่ม
load_dotenv()

def load_config(config_path="config.yaml"):
    """
    โหลดไฟล์ YAML config อย่างปลอดภัย
    ถ้าไม่เจอไฟล์ จะ return dict ว่างเพื่อป้องกันโปรแกรม crash
    """
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f) or {}
    return {}

def get_base_drug_names(db_dict):
    """
    แปลง keys จาก database (เช่น 'para_box_rot90') ให้เหลือแค่ชื่อยาหลัก ('para')
    Logic: ตัดคำตาม suffix ที่ระบุ (_box, _blister, etc.)
    """
    names = set()
    if not db_dict:
        return []
        
    for k in db_dict.keys():
        base = k.split('_box')[0].split('_blister')[0].split('_pack')[0].split('_rot')[0] 
        names.add(base)
    return sorted(list(names))

class AppPaths:
    """
    Helper Class สำหรับจัดการ Path ต่างๆ ให้เป็นระเบียบ
    เรียกใช้: paths = AppPaths(config)
    """
    def __init__(self, config):
        artifacts = config.get('artifacts', {})
        paths = config.get('paths', {})
        
        self.pkl_path = str(artifacts.get('pack_vec', 'database/pill_fingerprints.pkl'))
        self.json_path = str(artifacts.get('drug_list', 'database/drug_list.json'))
        self.model_path = str(artifacts.get('model', 'models/seg_best_process.pt'))
        self.img_db_root = str(paths.get('db_images', 'database_images'))
        
        # สร้างโฟลเดอร์อัตโนมัติเมื่อ Initialize
        os.makedirs(os.path.dirname(self.pkl_path), exist_ok=True)
        os.makedirs(self.img_db_root, exist_ok=True)