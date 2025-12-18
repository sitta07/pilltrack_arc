import streamlit as st
import os
import cv2
import torch
import numpy as np
import pickle
import shutil
import json
import yaml
import boto3
from datetime import datetime
from dotenv import load_dotenv
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO
from botocore.exceptions import ClientError

# ================= ⚙️ SETUP & CONFIG =================
load_dotenv()
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

BASE_DIR = os.getcwd()
BUCKET_NAME = os.getenv('S3_BUCKET_NAME')
S3_CLIENT = boto3.client('s3')

FILES = config['artifacts']
# นิยามไฟล์ Metadata
DRUG_LIST_FILE = "database/drug_list.json"
FILES['drug_list'] = DRUG_LIST_FILE

DINO_SIZE = config['settings']['dino_size']
DB_IMAGES_FOLDER = os.path.join(BASE_DIR, config['paths']['db_images'])
EFFICIENCY_TARGET = 40 

# ================= 🛠️ INITIALIZATION =================

def init_system():
    """เตรียมระบบให้พร้อมสำหรับการรันครั้งแรก"""
    for path_key in ['db_pkl_folder', 'models_folder', 'db_images']:
        os.makedirs(config['paths'][path_key], exist_ok=True)
    
    pkl_path = os.path.join(BASE_DIR, FILES['pack_vec'])
    if not os.path.exists(pkl_path):
        with open(pkl_path, 'wb') as f:
            pickle.dump({}, f)

init_system()

# ================= 🦕 CORE LOGIC =================

def load_pkl(path):
    local_path = os.path.join(BASE_DIR, path)
    if os.path.exists(local_path):
        with open(local_path, 'rb') as f: return pickle.load(f)
    return {}

def get_unique_drugs_from_pkl(packs_db):
    """สกัดรายชื่อยาจาก Keys ใน pkl เท่านั้น (Hardmode)"""
    names = set()
    for k in packs_db.keys():
        if "_pack" in k:
            clean_name = k.split('_pack')[0]
            names.add(clean_name)
    return sorted(list(names))

def create_drug_list_json(drug_names):
    """สร้างไฟล์ JSON สรุป Metadata ทั้งหมด"""
    metadata = {
        "drugs": drug_names,
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total": len(drug_names),
        "status": "production"
    }
    json_path = os.path.join(BASE_DIR, DRUG_LIST_FILE)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)
    return json_path

def check_s3_connection():
    if not BUCKET_NAME: return False, "🔴 Config Missing"
    try:
        S3_CLIENT.get_bucket_location(Bucket=BUCKET_NAME)
        return True, "🟢 Ready"
    except Exception:
        return False, "🔴 Offline/Access Denied"

def get_s3_assets():
    try:
        res = S3_CLIENT.list_objects_v2(Bucket=BUCKET_NAME, Prefix='latest/')
        return [obj['Key'].replace('latest/', '') for obj in res.get('Contents', []) if obj['Key'] != 'latest/']
    except: return []

@st.cache_resource
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    dino.eval().to(device)
    try:
        yolo = YOLO(FILES['model'])
    except:
        yolo = YOLO('yolov8n-seg.pt')
    preprocess = transforms.Compose([
        transforms.Resize((DINO_SIZE, DINO_SIZE), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return dino, yolo, preprocess, device, "🟢 GPU" if device.type == "cuda" else "🟡 CPU"

dino_model, yolo_model, preprocess_pipeline, device, device_status = load_models()

# ================= 🖥️ UI PREPARATION =================
st.set_page_config(page_title="PillTrack Senior Registry", layout="wide")

# Source of Truth: เคลียร์แคชแล้วโหลด pkl สดๆ ทุกครั้งที่ Refresh
packs_db = load_pkl(FILES['pack_vec'])
current_drugs = get_unique_drugs_from_pkl(packs_db)

if 'session_updates' not in st.session_state:
    st.session_state.session_updates = []
if 'last_crops' not in st.session_state:
    st.session_state.last_crops = []

# ================= 🖥️ UI LAYOUT =================
st.title("PillTrack: Pack Dataset Manager")

# Sidebar
st.sidebar.header("System Status")
s3_ok, s3_msg = check_s3_connection()
st.sidebar.write(f"S3 Connection: {s3_msg}")
st.sidebar.write(f"Compute: {device_status}")
st.sidebar.markdown("---")
if st.sidebar.button("FORCE REFRESH SYSTEM"):
    st.cache_resource.clear()
    st.rerun()

# Dashboard Summary
db_stats = []
for name in current_drugs:
    count = sum([len(v) for k, v in packs_db.items() if k.startswith(f"{name}_pack")])
    db_stats.append({'name': name, 'count': count})

m1, m2, m3 = st.columns(3)
m1.metric("Local Packs", len(db_stats))
m2.metric("Total Vectors", sum([s['count'] for s in db_stats]))
m3.metric("Cloud Status", "Ready" if s3_ok else "Offline")

col_l, col_r = st.columns(2)
with col_l:
    st.subheader("🟢 Local Efficiency")
    with st.container(border=True):
        if db_stats:
            for s in db_stats:
                eff = min(s['count'] / EFFICIENCY_TARGET, 1.0)
                icon = "🟢" if eff >= 1.0 else "🟡" if eff > 0.5 else "🔴"
                st.write(f"{icon} **{s['name'].upper()}**")
                st.progress(eff)
        else: st.caption("No data in .pkl. Add your first pack below.")

with col_r:
    st.subheader("🟢 Cloud S3 Inventory")
    with st.container(border=True):
        if s3_ok:
            for f in get_s3_assets(): st.code(f, language=None)
        else: st.error("S3 Permission Error")

st.divider()

# --- ACTION SECTION ---
st.subheader("🛠️ Dataset Management")
# อยู่นอก Form เพื่อให้ Dropdown สลับทันทีที่คลิก
mode = st.radio("Action Mode:", ["New Pack", "Enhance Existing"], horizontal=True)

with st.form("main_update_form", clear_on_submit=True):
    if mode == "New Pack":
        name_in = st.text_input("New Drug Name:").strip().lower()
    else:
        # ดึงจาก pkl มาทำ Dropdown เท่านั้น
        if current_drugs:
            name_in = st.selectbox("Select Existing Drug:", current_drugs)
        else:
            st.warning("⚠️ No drugs in .pkl. Use 'New Pack' first.")
            name_in = None
    
    files_in = st.file_uploader("Upload Samples:", accept_multiple_files=True)
    show_yolo = st.checkbox("Show YOLO previews", value=True)
    
    if st.form_submit_button("PROCESS & SAVE", use_container_width=True):
        if name_in and files_in:
            folder_name = f"{name_in}_pack"
            save_p = os.path.join(DB_IMAGES_FOLDER, folder_name)
            
            if mode == "New Pack":
                if os.path.exists(save_p): shutil.rmtree(save_p)
                for k in [k for k in packs_db.keys() if k.startswith(folder_name)]: del packs_db[k]
            
            os.makedirs(save_p, exist_ok=True)
            st.session_state.last_crops = []
            p_bar = st.progress(0)
            
            for i, file in enumerate(files_in):
                p_bar.progress((i+1)/len(files_in))
                img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), 1)
                res = yolo_model(img, verbose=False, conf=config['settings']['yolo_conf'])
                crop = img
                if len(res[0].boxes) > 0:
                    b = sorted(res[0].boxes, key=lambda x: x.conf, reverse=True)[0]
                    x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().astype(int)
                    crop = img[max(0, y1-20):min(img.shape[0], y2+20), max(0, x1-20):min(img.shape[1], x2+20)]
                    if show_yolo: st.session_state.last_crops.append(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                
                for angle, suffix in [(0,"_rot0"),(90,"_rot90"),(180,"_rot180"),(270,"_rot270")]:
                    r_img = crop.copy()
                    if angle == 90: r_img = cv2.rotate(r_img, cv2.ROTATE_90_CLOCKWISE)
                    elif angle == 180: r_img = cv2.rotate(r_img, cv2.ROTATE_180)
                    elif angle == 270: r_img = cv2.rotate(r_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    
                    t = preprocess_pipeline(Image.fromarray(cv2.cvtColor(r_img, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(device)
                    with torch.no_grad():
                        vec = dino_model(t).flatten().cpu().numpy()
                        vec = vec / (np.linalg.norm(vec) + 1e-8)
                    
                    f_key = f"{folder_name}{suffix}"
                    if f_key not in packs_db: packs_db[f_key] = []
                    packs_db[f_key].append(vec)

            # บันทึก pkl
            with open(os.path.join(BASE_DIR, FILES['pack_vec']), 'wb') as f:
                pickle.dump(packs_db, f)
            
            st.session_state.session_updates.append(f"{name_in.upper()} ({mode})")
            st.success(f"🟢 {name_in.upper()} Local Update Complete!")
            st.rerun()

if st.session_state.last_crops:
    with st.expander("🔍 YOLO Preview", expanded=True):
        st.image(st.session_state.last_crops[:12], width=110)

# --- PRODUCTION PUSH SECTION ---
st.divider()
c_p1, c_p2 = st.columns([1.5, 1])
with c_p2:
    st.subheader("🚀 Production Push")
    if st.session_state.session_updates:
        st.warning("🟡 Pending push:")
        for up in st.session_state.session_updates: st.caption(f"- {up}")
        
    if st.button("PUSH ALL TO S3", type="primary", use_container_width=True, disabled=not s3_ok):
        with st.status("Generating Metadata & Syncing...", expanded=True) as status:
            try:
                # 1. สร้าง Metadata JSON ก่อน Push
                latest_drugs = get_unique_drugs_from_pkl(packs_db)
                create_drug_list_json(latest_drugs)
                
                # 2. Push ทุกอย่าง (incl. drug_list.json)
                for key, filename in FILES.items():
                    local_p = os.path.join(BASE_DIR, filename)
                    if os.path.exists(local_p):
                        st.write(f"Uploading: {filename}...")
                        S3_CLIENT.upload_file(local_p, BUCKET_NAME, f"latest/{filename}")
                
                st.toast("🟢 Cloud Status: Synchronized!", icon="✅")
                st.session_state.session_updates = []
                st.rerun()
            except Exception as e:
                st.error(f"🔴 Push Failed: {str(e)}")