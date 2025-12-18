import streamlit as st
import yaml, os, cv2, shutil, glob, zipfile
import numpy as np
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from cloud_manager import CloudManager
from db_manager import DBManager
from engine import AIEngine

# ================= ⚙️ 1. ROBUST CONFIG LOAD =================
load_dotenv()

def load_config():
    """โหลด Config อย่างระมัดระวัง ถ้าพังให้คืน dict ว่าง"""
    if os.path.exists("config.yaml"):
        with open("config.yaml", "r") as f:
            return yaml.safe_load(f) or {}
    return {}

config = load_config()

# ดึงค่าแบบ Safe Mode (ป้องกัน KeyError) [cite: 2025-12-05]
ARTIFACTS = config.get('artifacts', {})
PATHS = config.get('paths', {})
SETTINGS = config.get('settings', {})

# กำหนด Path หลัก (มี Default กันตาย)
PKL_PATH = ARTIFACTS.get('pack_vec', 'database/db_packs_dino.pkl')
JSON_PATH = ARTIFACTS.get('drug_list', 'database/drug_list.json')
MODEL_PATH = ARTIFACTS.get('model', 'models/seg_best_process.pt')
IMG_DB_ROOT = PATHS.get('db_images', 'database_images')

# Settings AI
DINO_SIZE = SETTINGS.get('dino_size', 224)
YOLO_CONF = SETTINGS.get('yolo_conf', 0.75)
EFFICIENCY_TARGET = SETTINGS.get('efficiency_target', 40)

# ================= 🚀 2. APP INITIALIZATION =================
st.set_page_config(page_title="PillTrack Ops Hub", layout="wide")

cloud = CloudManager(os.getenv('S3_BUCKET_NAME'))
db = DBManager()

if "push_success_msg" in st.session_state:
    st.toast(st.session_state.push_success_msg, icon="✅")
    del st.session_state.push_success_msg 

@st.cache_resource
def get_engine():
    return AIEngine(MODEL_PATH, DINO_SIZE)

engine = get_engine()

# โหลดข้อมูล
packs_db = db.load_pkl(PKL_PATH)
current_drugs = db.get_unique_drugs(packs_db)

# ================= 🖥️ 3. UI DASHBOARD =================
st.title("💊 PillTrack: MLOps Producer Hub")

# --- Sidebar ---
st.sidebar.header("Operations Status")
s3_ok, s3_status = cloud.check_connection()
st.sidebar.write(f"Cloud S3: {s3_status}")
st.sidebar.write(f"Compute: {engine.device}")

st.sidebar.markdown("---")
with st.sidebar.expander("🕒 View Activity Logs", expanded=False):
    recent_logs = db.get_logs()
    if recent_logs:
        st.dataframe(pd.DataFrame(recent_logs), use_container_width=True, hide_index=True)

if st.sidebar.button("FORCE REFRESH SYSTEM"):
    st.cache_resource.clear()
    st.rerun()

# --- Metrics ---
m1, m2, m3 = st.columns(3)
m1.metric("Local Classes", len(current_drugs))
m2.metric("Total Vectors", sum([len(v) for v in packs_db.values()]))
m3.metric("Cloud Status", "Ready" if s3_ok else "Disconnected")

# --- Efficiency & Inventory ---
col_l, col_r = st.columns(2)
with col_l:
    st.subheader("🟢 Local Efficiency")
    with st.container(border=True):
        if current_drugs:
            for name in current_drugs:
                count = sum([len(v) for k, v in packs_db.items() if k.startswith(f"{name}_pack")])
                eff = min(count / EFFICIENCY_TARGET, 1.0)
                st.caption(f"{name.upper()} ({count} vecs)")
                st.progress(eff)
        else: st.caption("No data in database.")

with col_r:
    st.subheader("🟢 Cloud Inventory")
    with st.container(border=True):
        if s3_ok:
            for f in cloud.get_inventory(): st.code(f, language=None)
        else: st.error("S3 connection error.")

st.divider()

# ================= 🛠️ 4. DATASET MANAGEMENT (ZIP MODE) =================
st.subheader("📦 Dataset Management")
mode = st.radio("Action Mode:", ["New Pack", "Enhance Existing", "Bulk Import (Zip File)"], horizontal=True)

with st.form("update_form", clear_on_submit=False):
    drug_list_to_process = [] # List เก็บ tuple (ชื่อยา, [ไฟล์รูป])
    temp_extract_dir = "temp_unzip_area"
    
    # --- Mode Selection Logic ---
    if mode == "New Pack":
        name_in = st.text_input("Enter New Drug Name:").strip().lower()
        files_in = st.file_uploader("Upload Samples:", accept_multiple_files=True)
        if name_in and files_in: drug_list_to_process = [(name_in, files_in)]
    
    elif mode == "Enhance Existing":
        name_in = st.selectbox("Select Existing Drug:", current_drugs) if current_drugs else None
        files_in = st.file_uploader("Upload Samples:", accept_multiple_files=True)
        if name_in and files_in: drug_list_to_process = [(name_in, files_in)]
    
    elif mode == "Bulk Import (Zip File)":
        uploaded_zip = st.file_uploader("Upload .zip (Structure: DrugName/Images)", type="zip")
        st.caption("Tip: ระบบจะใช้ชื่อโฟลเดอร์ใน Zip เป็นชื่อยาอัตโนมัติ")

    show_yolo = st.checkbox("Show AI Segmentation Preview", value=True)
    
    # --- Button Logic ---
    if st.form_submit_button("🚀 PROCESS & SAVE LOCAL", use_container_width=True):
        
        # 1. Unzip Processing
        if mode == "Bulk Import (Zip File)" and uploaded_zip:
            if os.path.exists(temp_extract_dir): shutil.rmtree(temp_extract_dir)
            os.makedirs(temp_extract_dir, exist_ok=True)
            
            with st.spinner("Unzipping & Scanning..."):
                with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
                    zip_ref.extractall(temp_extract_dir)
                
                # Walk หาโฟลเดอร์ยา
                for root, dirs, files in os.walk(temp_extract_dir):
                    if "__MACOSX" in root: continue # ข้ามไฟล์ขยะ Mac
                    
                    images = [os.path.join(root, f) for f in files if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
                    if images:
                        drug_name = os.path.basename(root).lower().strip()
                        if drug_name:
                            drug_list_to_process.append((drug_name, images))

        # 2. Main Processing Loop
        if drug_list_to_process:
            st.session_state.last_crops = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, (d_name, d_items) in enumerate(drug_list_to_process):
                status_text.write(f"⚙️ Processing **{d_name.upper()}** ({len(d_items)} images)...")
                folder_name = f"{d_name}_pack"
                img_save_dir = os.path.join(IMG_DB_ROOT, d_name)
                
                # Clear old data (New/Bulk mode)
                if mode in ["New Pack", "Bulk Import (Zip File)"]:
                    if os.path.exists(img_save_dir): shutil.rmtree(img_save_dir)
                    for k in list(packs_db.keys()):
                        if k.startswith(folder_name): del packs_db[k]
                
                os.makedirs(img_save_dir, exist_ok=True)
                
                for i, item in enumerate(d_items):
                    # Load Image
                    if isinstance(item, str): 
                        img = cv2.imread(item)
                    else: 
                        img = cv2.imdecode(np.frombuffer(item.read(), np.uint8), 1)
                    
                    if img is None: continue

                    # Save Original
                    img_name = f"{d_name}_{datetime.now().strftime('%H%M%S')}_{i}.jpg"
                    cv2.imwrite(os.path.join(img_save_dir, img_name), img)
                    
                    # AI Processing
                    crop = engine.detect_and_crop(img, YOLO_CONF)
                    if show_yolo and len(st.session_state.last_crops) < 12:
                        st.session_state.last_crops.append(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                    
                    # 4-Angle Vector Extraction
                    for angle, suffix in [(0,"_rot0"),(90,"_rot90"),(180,"_rot180"),(270,"_rot270")]:
                        rot = crop.copy()
                        if angle == 90: rot = cv2.rotate(rot, cv2.ROTATE_90_CLOCKWISE)
                        elif angle == 180: rot = cv2.rotate(rot, cv2.ROTATE_180)
                        elif angle == 270: rot = cv2.rotate(rot, cv2.ROTATE_90_COUNTERCLOCKWISE)
                        
                        vec = engine.extract_vector(rot)
                        f_key = f"{folder_name}{suffix}"
                        if f_key not in packs_db: packs_db[f_key] = []
                        packs_db[f_key].append(vec)
                
                progress_bar.progress((idx + 1) / len(drug_list_to_process))

            # Cleanup & Save
            if os.path.exists(temp_extract_dir): shutil.rmtree(temp_extract_dir)
            db.save_pkl(packs_db, PKL_PATH)
            db.add_log(f"BATCH_{mode}", f"{len(drug_list_to_process)} drugs", 0)
            
            st.success(f"✅ Processed {len(drug_list_to_process)} drugs successfully!")
            st.rerun()
        else:
            if mode == "Bulk Import (Zip File)":
                st.warning("⚠️ Zip File is empty or has invalid structure.")
            else:
                st.warning("⚠️ Please upload files first.")

if "last_crops" in st.session_state and st.session_state.last_crops:
    with st.expander("🔍 AI Detection Preview", expanded=True):
        st.image(st.session_state.last_crops, width=110)

# ================= 🚀 5. RELEASE MANAGEMENT (FIXED PUSH) =================
st.divider()
c_p1, c_p2 = st.columns([2, 1])
with c_p2:
    st.subheader("Release to Cloud")
    if st.button("PUSH ALL ARTIFACTS TO S3", type="primary", use_container_width=True, disabled=not s3_ok):
        with st.status("Syncing Production Registry...", expanded=True) as status:
            try:
                # 1. Update Metadata (Timestamp will update here)
                latest_list = db.get_unique_drugs(packs_db)
                db.generate_metadata(latest_list, JSON_PATH)
                
                # 🔥 FIX: Force Push drug_list.json ก่อนเสมอ [cite: 2025-12-18]
                if os.path.exists(JSON_PATH):
                    status.write(f"Uploading Metadata: {JSON_PATH}...")
                    cloud.upload_file(JSON_PATH, f"latest/{JSON_PATH}")
                
                # 2. Push Artifacts in Config
                for k, path in ARTIFACTS.items():
                    # Skip drug_list if already uploaded
                    if path == JSON_PATH: continue
                    
                    if os.path.exists(path):
                        status.write(f"Uploading Artifact: {path}...")
                        cloud.upload_file(path, f"latest/{path}")
                    else:
                        status.write(f"⚠️ Skipped missing: {path}")
                
                st.session_state.push_success_msg = f"Synced {len(latest_list)} drugs to Cloud!"
                st.rerun()
            except Exception as e:
                st.error(f"🔴 Pipeline Error: {str(e)}")

st.sidebar.caption(f"PillTrack MLOps | {datetime.now().strftime('%Y-%m-%d %H:%M')}")