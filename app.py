import streamlit as st
import yaml, os, cv2, shutil, glob, zipfile
import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict 
from dotenv import load_dotenv
from cloud_manager import CloudManager
from db_manager import DBManager
from engine import AIEngine

# ================= 1. SETUP & CONFIGURATION =================
load_dotenv()
S3_BUCKET = os.getenv('S3_BUCKET_NAME')

def load_config():
    """Load configuration from YAML file safely."""
    if os.path.exists("config.yaml"):
        with open("config.yaml", "r") as f:
            return yaml.safe_load(f) or {}
    return {}

config = load_config()

# Safe Extraction of Artifacts, Paths, and Settings
ARTIFACTS = config.get('artifacts', {})
PATHS = config.get('paths', {})
SETTINGS = config.get('settings', {})

PKL_PATH = ARTIFACTS.get('pack_vec', 'database/pill_fingerprints.pkl') # แนะนำให้เปลี่ยนชื่อไฟล์ถ้าอยากแยก version
JSON_PATH = ARTIFACTS.get('drug_list', 'database/drug_list.json')
MODEL_PATH = ARTIFACTS.get('model', 'models/seg_best_process.pt')
IMG_DB_ROOT = PATHS.get('db_images', 'database_images')

DINO_SIZE = SETTINGS.get('dino_size', 224)
YOLO_CONF = SETTINGS.get('yolo_conf', 0.75)
EFFICIENCY_TARGET = SETTINGS.get('efficiency_target', 40)

# ================= 2. INITIALIZATION =================
st.set_page_config(page_title="PillTrack Ops Hub", layout="wide")

cloud = CloudManager(S3_BUCKET)
db = DBManager()

if "push_success_msg" in st.session_state:
    st.toast(st.session_state.push_success_msg, icon="✅")
    del st.session_state.push_success_msg 

@st.cache_resource
def get_engine():
    """Initialize AI Engine (YOLO-Seg + DINOv2 + SIFT)."""
    return AIEngine(MODEL_PATH, DINO_SIZE)

engine = get_engine()
packs_db = db.load_pkl(PKL_PATH)
current_drugs = db.get_unique_drugs(packs_db)

# Helper Function: นับจำนวนภาพไม่ว่าจะเป็น Data เก่า (List) หรือใหม่ (Dict)
def count_samples(val):
    if isinstance(val, dict):
        return len(val.get('dino', []))
    elif isinstance(val, list):
        return len(val)
    return 0

# ================= 3. UI DASHBOARD =================
st.title("PillTrack: MLOps Producer Hub (Hybrid Engine)")

# --- Sidebar Status and Logs ---
st.sidebar.header("Operations Status")
s3_ok, s3_status = cloud.check_connection()
st.sidebar.write(f"Cloud S3: {s3_status}")
st.sidebar.write(f"Compute Device: {engine.device}")

st.sidebar.markdown("---")
with st.sidebar.expander("🕒 View Activity Logs", expanded=False):
    recent_logs = db.get_logs()
    if recent_logs:
        st.dataframe(pd.DataFrame(recent_logs), width="stretch", hide_index=True)

if st.sidebar.button("FORCE REFRESH SYSTEM"):
    st.cache_resource.clear()
    st.rerun()

# --- Metrics Section ---
total_vectors = sum([count_samples(v) for v in packs_db.values()])

m1, m2, m3 = st.columns(3)
m1.metric("Local Classes", len(current_drugs))
m2.metric("Total Vectors (DINO+SIFT)", total_vectors)
m3.metric("Cloud Sync", "Ready" if s3_ok else "Disconnected")

col_l, col_r = st.columns(2)
with col_l:
    st.subheader("Local Efficiency")
    with st.container(border=True):
        if current_drugs:
            for name in current_drugs:
                # แก้ไข Logic นับจำนวนให้รองรับ Data Structure ใหม่
                count = sum([count_samples(v) for k, v in packs_db.items() if k.startswith(f"{name}_pack")])
                
                eff = min(count / EFFICIENCY_TARGET, 1.0)
                st.caption(f"{name.upper()} ({count} samples)")
                st.progress(eff)
        else:
            st.info("No drugs registered in the local database.")

with col_r:
    st.subheader("Cloud Inventory (latest/)")
    with st.container(border=True):
        if s3_ok:
            for f in cloud.get_inventory():
                st.code(f, language=None)
        else:
            st.error("Unable to fetch cloud inventory.")

st.divider()

# =================  4. DATASET MANAGEMENT =================
st.subheader("Dataset Management")
mode = st.radio("Action Mode:", ["New Pack", "Enhance Existing", "Bulk Import (Zip)"], horizontal=True)

with st.form("update_form", clear_on_submit=False):
    drug_map = defaultdict(list)
    temp_unzip_area = "temp_batch_process"
    
    if mode == "New Pack":
        name_in = st.text_input("Enter New Drug Name:").strip().lower()
        files_in = st.file_uploader("Upload Samples:", accept_multiple_files=True)
        if name_in and files_in:
            drug_map[name_in].extend(files_in)
    
    elif mode == "Enhance Existing":
        name_in = st.selectbox("Select Existing Drug:", current_drugs) if current_drugs else None
        files_in = st.file_uploader("Upload Samples:", accept_multiple_files=True)
        if name_in and files_in:
            drug_map[name_in].extend(files_in)
    
    elif mode == "Bulk Import (Zip)":
        uploaded_zip = st.file_uploader("Upload Zip File (Structure: DrugName/Images)", type="zip")
        if uploaded_zip:
            if os.path.exists(temp_unzip_area): shutil.rmtree(temp_unzip_area)
            os.makedirs(temp_unzip_area, exist_ok=True)
            
            with zipfile.ZipFile(uploaded_zip, 'r') as z:
                z.extractall(temp_unzip_area)
            
            for root, dirs, files in os.walk(temp_unzip_area):
                if "__MACOSX" in root: continue
                imgs = [os.path.join(root, f) for f in files if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
                if imgs:
                    d_name = os.path.basename(root).lower().strip()
                    drug_map[d_name].extend(imgs)

    show_preview = st.checkbox("Show AI Segmentation Preview", value=True)
    
    if st.form_submit_button("PROCESS & SAVE LOCAL", width="stretch"):
        # Convert map to list for processing
        drug_list_to_process = list(drug_map.items())
        
        if drug_list_to_process:
            st.session_state.last_crops = []
            
            total_imgs = sum([len(imgs) for _, imgs in drug_list_to_process])
            current_count = 0
            
            progress_bar = st.progress(0, text="Initializing...")
            
            with st.status(" Starting Batch Processing...", expanded=True) as status:
                
                for d_name, d_items in drug_list_to_process:
                    status.write(f"📂 **Processing Class: {d_name.upper()}** ({len(d_items)} images)")
                    folder_name = f"{d_name}_pack"
                    img_save_dir = os.path.join(IMG_DB_ROOT, d_name)
                    
                    # Logic: Clear old data ONLY ONCE per drug name
                    if mode in ["New Pack", "Bulk Import (Zip)"]:
                        if os.path.exists(img_save_dir): shutil.rmtree(img_save_dir)
                        # ลบข้อมูลเก่าทิ้ง (ไม่ว่าจะเป็น struct เก่าหรือใหม่)
                        for k in list(packs_db.keys()):
                            if k.startswith(folder_name): del packs_db[k]
                    
                    os.makedirs(img_save_dir, exist_ok=True)
                    
                    for i, item in enumerate(d_items):
                        current_count += 1
                        
                        progress_percentage = min(current_count / total_imgs, 1.0)
                        progress_bar.progress(progress_percentage, text=f"Processing {current_count}/{total_imgs}: {d_name}")
                        
                        # Load Image
                        img = cv2.imread(item) if isinstance(item, str) else cv2.imdecode(np.frombuffer(item.read(), np.uint8), 1)
                        if img is None: continue
                        
                        # Save
                        save_name = f"{d_name}_{i}_{datetime.now().strftime('%H%M%S')}.jpg"
                        cv2.imwrite(os.path.join(img_save_dir, save_name), img)
                        
                        # AI Engine
                        crop = engine.detect_and_crop(img, YOLO_CONF)
                        if show_preview and len(st.session_state.last_crops) < 12:
                            st.session_state.last_crops.append(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                        
                        # --- [CRITICAL CHANGE] Extract Features (DINO + SIFT) ---
                        for angle, suffix in [(0,"_rot0"),(90,"_rot90"),(180,"_rot180"),(270,"_rot270")]:
                            if angle == 90: rot = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
                            elif angle == 180: rot = cv2.rotate(crop, cv2.ROTATE_180)
                            elif angle == 270: rot = cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)
                            else: rot = crop.copy()
                            
                            # 1. เรียกใช้ฟังก์ชันใหม่ที่คืนค่ามา 2 ตัว
                            dino_vec, sift_des = engine.extract_features(rot)
                            
                            # 2. ใช้ DBManager จัดการ Insert ข้อมูลลง Structure ใหม่
                            f_key = f"{folder_name}{suffix}"
                            db.insert_data(packs_db, f_key, dino_vec, sift_des)

                    status.write(f"✨ Class **{d_name}** complete.")
                
                status.update(label="✅ Batch Processing Complete!", state="complete", expanded=False)

            if os.path.exists(temp_unzip_area): shutil.rmtree(temp_unzip_area)
            
            # Save Database (.pkl)
            db.save_pkl(packs_db, PKL_PATH)
            db.add_log("INGESTION", f"{len(drug_list_to_process)} drugs", total_imgs)
            st.success(f"Successfully processed {total_imgs} images (Hybrid DINO+SIFT).")
            st.rerun()

if "last_crops" in st.session_state and st.session_state.last_crops:
    with st.expander("AI Detection Preview", expanded=True):
        st.image(st.session_state.last_crops, width=110)

# ================= 5. EDGE FEEDBACK LOOP =================
st.divider()
st.subheader("Edge Data Feedback Loop")
c_e1, c_e2 = st.columns([2, 1])
with c_e1:
    st.write("Retrieve the latest captured data zip from Raspberry Pi. Files are stored in `edge_data_inbox`.")
    local_inbox = "edge_data_inbox"

with c_e2:
    if st.button("PULL LATEST DATA FROM EDGE", width="stretch"):
        with st.status("Fetching latest data from S3...", expanded=True) as status:
            try:
                prefix = "data_collection/"
                objects = cloud.s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=prefix)
                
                if 'Contents' not in objects:
                    st.warning("No new data available in S3 `data_collection/`")
                else:
                    latest = sorted(objects['Contents'], key=lambda x: x['LastModified'], reverse=True)[0]
                    s3_key = latest['Key']
                    zip_name = os.path.basename(s3_key)
                    local_zip = os.path.join(local_inbox, zip_name)
                    
                    os.makedirs(local_inbox, exist_ok=True)
                    status.write(f"Downloading: {zip_name}")
                    cloud.s3.download_file(S3_BUCKET, s3_key, local_zip)
                    
                    st.success(f"Pull Complete! Zip file saved to: `{local_zip}`")
                    db.add_log("EDGE_PULL", zip_name, 0, f"Saved to {local_inbox}")
            except Exception as e:
                st.error(f"Error during pull: {str(e)}")

# ================= 6. RELEASE MANAGEMENT =================
st.divider()
c_p1, c_p2 = st.columns([2, 1])
with c_p2:
    st.subheader("Release to Production")
    if st.button("PUSH ALL ARTIFACTS TO S3", type="primary", width="stretch", disabled=not s3_ok):
        with st.status("Synchronizing Cloud Production...", expanded=True) as status:
            try:
                latest_list = db.get_unique_drugs(packs_db)
                db.generate_metadata(latest_list, JSON_PATH)
                
                if os.path.exists(JSON_PATH):
                    status.write(f"Uploading Metadata: {JSON_PATH}")
                    cloud.upload_file(JSON_PATH, f"latest/{JSON_PATH}")
                
                for k, path in ARTIFACTS.items():
                    # อัปโหลดไฟล์ PKL (ซึ่งตอนนี้ข้างในมี SIFT แล้ว)
                    if path == JSON_PATH: continue
                    if os.path.exists(path):
                        status.write(f"Uploading Artifact: {path}")
                        cloud.upload_file(path, f"latest/{path}")
                
                st.session_state.push_success_msg = f"Successfully synced {len(latest_list)} drugs to Cloud."
                st.rerun()
            except Exception as e:
                st.error(f"Release Error: {str(e)}")

st.sidebar.caption(f"PillTrack MLOps | {datetime.now().strftime('%Y-%m-%d %H:%M')}")