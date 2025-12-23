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

# Safe Extraction from YAML (Strictly use config.yaml)
ARTIFACTS = config.get('artifacts', {})
PATHS = config.get('paths', {})
SETTINGS = config.get('settings', {})

PKL_PATH = ARTIFACTS.get('pack_vec', 'database/pill_fingerprints.pkl')
JSON_PATH = ARTIFACTS.get('drug_list', 'database/drug_list.json')
MODEL_PATH = ARTIFACTS.get('model', 'models/seg_best_process.pt')
IMG_DB_ROOT = PATHS.get('db_images', 'database_images')

# ใช้ค่าจาก Config เท่านั้น
DINO_SIZE = SETTINGS.get('dino_size', 224)
YOLO_CONF = SETTINGS.get('yolo_conf', 0.5) 
EFFICIENCY_TARGET = SETTINGS.get('efficiency_target', 10)

# ================= 2. INITIALIZATION =================
st.set_page_config(page_title="PillTrack Ops Hub", layout="wide")

cloud = CloudManager(S3_BUCKET)
db = DBManager()

# [ALERT SYSTEM] แจ้งเตือนแบบไม่หายเมื่อ Refresh
if "push_success_msg" in st.session_state:
    st.success(st.session_state.push_success_msg, icon="✅")
    del st.session_state.push_success_msg 

@st.cache_resource
def get_engine():
    """Initialize AI Engine."""
    return AIEngine(MODEL_PATH, DINO_SIZE)

engine = get_engine()
packs_db = db.load_pkl(PKL_PATH)

def get_base_drug_names(db_dict):
    names = set()
    for k in db_dict.keys():
        base = k.split('_box')[0].split('_blister')[0].split('_pack')[0] 
        base = base.split('_rot')[0] 
        names.add(base)
    return sorted(list(names))

current_drugs = get_base_drug_names(packs_db)

def count_samples(val):
    if isinstance(val, dict): return len(val.get('dino', []))
    elif isinstance(val, list): return len(val)
    return 0

# ================= 3. UI DASHBOARD (SIMPLIFIED) =================
st.title("PillTrack: MLOps Producer Hub")

# --- Sidebar ---
st.sidebar.header("System Status")
s3_ok, s3_status = cloud.check_connection()
st.sidebar.write(f"☁️ Cloud S3: {s3_status}")
st.sidebar.write(f"💻 Device: {engine.device}")
st.sidebar.write(f"⚙️ YOLO Conf: {YOLO_CONF}")
st.sidebar.write(f"⚙️ DINO Size: {DINO_SIZE}")

if st.sidebar.button("🔄 FORCE REFRESH SYSTEM"):
    st.cache_resource.clear()
    st.rerun()

# --- Basic Metrics ---
total_vectors = sum([count_samples(v) for v in packs_db.values()])
m1, m2, m3 = st.columns(3)
m1.metric("Unique Drugs (SKUs)", len(current_drugs))
m2.metric("Total Embeddings", total_vectors)
m3.metric("Cloud Sync", "Ready" if s3_ok else "Disconnected")

st.divider()

# =================  4. DATASET MANAGEMENT (UNIFIED FORM) =================
st.subheader("Dataset Operations")

# เลือกโหมด
mode = st.radio(
    "Select Action:", 
    ["Add New SKU / Update Existing", "Bulk Import (Zip)", "❌ Delete / Cleanup Mistake"], 
    horizontal=True,
    key="dataset_mode_select"
)

# เริ่ม Form เดียว คุมทุกอย่าง
with st.form("dataset_ops_form", clear_on_submit=False):
    
    # ---------------- UI INPUTS ----------------
    drug_name_input = ""
    pack_type = "Blister (แผง)"
    files_in = []
    target_to_delete = None
    uploaded_zip = None

    # UI: Mode 1 - Add/Update
    if mode == "Add New SKU / Update Existing":
        st.info("ℹ️ Upload clear images of Box or Blister.")
        c1, c2 = st.columns(2)
        with c1:
            input_method = st.radio("Name Source:", ["Select Existing", "Type New"], horizontal=True, label_visibility="collapsed")
            if input_method == "Select Existing" and current_drugs:
                drug_name_input = st.selectbox("Select Drug:", current_drugs)
            else:
                drug_name_input = st.text_input("New Drug Name (English):", placeholder="e.g. paracetamol").strip().lower()
        with c2:
            pack_type = st.selectbox("Product Type:", ["Blister (แผง)", "Box (กล่อง)"])
            
        files_in = st.file_uploader(f"Upload Images for {pack_type}:", accept_multiple_files=True)
        st.markdown("---")
        show_preview = st.checkbox("👁️ Show YOLO Preview", value=True)

    # UI: Mode 2 - Zip
    elif mode == "Bulk Import (Zip)":
        st.info("ℹ️ Upload zip containing folders named 'drug_box' or 'drug_blister'.")
        uploaded_zip = st.file_uploader("Upload Zip File", type="zip")
        st.markdown("---")
        show_preview = st.checkbox("👁️ Show YOLO Preview (Random)", value=True)

    # UI: Mode 3 - Delete
    elif mode == "❌ Delete / Cleanup Mistake":
        st.warning("⚠️ Permantently delete data from Disk and Memory.")
        all_keys = set([k.split('_rot')[0] for k in packs_db.keys()])
        target_to_delete = st.selectbox("Select Class to Delete:", sorted(list(all_keys)))

    # ---------------- SUBMIT BUTTON ----------------
    st.write("")
    btn_label = "🗑️ DELETE NOW" if "Delete" in mode else "🚀 PROCESS & SAVE"
    btn_type = "primary"
    
    submit_btn = st.form_submit_button(btn_label, type=btn_type, use_container_width=True)

    # ---------------- PROCESS LOGIC ----------------
    if submit_btn:
        
        # >>> CASE 1: DELETE <<<
        if mode == "❌ Delete / Cleanup Mistake":
            if target_to_delete:
                keys_to_del = [k for k in packs_db.keys() if k.startswith(target_to_delete)]
                for k in keys_to_del: del packs_db[k]
                
                path = os.path.join(IMG_DB_ROOT, target_to_delete)
                if os.path.exists(path): shutil.rmtree(path)
                
                db.save_pkl(packs_db, PKL_PATH)
                
                # Auto-Update JSON
                current_list = get_base_drug_names(packs_db)
                db.generate_metadata(current_list, JSON_PATH)
                
                st.session_state.push_success_msg = f"🗑️ Deleted {target_to_delete} and updated JSON list."
                st.rerun()
            else:
                st.error("Please select a class.")

        # >>> CASE 2: ADD / UPDATE <<<
        elif mode == "Add New SKU / Update Existing":
            if not drug_name_input or not files_in:
                st.error("❌ Missing Name or Images!")
            else:
                st.session_state.last_crops = []
                type_suffix = "blister" if "Blister" in pack_type else "box"
                final_class_name = f"{drug_name_input}_{type_suffix}"
                
                # Setup Dirs & Clear Old
                save_dir = os.path.join(IMG_DB_ROOT, final_class_name)
                if os.path.exists(save_dir): shutil.rmtree(save_dir)
                os.makedirs(save_dir, exist_ok=True)
                
                keys_to_del = [k for k in packs_db.keys() if k.startswith(final_class_name)]
                for k in keys_to_del: del packs_db[k]
                
                total = len(files_in)
                progress = st.progress(0)
                
                with st.status(f"⚡ Processing **{final_class_name}**...", expanded=True) as status:
                    for i, item in enumerate(files_in):
                        status.write(f"Processing image {i+1}/{total}...")
                        img = cv2.imread(item) if isinstance(item, str) else cv2.imdecode(np.frombuffer(item.read(), np.uint8), 1)
                        if img is None: continue
                        
                        save_name = f"{final_class_name}_{i}_{datetime.now().strftime('%H%M%S')}.jpg"
                        cv2.imwrite(os.path.join(save_dir, save_name), img)
                        
                        crop = engine.detect_and_crop(img, YOLO_CONF)
                        if show_preview and len(st.session_state.last_crops) < 8:
                            st.session_state.last_crops.append(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                        
                        angles = [0, 90, 180, 270] 
                        for angle in angles:
                            if angle == 0: rot = crop
                            elif angle == 90: rot = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
                            elif angle == 180: rot = cv2.rotate(crop, cv2.ROTATE_180)
                            elif angle == 270: rot = cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)
                            
                            dino_vec, _ = engine.extract_features(rot)
                            db.insert_data(packs_db, f"{final_class_name}_rot{angle}", dino_vec, None)
                        
                        progress.progress((i+1)/total)
                    
                    status.update(label="✅ Processing Complete!", state="complete", expanded=False)
                
                # Save PKL & JSON
                db.save_pkl(packs_db, PKL_PATH)
                current_list = get_base_drug_names(packs_db)
                db.generate_metadata(current_list, JSON_PATH)
                
                st.session_state.push_success_msg = f"✅ Saved {total} images for {final_class_name} and updated JSON."
                st.rerun()

        # >>> CASE 3: ZIP IMPORT <<<
        elif mode == "Bulk Import (Zip)":
            if not uploaded_zip:
                st.error("❌ Please upload a Zip file.")
            else:
                st.session_state.last_crops = []
                temp_dir = "temp_zip_process"
                if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
                os.makedirs(temp_dir, exist_ok=True)
                
                processed_count = 0
                
                with st.status("📦 Unzipping & Analyzing...", expanded=True) as status:
                    # 1. Unzip
                    with zipfile.ZipFile(uploaded_zip, 'r') as z:
                        z.extractall(temp_dir)
                    
                    # 2. Find valid folders
                    valid_tasks = []
                    for root, dirs, files in os.walk(temp_dir):
                        folder_name = os.path.basename(root).lower()
                        if "_box" in folder_name or "_blister" in folder_name:
                            imgs = [os.path.join(root, f) for f in files if f.lower().endswith(('.jpg','.png','.jpeg'))]
                            if imgs: valid_tasks.append((folder_name, imgs))
                    
                    if not valid_tasks:
                        st.error("No folders with '_box' or '_blister' found in zip.")
                        st.stop()
                        
                    # 3. Process Loops
                    prog_bar = st.progress(0)
                    for idx, (cls_name, img_paths) in enumerate(valid_tasks):
                        status.write(f"📂 Processing folder: **{cls_name}** ({len(img_paths)} imgs)...")
                        
                        # Cleanup Old
                        save_dir = os.path.join(IMG_DB_ROOT, cls_name)
                        if os.path.exists(save_dir): shutil.rmtree(save_dir)
                        os.makedirs(save_dir, exist_ok=True)
                        
                        keys_to_del = [k for k in packs_db.keys() if k.startswith(cls_name)]
                        for k in keys_to_del: del packs_db[k]
                        
                        type_suffix = "blister" if "_blister" in cls_name else "box"
                        
                        for i, p in enumerate(img_paths):
                            img = cv2.imread(p)
                            if img is None: continue
                            
                            save_name = f"{cls_name}_{i}_{datetime.now().strftime('%H%M%S')}.jpg"
                            cv2.imwrite(os.path.join(save_dir, save_name), img)
                            
                            crop = engine.detect_and_crop(img, YOLO_CONF)
                            if len(st.session_state.last_crops) < 8:
                                st.session_state.last_crops.append(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                                
                            angles = [0, 90, 180, 270] 
                            for angle in angles:
                                if angle == 0: rot = crop
                                elif angle == 90: rot = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
                                elif angle == 180: rot = cv2.rotate(crop, cv2.ROTATE_180)
                                elif angle == 270: rot = cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)
                                
                                dino_vec, _ = engine.extract_features(rot)
                                db.insert_data(packs_db, f"{cls_name}_rot{angle}", dino_vec, None)
                        
                        processed_count += 1
                        prog_bar.progress((idx+1)/len(valid_tasks))
                    
                    status.update(label=f"✅ Processed {processed_count} folders from Zip!", state="complete", expanded=False)
                
                # Cleanup Temp
                if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
                
                # Save PKL & JSON
                db.save_pkl(packs_db, PKL_PATH)
                current_list = get_base_drug_names(packs_db)
                db.generate_metadata(current_list, JSON_PATH)
                
                st.session_state.push_success_msg = f"✅ Zip Import Success! Added {processed_count} classes."
                st.rerun()

# --- PREVIEW SECTION ---
if "last_crops" in st.session_state and st.session_state.last_crops:
    st.divider()
    st.subheader(f"🖼️ YOLO Preview Results ({len(st.session_state.last_crops)} samples)")
    cols = st.columns(4)
    for idx, crop_img in enumerate(st.session_state.last_crops):
        with cols[idx % 4]:
            st.image(crop_img, caption=f"Crop #{idx+1}", use_container_width=True)

# ================= 5. UTILITIES (EDGE & RELEASE) =================
st.divider()
c_e1, c_e2 = st.columns([2, 1])

with c_e1:
    st.markdown("### 📡 Edge Feedback Loop")
    st.caption("Pull captured data from Raspberry Pi.")

with c_e2:
    if st.button("📥 PULL FROM EDGE", use_container_width=True):
        with st.status("Connecting to S3...", expanded=True) as status:
            try:
                prefix = "data_collection/"
                objs = cloud.s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=prefix)
                if 'Contents' in objs:
                    latest = sorted(objs['Contents'], key=lambda x: x['LastModified'], reverse=True)[0]
                    key = latest['Key']
                    name = os.path.basename(key)
                    local = os.path.join("edge_data_inbox", name)
                    os.makedirs("edge_data_inbox", exist_ok=True)
                    
                    status.write(f"Downloading {name}...")
                    cloud.s3.download_file(S3_BUCKET, key, local)
                    st.success(f"Saved: {local}")
                else:
                    st.warning("No data found.")
            except Exception as e:
                st.error(f"Error: {e}")

st.divider()
if st.button("🚀 RELEASE TO PRODUCTION", type="primary", use_container_width=True):
    with st.status("Syncing Production...", expanded=True) as status:
        try:
            latest_list = get_base_drug_names(packs_db)
            db.generate_metadata(latest_list, JSON_PATH)
            status.write("Uploading Metadata...")
            cloud.upload_file(JSON_PATH, f"latest/{JSON_PATH}")
            status.write("Uploading Database...")
            cloud.upload_file(PKL_PATH, f"latest/{PKL_PATH}")
            st.success(f"Synced {len(latest_list)} SKUs!")
        except Exception as e:
            st.error(str(e))