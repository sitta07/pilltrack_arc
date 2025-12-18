import streamlit as st
import yaml, os, cv2, shutil
import numpy as np
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from cloud_manager import CloudManager
from db_manager import DBManager
from engine import AIEngine

# ================= ⚙️ SETUP & CONFIG =================
load_dotenv()
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# ปรับ UI ให้กว้างสะใจสไตล์ Developer [cite: 2025-11-11]
st.set_page_config(page_title="PillTrack Ops Hub", layout="wide")

cloud = CloudManager(os.getenv('S3_BUCKET_NAME'))
db = DBManager()

# --- SUCCESS POPUP LOGIC --- [cite: 2025-11-11, 2025-12-05]
# โชว์ Popup เมื่อหน้าเว็บรีเฟรชกลับมาหลัง Push สำเร็จ
if "push_success_msg" in st.session_state:
    st.toast(st.session_state.push_success_msg, icon="✅")
    del st.session_state.push_success_msg 

@st.cache_resource
def get_engine():
    """โหลด AI Engine เพียงครั้งเดียว (Singleton) [cite: 2025-12-05]"""
    return AIEngine(config['artifacts']['model'], config['settings']['dino_size'])

engine = get_engine()
PKL_PATH = config['artifacts']['pack_vec']
JSON_PATH = "database/drug_list.json"

# ================= 🖥️ DATA PREPARATION =================
packs_db = db.load_pkl(PKL_PATH)
current_drugs = db.get_unique_drugs(packs_db)

# ================= 🖥️ UI LAYOUT =================
st.title("PillTrack: MLOps Producer Hub")

# --- SIDEBAR: STATUS & LOGS ---
st.sidebar.header("Operations Status")
s3_ok, s3_status = cloud.check_connection()
st.sidebar.write(f"Cloud S3: {s3_status}")
st.sidebar.write(f"Compute: {engine.device}")

st.sidebar.markdown("---")
# ระบบ Activity Log สำหรับกดดูประวัติ [cite: 2025-12-05]
with st.sidebar.expander("🕒 View Activity Logs", expanded=False):
    recent_logs = db.get_logs()
    if recent_logs:
        st.dataframe(pd.DataFrame(recent_logs), use_container_width=True, hide_index=True)
    else:
        st.caption("No history yet.")

if st.sidebar.button("FORCE REFRESH SYSTEM"):
    st.cache_resource.clear()
    st.rerun()

# Dashboard Metrics [cite: 2025-11-11]
m1, m2, m3 = st.columns(3)
m1.metric("Local Classes", len(current_drugs))
m2.metric("Total Vectors", sum([len(v) for v in packs_db.values()]))
m3.metric("Cloud Status", "Ready" if s3_ok else "Disconnected")

col_l, col_r = st.columns(2)
with col_l:
    st.subheader("🟢 Local Efficiency")
    with st.container(border=True):
        if current_drugs:
            for name in current_drugs:
                count = sum([len(v) for k, v in packs_db.items() if k.startswith(f"{name}_pack")])
                eff = min(count / 40, 1.0)
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

# ================= 🛠️ ACTION LAYER (HARDMODE) =================
st.subheader("Dataset Management")
mode = st.radio("Action Mode:", ["New Pack", "Enhance Existing"], horizontal=True)

with st.form("update_form", clear_on_submit=True):
    if mode == "New Pack":
        name_in = st.text_input("Enter New Drug Name:").strip().lower()
    else:
        # ดึงจาก .pkl มาทำ Dropdown เท่านั้นตามโจทย์ [cite: 2025-12-05]
        name_in = st.selectbox("Select Existing Drug:", current_drugs) if current_drugs else None

    files_in = st.file_uploader("Upload Samples:", accept_multiple_files=True)
    show_yolo = st.checkbox("Show YOLO Detection Preview", value=True)
    
    if st.form_submit_button("PROCESS & SAVE LOCAL", use_container_width=True):
        if name_in and files_in:
            folder_name = f"{name_in}_pack"
            img_save_dir = os.path.join(config['paths']['db_images'], folder_name)
            
            if mode == "New Pack":
                if os.path.exists(img_save_dir): shutil.rmtree(img_save_dir)
                for k in [k for k in packs_db.keys() if k.startswith(folder_name)]: del packs_db[k]
            
            os.makedirs(img_save_dir, exist_ok=True)
            st.session_state.last_crops = []
            
            # --- START PROCESSING ---
            for i, file in enumerate(files_in):
                raw_bytes = file.read()
                img = cv2.imdecode(np.frombuffer(raw_bytes, np.uint8), 1)
                
                # บันทึกรูปภาพลงโฟลเดอร์ database_images/ [cite: 2025-12-05]
                img_name = f"{datetime.now().timestamp()}_{i}.jpg"
                cv2.imwrite(os.path.join(img_save_dir, img_name), img)
                
                # YOLO & Feature Extraction [cite: 2025-11-11]
                crop = engine.detect_and_crop(img, config['settings']['yolo_conf'])
                if show_yolo: st.session_state.last_crops.append(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                
                for angle, suffix in [(0,"_rot0"),(90,"_rot90"),(180,"_rot180"),(270,"_rot270")]:
                    rot = crop.copy()
                    if angle == 90: rot = cv2.rotate(rot, cv2.ROTATE_90_CLOCKWISE)
                    elif angle == 180: rot = cv2.rotate(rot, cv2.ROTATE_180)
                    elif angle == 270: rot = cv2.rotate(rot, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    
                    vec = engine.extract_vector(rot)
                    f_key = f"{folder_name}{suffix}"
                    if f_key not in packs_db: packs_db[f_key] = []
                    packs_db[f_key].append(vec)

            # บันทึกข้อมูลและลง Log [cite: 2025-12-05]
            db.save_pkl(packs_db, PKL_PATH)
            db.add_log(event_type=f"ADD_DATA ({mode})", drug_name=name_in, count=len(files_in))
            
            st.success("🟢 Local Database Updated!")
            st.rerun()

if "last_crops" in st.session_state and st.session_state.last_crops:
    with st.expander("🔍 YOLO Detection Preview", expanded=True):
        st.image(st.session_state.last_crops[:12], width=110)

# ================= 🚀 RELEASE MANAGEMENT (PUSH) =================
st.divider()
c_p1, c_p2 = st.columns([2, 1])
with c_p2:
    st.subheader("Release to Cloud")
    if st.button("PUSH ALL ARTIFACTS TO S3", type="primary", use_container_width=True, disabled=not s3_ok):
        with st.status("MLOps Pipeline: Syncing Production...", expanded=True) as status:
            try:
                # 1. Generate Metadata
                latest_list = db.get_unique_drugs(packs_db)
                db.generate_metadata(latest_list, JSON_PATH)
                
                # 2. Push ทุกไฟล์ที่เกี่ยวข้องขึ้น S3 [cite: 2025-11-11]
                artifacts = {**config['artifacts'], "drug_list": JSON_PATH}
                for k, path in artifacts.items():
                    if os.path.exists(path):
                        cloud.upload_file(path, f"latest/{path}")
                
                # 3. บันทึก Log การ Push [cite: 2025-12-05]
                db.add_log(event_type="PRODUCTION_PUSH", details=f"Sync {len(latest_list)} classes to S3")
                
                st.session_state.push_success_msg = f"Push Successful: {len(latest_list)} drugs in registry"
                st.rerun()
            except Exception as e:
                st.error(f"🔴 Pipeline Error: {str(e)}")

st.sidebar.caption(f"PillTrack MLOps | {datetime.now().strftime('%Y-%m-%d %H:%M')}")