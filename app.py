import streamlit as st
import os
import cv2
import shutil
import numpy as np
import plotly.express as px
from datetime import datetime

# --- Import Custom Modules from 'src' ---
# ตรวจสอบให้แน่ใจว่าย้ายไฟล์ cloud_manager, db_manager, engine ไปไว้ใน folder 'src' แล้ว
from src.utils import load_config, AppPaths, get_base_drug_names
from src.analytics import DatasetAuditor
from src.cloud_manager import CloudManager
from src.db_manager import DBManager
from src.engine import AIEngine

# ================= 1. SETUP & CONFIGURATION =================
st.set_page_config(page_title="PillTrack Ops Hub", layout="wide", page_icon="💊")

# Load Config & Paths (ใช้ Helper Class จาก utils.py)
config = load_config()
paths = AppPaths(config)
SETTINGS = config.get('settings', {})

# Constants
S3_BUCKET = os.getenv('S3_BUCKET_NAME')
PULL_SOURCE_PREFIX = "latest/staging_model"
PUSH_TARGET_PREFIX = "latest/register_model"
DINO_SIZE = SETTINGS.get('dino_size', 224)
YOLO_CONF = SETTINGS.get('yolo_conf', 0.5)

# Initialize Services
cloud = CloudManager(S3_BUCKET)
db = DBManager()

# Global Session Alert
if "alert_msg" in st.session_state:
    st.success(st.session_state.alert_msg, icon="✅")
    del st.session_state.alert_msg 

@st.cache_resource
def get_engine():
    """Load AI Model only once"""
    return AIEngine(paths.model_path, DINO_SIZE)

# Load Data
engine = get_engine()
packs_db = db.load_pkl(paths.pkl_path)
current_drugs = get_base_drug_names(packs_db)

# ================= 2. SIDEBAR =================
with st.sidebar:
    st.header("☁️ Cloud Manager")
    _, s3_status = cloud.check_connection()
    st.write(f"S3 Status: `{s3_status}`")

    st.subheader("📥 Source: Staging")
    if st.button("📥 PULL STAGING MODEL", use_container_width=True):
        with st.status(f"Syncing from {PULL_SOURCE_PREFIX}...") as s:
            try:
                cloud.s3.download_file(str(S3_BUCKET), f"{PULL_SOURCE_PREFIX}/pill_fingerprints.pkl", paths.pkl_path)
                cloud.s3.download_file(str(S3_BUCKET), f"{PULL_SOURCE_PREFIX}/drug_list.json", paths.json_path)
                s.update(label="✅ Pulled Successfully!", state="complete")
                st.session_state.alert_msg = f"Updated local data from {PULL_SOURCE_PREFIX}"
                st.rerun()
            except Exception as e:
                st.error(f"Pull Failed: {str(e)}")

    if st.button("🔄 REFRESH APP", use_container_width=True):
        st.cache_resource.clear()
        st.rerun()

    st.divider()
    st.caption(f"Last Update: {datetime.now().strftime('%H:%M:%S')}")

# ================= 3. MAIN DASHBOARD =================
st.title("💊 PillTrack: MLOps Producer Hub")
tab_ops, tab_analytics, tab_deploy = st.tabs(["🛠️ Operations", "📊 Dataset Analytics & Fix", "🚀 Register Model"])

# ================= TAB 1: OPERATIONS =================
with tab_ops:
    # Metrics
    m1, m2 = st.columns(2)
    m1.metric("Unique Drugs (SKUs)", len(current_drugs))
    total_vecs = sum([len(v.get('dino', [])) for v in packs_db.values() if isinstance(v, dict)])
    m2.metric("Total Embeddings", total_vecs)
    
    st.divider()
    st.subheader("🗑️ Delete / Cleanup")
    
    # Delete Logic
    target_to_delete = st.selectbox("Select Class to Delete:", sorted(list(set([k.split('_rot')[0] for k in packs_db.keys()]))))
    
    if st.button("❌ DELETE SELECTED CLASS", type="primary"):
        if target_to_delete:
            # ลบจาก Dict
            keys_to_del = [k for k in packs_db.keys() if k.startswith(target_to_delete)]
            for k in keys_to_del: 
                del packs_db[k]
            
            # ลบไฟล์จริง
            path_to_remove = os.path.join(paths.img_db_root, target_to_delete)
            if os.path.exists(path_to_remove): 
                shutil.rmtree(path_to_remove)
            
            # Save กลับ
            db.save_pkl(packs_db, paths.pkl_path)
            db.generate_metadata(get_base_drug_names(packs_db), paths.json_path)
            
            st.session_state.alert_msg = f"Deleted {target_to_delete} completely."
            st.rerun()

# ================= TAB 2: DATASET ANALYTICS (REFACTORED) =================
with tab_analytics:
    st.header("📊 Dataset Health & AI Audit")
    
    if not packs_db:
        st.warning("No data available. Please Pull from Staging first.")
    else:
        # --- ใช้ Class Auditor จาก analytics.py ---
        auditor = DatasetAuditor(packs_db)
        
        with st.spinner("🤖 AI Auditor is analyzing your dataset..."):
            df_audit = auditor.run_audit()
            low_data, confused, high_spread = auditor.get_suggestions() # รับค่าคำแนะนำมาแสดงผล

        # 1. แสดง AI Suggestions
        st.subheader("🤖 AI Assistant Suggestions")
        col_s1, col_s2, col_s3 = st.columns(3)
        
        with col_s1:
            st.markdown("#### 🚨 Critical: Low Data")
            if low_data:
                for cls in low_data: st.error(f"**{cls}**: น้อยกว่าเกณฑ์")
            else:
                st.success("✅ จำนวนรูปผ่านเกณฑ์")

        with col_s2:
            st.markdown("#### ⚠️ Warning: Confusion")
            if confused:
                for cls in confused: st.warning(f"**{cls}** เสี่ยงสับสน")
            else:
                st.success("✅ Class แยกตัวกันดี")

        with col_s3:
            st.markdown("#### 📸 Tip: Retake Photos")
            if high_spread:
                for cls in high_spread: st.info(f"**{cls}** กระจายตัวสูง")
            else:
                st.success("✅ ข้อมูลเกาะกลุ่มกันดี")

        st.divider()

        # 2. Visualization (ใช้ข้อมูลที่ Auditor คำนวณมาแล้ว)
        c_chart1, c_chart2 = st.columns([1, 2])
        with c_chart1:
            st.subheader("Sample Count")
            if not df_audit.empty:
                df_count = df_audit.sort_values('Count')
                fig_bar = px.bar(df_count, x='Count', y='Class', orientation='h', color='Count')
                st.plotly_chart(fig_bar, use_container_width=True)
        
        with c_chart2:
            st.subheader("Feature Space (PCA)")
            df_pca = auditor.get_pca_dataframe() # เรียกฟังก์ชัน PCA จาก class
            if not df_pca.empty:
                fig_scatter = px.scatter(df_pca, x='PC1', y='PC2', color='Class', hover_data=['Class'])
                st.plotly_chart(fig_scatter, use_container_width=True)

        st.divider()

        # 3. INSTANT FIX SECTION
        st.subheader("🛠️ Instant Fix: Add More Images")
        
        with st.form("instant_add_form"):
            c_fix1, c_fix2, c_fix3 = st.columns([2, 1, 1])
            with c_fix1:
                input_choice = st.radio("Target:", ["Existing Class", "New Class"], horizontal=True)
                if input_choice == "Existing Class" and current_drugs:
                    target_class_fix = st.selectbox("Select Drug to Fix:", current_drugs)
                else:
                    target_class_fix = st.text_input("New Drug Name:").strip().lower()
            
            with c_fix2:
                 fix_type = st.selectbox("Type:", ["Blister", "Box"], key="fix_type")

            with c_fix3:
                 files_fix = st.file_uploader("Upload New Images:", accept_multiple_files=True, key="fix_uploader")

            submitted_fix = st.form_submit_button("🚀 PROCESS & UPDATE", type="primary", use_container_width=True)

            if submitted_fix and target_class_fix and files_fix:
                cls_name = f"{target_class_fix}_{fix_type.lower()}"
                save_dir = os.path.join(paths.img_db_root, cls_name)
                os.makedirs(save_dir, exist_ok=True)
                
                # A. Save Files
                current_count = len(os.listdir(save_dir)) if os.path.exists(save_dir) else 0
                saved_files = []
                for i, f in enumerate(files_fix):
                    file_path = os.path.join(save_dir, f"new_{current_count + i}.jpg")
                    with open(file_path, "wb") as buffer:
                        buffer.write(f.getbuffer())
                    saved_files.append(file_path)
                
                # B. Clear Old Vectors in DB (for this class)
                keys_to_del = [k for k in packs_db.keys() if k.startswith(cls_name)]
                for k in keys_to_del: del packs_db[k]
                
                # C. Re-Process (Using Engine)
                # NOTE: ถ้าอยาก Clean กว่านี้ ย้าย Logic นี้ไปไว้ใน src/engine.py ได้ครับ
                all_imgs = [os.path.join(save_dir, f) for f in os.listdir(save_dir) if f.endswith(('.jpg','.png'))]
                
                with st.spinner(f"Processing {len(all_imgs)} images for {cls_name}..."):
                    for p in all_imgs:
                        img = cv2.imread(p)
                        if img is None: continue
                        crop = engine.detect_and_crop(img, YOLO_CONF)
                        if crop is None: continue
                        
                        # Augment & Embed
                        for angle in [0, 90, 180, 270]:
                            if angle != 0:
                                rot_code = {90:cv2.ROTATE_90_CLOCKWISE, 180:cv2.ROTATE_180, 270:cv2.ROTATE_90_COUNTERCLOCKWISE}[angle]
                                rot = cv2.rotate(crop, rot_code)
                            else:
                                rot = crop
                                
                            vec, _ = engine.extract_features(rot)
                            db.insert_data(packs_db, f"{cls_name}_rot{angle}", vec, None)
                
                # D. Save & Refresh
                db.save_pkl(packs_db, paths.pkl_path)
                db.generate_metadata(get_base_drug_names(packs_db), paths.json_path)
                st.session_state.alert_msg = f"✅ Updated {cls_name} successfully!"
                st.rerun()

# ================= TAB 3: REGISTER MODEL =================
with tab_deploy:
    st.header("🚀 Register Model")
    st.info(f"Target Bucket: `s3://{S3_BUCKET}/{PUSH_TARGET_PREFIX}/`")
    
    col_d1, col_d2 = st.columns([3, 1])
    with col_d1:
        st.markdown("""
        **Ready for Production?**
        1. ✅ **Data Quality:** No critical alerts in Analytics tab.
        2. ✅ **Verification:** Model performs well on local tests.
        """)
    
    with col_d2:
        if st.button("🚀 PUSH TO PRODUCTION", type="primary", use_container_width=True):
            with st.status(f"Uploading artifacts...") as s:
                try:
                    cloud.upload_file(paths.pkl_path, f"{PUSH_TARGET_PREFIX}/pill_fingerprints.pkl")
                    cloud.upload_file(paths.json_path, f"{PUSH_TARGET_PREFIX}/drug_list.json")
                    # ถ้ามีไฟล์ Model.pt ก็ upload ด้วยได้เลย
                    # cloud.upload_file(paths.model_path, f"{PUSH_TARGET_PREFIX}/model.pt")
                    
                    s.update(label="✅ Deployed Successfully!", state="complete")
                    st.toast("Model Registered for Production!", icon="🚀")
                except Exception as e:
                    st.error(f"Deploy Failed: {str(e)}")