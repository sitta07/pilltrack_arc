import streamlit as st
import yaml, os, cv2, shutil
import numpy as np
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

# --- Analytics Libraries ---
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances
import plotly.express as px

# Import Custom Modules
from cloud_manager import CloudManager
from db_manager import DBManager
from engine import AIEngine

# ================= 1. SETUP & CONFIGURATION =================
load_dotenv()
S3_BUCKET = os.getenv('S3_BUCKET_NAME')

def load_config():
    if os.path.exists("config.yaml"):
        with open("config.yaml", "r") as f:
            return yaml.safe_load(f) or {}
    return {}

config = load_config()
ARTIFACTS = config.get('artifacts', {})
PATHS = config.get('paths', {})
SETTINGS = config.get('settings', {})

# Local Paths
PKL_PATH = str(ARTIFACTS.get('pack_vec', 'database/pill_fingerprints.pkl'))
JSON_PATH = str(ARTIFACTS.get('drug_list', 'database/drug_list.json'))
MODEL_PATH = str(ARTIFACTS.get('model', 'models/seg_best_process.pt'))
IMG_DB_ROOT = str(PATHS.get('db_images', 'database_images'))

# Config Paths
PULL_SOURCE_PREFIX = "latest/staging_model"
PUSH_TARGET_PREFIX = "latest/register_model"

# Engine Settings
DINO_SIZE = SETTINGS.get('dino_size', 224)
YOLO_CONF = SETTINGS.get('yolo_conf', 0.5)

os.makedirs(os.path.dirname(PKL_PATH), exist_ok=True)
os.makedirs(IMG_DB_ROOT, exist_ok=True)

# ================= 2. INITIALIZATION =================
st.set_page_config(page_title="PillTrack Ops Hub", layout="wide")
cloud = CloudManager(S3_BUCKET)
db = DBManager()

if "alert_msg" in st.session_state:
    st.success(st.session_state.alert_msg, icon="✅")
    del st.session_state.alert_msg 

@st.cache_resource
def get_engine():
    return AIEngine(MODEL_PATH, DINO_SIZE)

engine = get_engine()
packs_db = db.load_pkl(PKL_PATH)

def get_base_drug_names(db_dict):
    names = set()
    for k in db_dict.keys():
        base = k.split('_box')[0].split('_blister')[0].split('_pack')[0].split('_rot')[0] 
        names.add(base)
    return sorted(list(names))

current_drugs = get_base_drug_names(packs_db)

# ================= 3. SIDEBAR =================
st.sidebar.header("☁️ Cloud Manager")
_, s3_status = cloud.check_connection()
st.sidebar.write(f"S3: `{s3_status}`")

st.sidebar.subheader("📥 Source: Staging")
if st.sidebar.button("📥 PULL STAGING MODEL", use_container_width=True):
    with st.sidebar.status(f"Syncing from {PULL_SOURCE_PREFIX}...") as s:
        try:
            cloud.s3.download_file(str(S3_BUCKET), f"{PULL_SOURCE_PREFIX}/pill_fingerprints.pkl", PKL_PATH)
            cloud.s3.download_file(str(S3_BUCKET), f"{PULL_SOURCE_PREFIX}/drug_list.json", JSON_PATH)
            s.update(label="✅ Pulled Successfully!", state="complete")
            st.session_state.alert_msg = f"Updated local data from {PULL_SOURCE_PREFIX}"
            st.rerun()
        except Exception as e:
            st.error(f"Pull Failed: {str(e)}")

if st.sidebar.button("🔄 REFRESH APP", use_container_width=True):
    st.cache_resource.clear()
    st.rerun()

st.sidebar.divider()
st.sidebar.caption(f"Last Update: {datetime.now().strftime('%H:%M')}")

# ================= 4. MAIN DASHBOARD =================
st.title("💊 PillTrack: MLOps Producer Hub")
tab_ops, tab_analytics, tab_deploy = st.tabs(["🛠️ Operations", "📊 Dataset Analytics & Fix", "🚀 Register Model"])

# ================= TAB 1: OPERATIONS =================
with tab_ops:
    m1, m2 = st.columns(2)
    m1.metric("Unique Drugs (SKUs)", len(current_drugs))
    m2.metric("Total Embeddings", sum([len(v.get('dino', [])) for v in packs_db.values() if isinstance(v, dict)]))
    
    st.divider()
    st.subheader("🗑️ Delete / Cleanup")
    
    target_to_delete = st.selectbox("Select Class to Delete:", sorted(list(set([k.split('_rot')[0] for k in packs_db.keys()]))))
    
    if st.button("❌ DELETE SELECTED CLASS", type="primary"):
        if target_to_delete:
            for k in [k for k in packs_db.keys() if k.startswith(target_to_delete)]: del packs_db[k]
            path = os.path.join(IMG_DB_ROOT, target_to_delete)
            if os.path.exists(path): shutil.rmtree(path)
            db.save_pkl(packs_db, PKL_PATH)
            db.generate_metadata(get_base_drug_names(packs_db), JSON_PATH)
            st.session_state.alert_msg = f"Deleted {target_to_delete}"
            st.rerun()

# ================= TAB 2: DATASET ANALYTICS + INSTANT FIX =================
with tab_analytics:
    st.header("📊 Dataset Health & AI Audit")
    
    if not packs_db:
        st.warning("No data available.")
    else:
        # --- 1. DATA PREPARATION ---
        with st.spinner("🤖 AI Auditor is analyzing your dataset..."):
            data_points = []
            class_groups = {} 
            
            for key, val in packs_db.items():
                base_name = key.split('_rot')[0]
                vectors = val.get('dino', [])
                if base_name not in class_groups: class_groups[base_name] = []
                class_groups[base_name].extend(vectors)
                for vec in vectors:
                    data_points.append({'Class': base_name, 'Vector': vec})

            # --- 2. CALCULATE METRICS ---
            audit_results = []
            centroids = {}
            for cls, vecs in class_groups.items():
                if len(vecs) > 0: centroids[cls] = np.mean(vecs, axis=0)

            for cls, vecs in class_groups.items():
                if len(vecs) == 0: continue
                count = len(vecs)
                centroid = centroids[cls]
                distances = [np.linalg.norm(v - centroid) for v in vecs]
                spread_score = np.mean(distances) if distances else 0
                
                min_dist_to_enemy = float('inf')
                nearest_enemy = "None"
                
                if len(centroids) > 1:
                    for other_cls, other_cent in centroids.items():
                        if cls == other_cls: continue
                        dist = np.linalg.norm(centroid - other_cent)
                        if dist < min_dist_to_enemy:
                            min_dist_to_enemy = dist
                            nearest_enemy = other_cls
                else:
                    min_dist_to_enemy = 999 
                
                audit_results.append({
                    "Class": cls, "Count": count, "Spread": spread_score,
                    "Nearest_Dist": min_dist_to_enemy, "Nearest_Enemy": nearest_enemy
                })
            
            df_audit = pd.DataFrame(audit_results)

        # --- 3. AI SUGGESTIONS (TOP) ---
        st.subheader("🤖 AI Assistant Suggestions")
        col_s1, col_s2, col_s3 = st.columns(3)
        
        # Prepare lists for later use in "Instant Fix"
        list_low_data = []
        list_confused = []
        list_high_spread = []

        # Suggestion 1: Imbalance
        with col_s1:
            st.markdown("#### 🚨 Critical: Low Data")
            low_data = df_audit[df_audit['Count'] < 15]
            if not low_data.empty:
                list_low_data = low_data['Class'].tolist() # เก็บรายชื่อไว้ใช้ข้างล่าง
                for _, row in low_data.iterrows():
                    st.error(f"**{row['Class']}**: มีแค่ {row['Count']} รูป")
            else:
                st.success("✅ จำนวนรูปผ่านเกณฑ์")

        # Suggestion 2: Confusion Risk
        with col_s2:
            st.markdown("#### ⚠️ Warning: Confusion")
            if len(df_audit) > 1:
                threshold_dist = df_audit['Nearest_Dist'].quantile(0.20)
                confused = df_audit[df_audit['Nearest_Dist'] <= threshold_dist].sort_values('Nearest_Dist')
                if not confused.empty:
                    list_confused = confused['Class'].tolist() # เก็บรายชื่อ
                    for _, row in confused.iterrows():
                        st.warning(f"**{row['Class']}** ใกล้กับ **{row['Nearest_Enemy']}**")
                else:
                    st.success("✅ Class แยกตัวกันดี")

        # Suggestion 3: High Spread
        with col_s3:
            st.markdown("#### 📸 Tip: Retake Photos")
            if len(df_audit) > 1:
                threshold_spread = df_audit['Spread'].quantile(0.80)
                high_spread = df_audit[df_audit['Spread'] >= threshold_spread].sort_values('Spread', ascending=False)
                if not high_spread.empty:
                    list_high_spread = high_spread['Class'].tolist() # เก็บรายชื่อ
                    for _, row in high_spread.iterrows():
                        st.info(f"**{row['Class']}** กระจายตัวสูง")
                else:
                    st.success("✅ ข้อมูลเกาะกลุ่มกันดี")

        st.divider()

        # --- 4. VISUALIZATION ---
        c_chart1, c_chart2 = st.columns([1, 2])
        with c_chart1:
            st.subheader("Sample Count")
            if not df_audit.empty:
                df_count = df_audit.sort_values('Count')
                fig_bar = px.bar(df_count, x='Count', y='Class', orientation='h', color='Count')
                st.plotly_chart(fig_bar, use_container_width=True)
        with c_chart2:
            st.subheader("Feature Space (PCA)")
            if len(data_points) > 5:
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform([d['Vector'] for d in data_points])
                df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
                df_pca['Class'] = [d['Class'] for d in data_points]
                fig_scatter = px.scatter(df_pca, x='PC1', y='PC2', color='Class', hover_data=['Class'])
                st.plotly_chart(fig_scatter, use_container_width=True)

        st.divider()

        # --- 5. INSTANT FIX SECTION (UPDATED!) ---
        st.subheader("🛠️ Instant Fix: Add More Images")
        
        # [NEW] Priority Fix List Display
        if list_low_data or list_confused or list_high_spread:
            st.markdown("##### 🔥 รายการที่ต้องแก้ไขด่วน (Priority List):")
            c_p1, c_p2, c_p3 = st.columns(3)
            
            with c_p1:
                if list_low_data:
                    st.error(f"🔴 **รูปน้อย ({len(list_low_data)}):** {', '.join(list_low_data)}")
                else:
                    st.caption("✅ รูปครบถ้วน")
            
            with c_p2:
                if list_confused:
                    st.warning(f"🟡 **สับสน ({len(list_confused)}):** {', '.join(list_confused)}")
                else:
                    st.caption("✅ แยกตัวดี")
                    
            with c_p3:
                if list_high_spread:
                    st.info(f"🔵 **กระจาย ({len(list_high_spread)}):** {', '.join(list_high_spread)}")
                else:
                    st.caption("✅ เกาะกลุ่มดี")
        else:
            st.success("✨ ข้อมูลสุขภาพดีมาก ไม่ต้องแก้เพิ่มครับ!")

        # Fix Form
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

            submitted_fix = st.form_submit_button("🚀 PROCESS & UPDATE GRAPH", type="primary", use_container_width=True)

            if submitted_fix and target_class_fix and files_fix:
                cls_name = f"{target_class_fix}_{fix_type.lower()}"
                save_dir = os.path.join(IMG_DB_ROOT, cls_name)
                os.makedirs(save_dir, exist_ok=True)
                
                # 1. Save New Images
                current_count = len(os.listdir(save_dir)) if os.path.exists(save_dir) else 0
                for i, f in enumerate(files_fix):
                    img = cv2.imdecode(np.frombuffer(f.read(), np.uint8), 1)
                    if img is None: continue
                    cv2.imwrite(os.path.join(save_dir, f"new_{current_count + i}.jpg"), img)
                
                # 2. Re-Process Whole Folder
                keys_to_del = [k for k in packs_db.keys() if k.startswith(cls_name)]
                for k in keys_to_del: del packs_db[k]
                
                all_imgs = [os.path.join(save_dir, f) for f in os.listdir(save_dir) if f.endswith(('.jpg','.png'))]
                
                with st.spinner(f"Re-processing {len(all_imgs)} images for {cls_name}..."):
                    for p in all_imgs:
                        img = cv2.imread(p)
                        if img is None: continue
                        crop = engine.detect_and_crop(img, YOLO_CONF)
                        if crop is None: continue
                        for angle in [0, 90, 180, 270]:
                            rot = cv2.rotate(crop, {90:cv2.ROTATE_90_CLOCKWISE, 180:cv2.ROTATE_180, 270:cv2.ROTATE_90_COUNTERCLOCKWISE}.get(angle, -1)) if angle != 0 else crop
                            vec, _ = engine.extract_features(rot)
                            db.insert_data(packs_db, f"{cls_name}_rot{angle}", vec, None)
                
                # 3. Save & Refresh
                db.save_pkl(packs_db, PKL_PATH)
                db.generate_metadata(get_base_drug_names(packs_db), JSON_PATH)
                st.session_state.alert_msg = f"✅ Updated {cls_name} with {len(files_fix)} new images! Graph Refreshed."
                st.rerun()

# ================= TAB 3: REGISTER MODEL =================
with tab_deploy:
    st.header("🚀 Register Model")
    st.info(f"Target: `{PUSH_TARGET_PREFIX}/`")
    
    col_d1, col_d2 = st.columns([3, 1])
    with col_d1:
        st.markdown("""
        **Checklist:**
        1. ✅ **Imbalance:** ไม่มีแถบแดงในช่อง Critical
        2. ✅ **Confusion:** เช็คคู่ที่แจ้งเตือน Warning แล้ว
        """)
    
    with col_d2:
        if st.button("🚀 REGISTER / PUSH", type="primary", use_container_width=True):
            with st.status(f"Uploading to {PUSH_TARGET_PREFIX}...") as s:
                try:
                    cloud.upload_file(PKL_PATH, f"{PUSH_TARGET_PREFIX}/pill_fingerprints.pkl")
                    cloud.upload_file(JSON_PATH, f"{PUSH_TARGET_PREFIX}/drug_list.json")
                    s.update(label="✅ Success!", state="complete")
                    st.toast("Registered!")
                except Exception as e:
                    st.error(str(e))