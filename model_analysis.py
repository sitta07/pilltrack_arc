import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
import warnings
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# Config Visual
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 10)
plt.rcParams['font.size'] = 12

class PillTrackAnalyzer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.X = None
        self.y = None
        self.y_grouped = None # สำหรับเก็บชื่อยาที่ตัด rotation ออกแล้ว

    def load_data(self):
        """โหลดไฟล์ Pickle"""
        if not os.path.exists(self.file_path):
            print(f"❌ Error: ไม่พบไฟล์ที่ {self.file_path}")
            return False
        with open(self.file_path, 'rb') as f:
            self.data = pickle.load(f)
        print(f"✅ โหลดไฟล์สำเร็จ! ประเภท: {type(self.data)}")
        return True

    def process_data(self):
        """สกัดข้อมูล, แก้ปัญหา Inhomogeneous Shape และจัดกลุ่มชื่อยา"""
        X_raw = []
        y_list = []
        shapes = {}

        print("🔍 กำลังประมวลผลข้อมูล...")
        
        for class_name, items in self.data.items():
            # ดึงข้อมูลจาก Dict/List
            target_items = items.values() if isinstance(items, dict) else (items if isinstance(items, (list, np.ndarray)) else [items])

            for item in target_items:
                try:
                    vec = np.array(item, dtype=float).flatten()
                    X_raw.append(vec)
                    y_list.append(str(class_name))
                    shapes[vec.shape[0]] = shapes.get(vec.shape[0], 0) + 1
                except:
                    continue

        if not X_raw: return False

        # Senior Logic: จัดการมิติไม่เท่ากัน (Padding)
        max_dim = max(shapes.keys())
        X_fixed = [np.pad(v, (0, max_dim - v.shape[0]), mode='constant') if v.shape[0] < max_dim else v for v in X_raw]
        
        self.X = np.array(X_fixed)
        self.y = np.array(y_list)
        
        # --- Grouping Logic: ตัด _rot ออกเพื่อรวมกลุ่มยา ---
        # ใช้ Regex ตัดส่วนท้าย เช่น _rot0, _rot180 ออก
        self.y_grouped = np.array([re.sub(r'_rot\d+', '', label) for label in self.y])
        
        # Scaling
        self.X = StandardScaler().fit_transform(self.X)
        print(f"✅ สกัดข้อมูลสำเร็จ: {self.X.shape}")
        return True

    def plot_distribution(self):
        """พล็อตกราฟแท่งแสดงจำนวนภาพต่อตัวยา (รวมทุกมุม)"""
        print("📊 กำลังสร้างกราฟแสดงจำนวนข้อมูล...")
        
        # นับจำนวนรายชื่อยาที่รวมกลุ่มแล้ว
        df = pd.DataFrame({'Drug': self.y_grouped})
        counts = df['Drug'].value_counts().reset_index()
        counts.columns = ['Drug Name', 'Total Samples']

        # พล็อตกราฟแท่งแนวนอน
        plt.figure(figsize=(12, 12))
        ax = sns.barplot(x='Total Samples', y='Drug Name', data=counts, palette='magma')
        
        # ใส่ตัวเลขกำกับปลายแท่ง
        for i, v in enumerate(counts['Total Samples']):
            ax.text(v + 1, i, str(v), va='center', fontweight='bold')

        plt.title('Total Samples per Drug (Combined All Rotations)', fontsize=16, fontweight='bold')
        plt.xlabel('Number of Samples')
        plt.ylabel('Drug Base Name')
        plt.tight_layout()
        plt.show()

    def run_analysis(self):
        if not self.load_data(): return
        if not self.process_data(): return
        
        # 1. แสดงกราฟแท่งสรุปจำนวนยา
        self.plot_distribution()
        
        # 2. คำนวณ Metrics พื้นฐาน
        sil = silhouette_score(self.X, self.y)
        print(f"\n🎯 Cluster Quality (Original Classes):")
        print(f"• Silhouette Score: {sil:.4f}")
        
        # 3. Visual Analysis (t-SNE)
        print("\n🎨 Generating t-SNE (Grouping by Base Name)...")
        tsne = TSNE(n_components=2, perplexity=min(30, len(self.X)-1), random_state=42)
        X_tsne = tsne.fit_transform(self.X)
        
        plt.figure(figsize=(12, 8))
        sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=self.y_grouped, palette='husl', alpha=0.7)
        plt.title('t-SNE Visualization (Colors by Drug Name)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # ระบุ Path ไฟล์ pkl ของคุณ
    PATH = 'database/pill_fingerprints.pkl'
    analyzer = PillTrackAnalyzer(PATH)
    analyzer.run_analysis()