import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances

class DatasetAuditor:
    def __init__(self, packs_db):
        """
        รับ packs_db (Dictionary) เข้ามาเตรียมประมวลผล
        """
        self.packs_db = packs_db
        self.df_audit = pd.DataFrame()
        self.data_points = []
        self._prepare_data()

    def _prepare_data(self):
        """
        แปลง Dict โครงสร้างซับซ้อน ให้เป็น List ของ Data Points
        เพื่อเตรียมเข้า PCA และคำนวณ Distance
        """
        self.data_points = []
        class_groups = {}
        
        if not self.packs_db:
            return

        # 1. รวบรวม Vector ทั้งหมด
        for key, val in self.packs_db.items():
            if not isinstance(val, dict): continue
            
            base_name = key.split('_rot')[0]
            vectors = val.get('dino', [])
            
            if base_name not in class_groups: 
                class_groups[base_name] = []
            
            class_groups[base_name].extend(vectors)
            
            for vec in vectors:
                self.data_points.append({'Class': base_name, 'Vector': vec})
        
        self.class_groups = class_groups

    def run_audit(self):
        """
        คำนวณค่าทางสถิติทั้งหมด: Count, Spread, Nearest Enemy
        Return: DataFrame ของผลลัพธ์
        """
        if not self.class_groups:
            return pd.DataFrame()

        # คำนวณ Centroid ของแต่ละคลาส
        centroids = {}
        for cls, vecs in self.class_groups.items():
            if len(vecs) > 0: 
                centroids[cls] = np.mean(vecs, axis=0)

        audit_results = []
        
        for cls, vecs in self.class_groups.items():
            if len(vecs) == 0: continue
            
            # 1. Intra-class Spread (การกระจายตัวภายใน)
            centroid = centroids[cls]
            distances = [np.linalg.norm(v - centroid) for v in vecs]
            spread_score = np.mean(distances) if distances else 0
            
            # 2. Inter-class Distance (ระยะห่างกับศัตรูที่ใกล้ที่สุด)
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
                min_dist_to_enemy = 999  # กรณีมีแค่คลาสเดียว
            
            audit_results.append({
                "Class": cls, 
                "Count": len(vecs), 
                "Spread": spread_score,
                "Nearest_Dist": min_dist_to_enemy, 
                "Nearest_Enemy": nearest_enemy
            })
            
        self.df_audit = pd.DataFrame(audit_results)
        return self.df_audit

    def get_pca_dataframe(self, n_components=2):
        """
        คำนวณ PCA เพื่อใช้ plot กราฟ
        """
        if len(self.data_points) < n_components + 1:
            return pd.DataFrame() # ข้อมูลน้อยไปทำ PCA ไม่ได้

        vectors = [d['Vector'] for d in self.data_points]
        classes = [d['Class'] for d in self.data_points]
        
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(vectors)
        
        df_pca = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(n_components)])
        df_pca['Class'] = classes
        return df_pca

    def get_suggestions(self):
        """
        Logic การแนะนำ AI (แยกจาก UI)
        Return: Tuple ของ lists (low_data, confused, high_spread)
        """
        if self.df_audit.empty:
            return [], [], []

        # 1. Low Data
        low_data = self.df_audit[self.df_audit['Count'] < 15]['Class'].tolist()
        
        # 2. Confusion Risk (ใกล้กันเกินไป Top 20%)
        confused = []
        if len(self.df_audit) > 1:
            threshold_dist = self.df_audit['Nearest_Dist'].quantile(0.20)
            confused = self.df_audit[self.df_audit['Nearest_Dist'] <= threshold_dist].sort_values('Nearest_Dist')['Class'].tolist()

        # 3. High Spread (กระจายตัวมากเกินไป Top 20%)
        high_spread = []
        if len(self.df_audit) > 1:
            threshold_spread = self.df_audit['Spread'].quantile(0.80)
            high_spread = self.df_audit[self.df_audit['Spread'] >= threshold_spread]['Class'].tolist()
            
        return low_data, confused, high_spread