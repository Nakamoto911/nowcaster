from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import pandas as pd
import numpy as np

class RegimeModel:
    def __init__(self):
        self.pca = PCA(n_components=2)
        # Use 5 components to handle "Deep Contraction" outliers separately
        # Mapping: Exp, Stag, Con_Mild, Con_Deep, Rec -> 4 Regimes
        self.gmm = GaussianMixture(n_components=5, covariance_type='full', n_init=1, random_state=42)
        self.pc1_sign = 1
        self.pc2_sign = 1
        self.regime_map = {}
        self.final_labels = ["Recovery", "Expansion", "Stagflation", "Contraction"]

    def fit(self, data):
        # Fit on Pre-2020 data
        train_data = data[data.index < '2020-01-01']
        
        # 1. PCA
        self.pca.fit(train_data)
        
        # Check Loadings for Sign Flipping
        components = pd.DataFrame(self.pca.components_, columns=data.columns, index=['PC1', 'PC2'])
        
        # PC1: Growth (should be + correlated with INDPRO)
        indpro_loading = components.loc['PC1', 'INDPRO'] if 'INDPRO' in components.columns else 0
        self.pc1_sign = 1 if indpro_loading > 0 else -1
            
        # PC2: Inflation (should be + correlated with CPIAUCSL or PCEPI)
        # Use CPIAUCSL
        cpi_loading = components.loc['PC2', 'CPIAUCSL'] if 'CPIAUCSL' in components.columns else 0
        if cpi_loading == 0 and 'PCEPI' in components.columns:
             cpi_loading = components.loc['PC2', 'PCEPI']
             
        self.pc2_sign = 1 if cpi_loading > 0 else -1
            
        print(f"PC1 (Growth) Sign: {self.pc1_sign}")
        print(f"PC2 (Inflation) Sign: {self.pc2_sign}")

        # Transform training data for GMM
        pca_train = self.pca.transform(train_data)
        pca_train[:, 0] *= self.pc1_sign
        pca_train[:, 1] *= self.pc2_sign
        
        # 2. GMM
        # Initialize GMM centroids for 5 clusters
        # 0: Expansion (Q1)
        # 1: Stagflation (Q2)
        # 2: Mild Contraction (Q3 Mean)
        # 3: Deep Contraction (Q3 Min/Tail)
        # 4: Recovery (Q4)
        
        q1_mask = (pca_train[:, 0] > 0) & (pca_train[:, 1] > 0) # Expansion
        q2_mask = (pca_train[:, 0] < 0) & (pca_train[:, 1] > 0) # Stagflation
        q3_mask = (pca_train[:, 0] < 0) & (pca_train[:, 1] < 0) # Contraction
        q4_mask = (pca_train[:, 0] > 0) & (pca_train[:, 1] < 0) # Recovery
        
        # Centroids
        c_exp = pca_train[q1_mask].mean(axis=0) if q1_mask.any() else np.array([1.0, 1.0])
        c_stag = pca_train[q2_mask].mean(axis=0) if q2_mask.any() else np.array([-1.0, 1.0])
        c_rec = pca_train[q4_mask].mean(axis=0) if q4_mask.any() else np.array([1.0, -1.0])
        
        # Split Contraction
        if q3_mask.any():
            con_points = pca_train[q3_mask]
            c_con_mild = con_points.mean(axis=0)
            # Use point with min PC1 (Deepest recession) as Deep Contraction
            idx_min = con_points[:, 0].argmin()
            c_con_deep = con_points[idx_min]
        else:
            c_con_mild = np.array([-1.0, -1.0])
            c_con_deep = np.array([-3.0, -3.0])
        
        self.gmm.means_init = np.array([c_exp, c_stag, c_con_mild, c_con_deep, c_rec])
        self.gmm.n_init = 1 
        self.gmm.init_params = 'kmeans'
        
        self.gmm.fit(pca_train)
        
        # 3. Semantic Mapping (Hardcoded by Initialization Order)
        self.regime_map = {
            0: "Expansion",
            1: "Stagflation",
            2: "Contraction",
            3: "Contraction",
            4: "Recovery"
        }
            
        print(f"Regime Map: {self.regime_map}")

    def transform(self, data):
        # PCA
        pca_data = self.pca.transform(data)
        pca_data[:, 0] *= self.pc1_sign
        pca_data[:, 1] *= self.pc2_sign
        
        # GMM Probabilities
        probs = self.gmm.predict_proba(pca_data)
        
        # Create DataFrame with raw cluster columns
        raw_res = pd.DataFrame(probs, index=data.index)
        
        # Aggregate by label (in case multiple clusters map to same label)
        # Initialize result df with 0s for all final labels
        final_res = pd.DataFrame(0.0, index=data.index, columns=self.final_labels)
        
    def transform(self, data):
        # PCA
        pca_data = self.pca.transform(data)
        pca_data[:, 0] *= self.pc1_sign
        pca_data[:, 1] *= self.pc2_sign
        
        # GMM Probabilities
        probs = self.gmm.predict_proba(pca_data)
        
        # Create DataFrame with raw cluster columns
        raw_res = pd.DataFrame(probs, index=data.index)
        
        # Aggregate by label (in case multiple clusters map to same label)
        # Initialize result df with 0s for all final labels
        final_res = pd.DataFrame(0.0, index=data.index, columns=self.final_labels)
        
        for i in range(5): # Loop over 5 trained clusters
            label = self.regime_map[i]
            final_res[label] += raw_res[i]
            
        # Get Hard Assignment but mapped to 4 Regimes for Phase Diagram Color
        # We need the regime name for coloring
        
        # PCA DF for Phase Diagram
        pca_df = pd.DataFrame(pca_data, index=data.index, columns=['Growth', 'Inflation'])
        
        # Add hard assignment column
        hard_cluster = self.gmm.predict(pca_data)
        pca_df['Cluster'] = hard_cluster
        pca_df['Regime'] = pca_df['Cluster'].map(self.regime_map)
        
        return final_res, pca_df
