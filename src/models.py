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
            
            # --- 1. PCA Fitting ---
            self.pca.fit(train_data)
            
            # Check Loadings for Sign Flipping
            components = pd.DataFrame(self.pca.components_, columns=data.columns, index=['PC1', 'PC2'])
            
            # PC1: Growth
            indpro_loading = components.loc['PC1', 'INDPRO'] if 'INDPRO' in components.columns else 0
            self.pc1_sign = 1 if indpro_loading > 0 else -1
                
            # PC2: Inflation
            cpi_loading = components.loc['PC2', 'CPIAUCSL'] if 'CPIAUCSL' in components.columns else 0
            if cpi_loading == 0 and 'PCEPI' in components.columns:
                cpi_loading = components.loc['PC2', 'PCEPI']
            self.pc2_sign = 1 if cpi_loading > 0 else -1
                
            print(f"PC1 (Growth) Sign: {self.pc1_sign}")
            print(f"PC2 (Inflation) Sign: {self.pc2_sign}")

            # Transform training data
            pca_train = self.pca.transform(train_data)
            pca_train[:, 0] *= self.pc1_sign
            pca_train[:, 1] *= self.pc2_sign
            
            # --- 2. GMM Fitting ---
            self.gmm.means_init = None 
            self.gmm.n_init = 10 
            self.gmm.init_params = 'kmeans'
            self.gmm.fit(pca_train)
            
            # --- 3. Robust Semantic Mapping (Draft Logic) ---
            self.regime_map = {}
            means = self.gmm.means_
            
            # Initial split: Right side (Growth) vs Left side (Downturn)
            high_growth_indices = []
            low_growth_indices = []
            
            for i in range(self.gmm.n_components):
                if means[i, 0] > -0.1: 
                    high_growth_indices.append(i)
                else:
                    low_growth_indices.append(i)

            # RECOVERY RESCUE: 
            # If we have 0 or 1 growth cluster, we can't form both Recovery and Expansion.
            # We must "draft" the best cluster from low_growth (highest PC1) to be Recovery.
            if len(high_growth_indices) < 2 and len(low_growth_indices) > 0:
                # Find the cluster in low_growth with the highest Growth Score (closest to positive)
                best_candidate = sorted(low_growth_indices, key=lambda x: means[x, 0])[-1]
                
                # Move it to high_growth
                low_growth_indices.remove(best_candidate)
                high_growth_indices.append(best_candidate)
                print(f"Drafted Cluster {best_candidate} into Growth Group to ensure Recovery exists.")
                    
            # --- Assign Growth Regimes (Recovery / Expansion) ---
            if high_growth_indices:
                # Sort by Inflation (PC2)
                sorted_growth = sorted(high_growth_indices, key=lambda x: means[x, 1])
                
                # Lowest Inflation -> Recovery
                self.regime_map[sorted_growth[0]] = "Recovery"
                
                # Highest Inflation -> Expansion
                self.regime_map[sorted_growth[-1]] = "Expansion"
                
                # Handle Middle Clusters (if any exist between Rec and Exp)
                if len(sorted_growth) > 2:
                    for mid_idx in sorted_growth[1:-1]:
                        if means[mid_idx, 1] > 0:
                            self.regime_map[mid_idx] = "Expansion"
                        else:
                            self.regime_map[mid_idx] = "Recovery"

            # --- Assign Downturn Regimes (Stagflation / Contraction) ---
            if low_growth_indices:
                # Sort by Inflation (PC2)
                sorted_downturn = sorted(low_growth_indices, key=lambda x: means[x, 1])
                
                # Check the highest inflation candidate
                stag_candidate = sorted_downturn[-1]
                lowest_candidate = sorted_downturn[0]
                
                # Logic: Is the "Stag" candidate actually inflationary?
                # It must be notably higher than the deep contraction, OR simply positive.
                is_high_inflation = means[stag_candidate, 1] > 0
                is_much_higher = means[stag_candidate, 1] > means[lowest_candidate, 1] + 1.0
                
                if is_high_inflation or is_much_higher:
                    self.regime_map[stag_candidate] = "Stagflation"
                    remaining = sorted_downturn[:-1]
                else:
                    # If no inflation pressure, everything is Contraction
                    remaining = sorted_downturn

                for idx in remaining:
                    self.regime_map[idx] = "Contraction"

            print(f"Cluster Centroids: {means}")
            print(f"Dynamic Regime Map: {self.regime_map}")
    
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
        
    def get_regime_blocks(self, pca_df, min_duration=3):
        """
        Aggregates monthly regime labels into stable time blocks.
        Merges short-lived 'flickers' (< min_duration) into the previous regime.
        """
        # Work on a copy to avoid affecting the original df
        df = pca_df[['Regime']].copy().sort_index()
        df.index.name = 'index' # Ensure index is named 'index' for reset_index
        
        # Identify Groups: Create a unique ID for each continuous block
        df['block_id'] = (df['Regime'] != df['Regime'].shift()).cumsum()
        
        # Initial Aggregation
        blocks = df.reset_index().groupby('block_id').agg(
            Regime=('Regime', 'first'),
            Start=('index', 'min'), # note: index is the date
            End=('index', 'max'),
            Months=('Regime', 'count')
        ).reset_index(drop=True)
        
        # Smoothing Loop (Hysteresis)
        clean_blocks = []
        if blocks.empty: return pd.DataFrame()
        
        current = blocks.iloc[0].to_dict()
        
        for i in range(1, len(blocks)):
            next_block = blocks.iloc[i].to_dict()
            
            # IF next block is Noise (< min_dur) OR same as current (Merge):
            if (next_block['Months'] < min_duration) or (next_block['Regime'] == current['Regime']):
                # Extend current block
                current['End'] = next_block['End']
                current['Months'] += next_block['Months']
            else:
                # Save current, start new
                clean_blocks.append(current)
                current = next_block
                
        clean_blocks.append(current)
        
        return pd.DataFrame(clean_blocks)

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
