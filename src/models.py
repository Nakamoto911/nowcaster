from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import pandas as pd
import numpy as np

class RegimeModel:
    def __init__(self):
        self.pca = PCA(n_components=2)
        # Use 10 components to aggressively split the "Low Growth" density
        self.gmm = GaussianMixture(n_components=10, covariance_type='full', n_init=10, random_state=42)
        self.pc1_sign = 1
        self.pc2_sign = 1
        self.regime_map = {}
        self.final_labels = ["Recovery", "Expansion", "Stagflation", "Contraction"]

    def fit(self, data, inflation_series=None):
        # 1. PCA
        self.pca.fit(data)
        
        # Check signs
        components = pd.DataFrame(self.pca.components_, columns=data.columns)
        if 'IPMANSICS' in data.columns:
            if components.loc[0, 'IPMANSICS'] < 0: self.pc1_sign = -1
        if 'CPIAUCSL' in data.columns:
            if components.loc[1, 'CPIAUCSL'] < 0: self.pc2_sign = -1
            
        pca_train = self.pca.transform(data)
        pca_train[:, 0] *= self.pc1_sign
        pca_train[:, 1] *= self.pc2_sign
        
        # 2. Feature Augmentation
        features_list = [pca_train] # [Growth, Accel]
        
        # A. Inflation Level
        if inflation_series is not None:
            infl_aligned = inflation_series.reindex(data.index).ffill().fillna(0)
            infl_mean = infl_aligned.mean()
            infl_std = infl_aligned.std()
            self.infl_stats = {'mean': infl_mean, 'std': infl_std}
            infl_z = (infl_aligned - infl_mean) / (infl_std + 1e-6)
            features_list.append(infl_z.values.reshape(-1, 1))
        else:
            self.infl_stats = None
            
        # B. Growth Momentum (PC1 Change)
        # Calculate on the PCA training data
        pc1_series = pd.Series(pca_train[:, 0], index=data.index)
        # 3-month smoothed momentum
        momentum = pc1_series.diff(3).fillna(0)
        
        mom_mean = momentum.mean()
        mom_std = momentum.std()
        self.mom_stats = {'mean': mom_mean, 'std': mom_std}
        
        mom_z = (momentum - mom_mean) / (mom_std + 1e-6)
        features_list.append(mom_z.values.reshape(-1, 1))
        
        # Combine: [Growth, Accel, (Infl), Momentum]
        train_features = np.column_stack(features_list)

        # 3. Fit GMM
        try:
            self.gmm.fit(train_features)
        except Exception:
            self.gmm.reg_covar = 1e-4
            self.gmm.init_params = 'kmeans'
            self.gmm.fit(train_features)
            
        # 4. robust Semantic Mapping (4D)
        self.regime_map = {}
        means = self.gmm.means_
        n_feats = means.shape[1]
        
        # Indices for features
        # 0: Growth
        # 1: Accel
        # 2: Inflation (if present)
        # 3: Momentum (if Infl present, else 2)
        
        has_infl = (self.infl_stats is not None)
        idx_growth = 0
        idx_infl = 2 if has_infl else -1
        idx_mom = 3 if has_infl else 2
        
        # 1. Contraction Priority: Growth < -2.0
        remaining = []
        for i in range(self.gmm.n_components):
            if means[i, idx_growth] < -2.0:
                self.regime_map[i] = "Contraction"
            else:
                remaining.append(i)
                
        # 2. Score remaining clusters
        for idx in remaining:
            growth = means[idx, idx_growth]
            infl = means[idx, idx_infl] if idx_infl != -1 else 0
            mom = means[idx, idx_mom]
            
            # Logic Hierarchy
            
            # A. Stagflation: High Inflation + Not Deep Contraction
            if idx_infl != -1 and infl > 0.5:
                # Prioritize Stagflation BUT check if it's actually High Mom Recovery?
                # Sometimes Recovery has inflation? 
                # Let's keep it simple: if Infl is high, it's Stagflation unless Momentum is HUGE.
                self.regime_map[idx] = "Stagflation"
                continue
                
            # B. Recovery: High Momentum + Moderate Growth
            # 1. High Momentum (> 0.5) AND Growth < 0.5 (Not Boom)
            # 2. OR: Below Trend Growth (< -0.5) AND Positive Momentum (> 0.1)
            
            # Cluster 0 (K=10): Growth 0.25, Mom 1.09 -> Match Rule 1
            # Cluster 1 (K=10): Growth -1.29, Mom -0.09 -> Fails both (Expansion/Stagnation)
            
            if (mom > 0.5 and growth < 0.5) or (growth < -0.5 and mom > 0.1):
                self.regime_map[idx] = "Recovery"
                continue
                
            # C. Expansion: The rest
            self.regime_map[idx] = "Expansion"

        print(f"Cluster Centroids (4D): \n{means}")
        print(f"Dynamic Regime Map: {self.regime_map}")
    

        
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

    def transform(self, data, inflation_series=None):
        # PCA
        pca_data = self.pca.transform(data)
        pca_data[:, 0] *= self.pc1_sign
        pca_data[:, 1] *= self.pc2_sign
        
        features_list = [pca_data] # [Growth, Accel]
        
        # Feature Augmentation
        # A. Inflation
        if inflation_series is not None and self.infl_stats is not None:
             infl_aligned = inflation_series.reindex(data.index).ffill().fillna(0)
             infl_z = (infl_aligned - self.infl_stats['mean']) / (self.infl_stats['std'] + 1e-6)
             features_list.append(infl_z.values.reshape(-1, 1))
             
        # B. Momentum
        if hasattr(self, 'mom_stats'):
             pc1_series = pd.Series(pca_data[:, 0], index=data.index)
             momentum = pc1_series.diff(3).fillna(0)
             mom_z = (momentum - self.mom_stats['mean']) / (self.mom_stats['std'] + 1e-6)
             features_list.append(mom_z.values.reshape(-1, 1))
             
        features = np.column_stack(features_list)
        
        # GMM Probabilities
        probs = self.gmm.predict_proba(features)
        
        # Create DataFrame with raw cluster columns
        raw_res = pd.DataFrame(probs, index=data.index)
        
        # Aggregate by label
        final_res = pd.DataFrame(0.0, index=data.index, columns=self.final_labels)
        
        for i in range(self.gmm.n_components):
            label = self.regime_map[i]
            final_res[label] += raw_res[i]
            
        # Hard Assignment
        pca_df = pd.DataFrame(pca_data, index=data.index, columns=['Growth', 'Inflation'])
        hard_cluster = self.gmm.predict(features)
        pca_df['Cluster'] = hard_cluster
        pca_df['Regime'] = pca_df['Cluster'].map(self.regime_map)
        
        return final_res, pca_df

from hmmlearn import hmm

class RegimeHMM:
    def __init__(self, n_components=6):
        self.pca = PCA(n_components=2)
        # HMM with Gaussian emissions
        # n_components=6 to allow for transient states (Early/Late Recovery)
        self.n_components = n_components
        self.model = hmm.GaussianHMM(n_components=n_components, covariance_type='full', n_iter=100, random_state=42, init_params='stmc')
        self.pc1_sign = 1
        self.pc2_sign = 1
        self.regime_map = {}
        self.final_labels = ["Recovery", "Expansion", "Stagflation", "Contraction"]
        self.infl_stats = None
        self.mom_stats = None

    def fit(self, data, inflation_series=None):
        # 1. PCA Logic (Shared)
        self.pca.fit(data)
        
        components = pd.DataFrame(self.pca.components_, columns=data.columns)
        if 'IPMANSICS' in data.columns:
            if components.loc[0, 'IPMANSICS'] < 0: self.pc1_sign = -1
        if 'CPIAUCSL' in data.columns:
            if components.loc[1, 'CPIAUCSL'] < 0: self.pc2_sign = -1
            
        pca_train = self.pca.transform(data)
        pca_train[:, 0] *= self.pc1_sign
        pca_train[:, 1] *= self.pc2_sign
        
        # 2. Feature Augmentation (4D)
        features_list = [pca_train]
        
        # Inflation
        if inflation_series is not None:
            infl_aligned = inflation_series.reindex(data.index).ffill().fillna(0)
            infl_mean = infl_aligned.mean()
            infl_std = infl_aligned.std()
            self.infl_stats = {'mean': infl_mean, 'std': infl_std}
            infl_z = (infl_aligned - infl_mean) / (infl_std + 1e-6)
            features_list.append(infl_z.values.reshape(-1, 1))
        else:
            self.infl_stats = None
            
        # Momentum
        pc1_series = pd.Series(pca_train[:, 0], index=data.index)
        momentum = pc1_series.diff(3).fillna(0)
        mom_mean = momentum.mean()
        mom_std = momentum.std()
        self.mom_stats = {'mean': mom_mean, 'std': mom_std}
        mom_z = (momentum - mom_mean) / (mom_std + 1e-6)
        features_list.append(mom_z.values.reshape(-1, 1))
        
        train_features = np.column_stack(features_list)
        
        # 3. Fit HMM
        # hmmlearn sometimes needs careful init
        self.model.fit(train_features)
        
        # 4. Topological Mapping
        self.regime_map = {}
        means = self.model.means_
        transmat = self.model.transmat_
        
        # Indices
        has_infl = (self.infl_stats is not None)
        idx_growth = 0
        idx_infl = 2 if has_infl else -1
        idx_mom = 3 if has_infl else 2
        
        # A. Identify Contraction (Lowest Growth)
        sorted_by_growth = np.argsort(means[:, idx_growth])
        contraction_idx = sorted_by_growth[0] 
        
        # Check deep contraction
        contraction_candidates = []
        for i in range(self.n_components):
            if means[i, idx_growth] < -2.0:
                self.regime_map[i] = "Contraction"
                contraction_candidates.append(i)
        
        if not contraction_candidates:
             self.regime_map[contraction_idx] = "Contraction"
             contraction_candidates.append(contraction_idx)

        # B. Identify Recovery (Group Transition Logic)
        # Find the state that the Contraction Group transitions to most frequently
        
        # 1. Sum outgoing probabilities from Contraction Set to Non-Contraction Set
        transition_scores = np.zeros(self.n_components)
        
        for c_idx in contraction_candidates:
            row = transmat[c_idx]
            for dest_idx, prob in enumerate(row):
                if dest_idx not in contraction_candidates:
                    transition_scores[dest_idx] += prob
                    
        # 2. Candidate is the one with highest score
        if np.max(transition_scores) > 0:
            recovery_candidate = np.argmax(transition_scores)
        else:
            recovery_candidate = -1

        remaining = [i for i in range(self.n_components) if i not in self.regime_map]
        
        for idx in remaining:
            growth = means[idx, idx_growth]
            infl = means[idx, idx_infl] if idx_infl != -1 else 0
            mom = means[idx, idx_mom]
            
            # Stagflation: High Inflation + Not Contraction
            if idx_infl != -1 and infl > 0.5:
                 self.regime_map[idx] = "Stagflation"
                 continue
            
            # Recovery Check
            # Rule 1: Topological Successor to Contraction Group
            is_successor = (idx == recovery_candidate)
            
            # Rule 2: High Momentum + Moderate Growth (Static Backup)
            # Relaxed momentum for HMM states which averaging
            is_high_mom = (mom > 0.1 and growth < 0.0)
            
            if is_successor:
                 # Ensure it's not a Boom state (Growth > 1.0)
                 if growth < 1.0:
                     self.regime_map[idx] = "Recovery"
                 else:
                     self.regime_map[idx] = "Expansion"
            elif is_high_mom:
                self.regime_map[idx] = "Recovery"
            else:
                self.regime_map[idx] = "Expansion"
                
        print(f"HMM Means (K={self.n_components}): \n{means}")
        # print(f"HMM TransMat: \n{transmat}")
        print(f"HMM Map: {self.regime_map}")
        
    def get_regime_blocks(self, pca_df, min_duration=3):
        """
        Aggregates monthly regime labels into stable time blocks.
        """
        # Work on a copy to avoid affecting the original df
        df = pca_df[['Regime']].copy().sort_index()
        df.index.name = 'index' # Ensure index is named 'index' for reset_index
        
        # Identify Groups
        df['block_id'] = (df['Regime'] != df['Regime'].shift()).cumsum()
        
        # Initial Aggregation
        blocks = df.reset_index().groupby('block_id').agg(
            Regime=('Regime', 'first'),
            Start=('index', 'min'), 
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
        
    def _apply_duration_filter(self, pca_df, max_recovery_months=24):
        """
        Post-processing heuristic:
        If a Recovery block is longer than max_recovery_months, it's likely "Slow Expansion".
        Reclassify it as Expansion.
        """
        df = pca_df.copy()
        # Identify blocks
        df['block_id'] = (df['Regime'] != df['Regime'].shift()).cumsum()
        
        # Iterate over blocks
        for block_id, group in df.groupby('block_id'):
            regime = group['Regime'].iloc[0]
            if regime == "Recovery":
                duration = len(group)
                if duration > max_recovery_months:
                    # Reclassify to Expansion
                    df.loc[group.index, 'Regime'] = "Expansion"
                    
        return df['Regime']

    def transform(self, data, inflation_series=None):
        # PCA
        pca_data = self.pca.transform(data)
        pca_data[:, 0] *= self.pc1_sign
        pca_data[:, 1] *= self.pc2_sign
        
        features_list = [pca_data]
        
        if inflation_series is not None and self.infl_stats is not None:
             infl_aligned = inflation_series.reindex(data.index).ffill().fillna(0)
             infl_z = (infl_aligned - self.infl_stats['mean']) / (self.infl_stats['std'] + 1e-6)
             features_list.append(infl_z.values.reshape(-1, 1))
             
        if hasattr(self, 'mom_stats'):
             pc1_series = pd.Series(pca_data[:, 0], index=data.index)
             momentum = pc1_series.diff(3).fillna(0)
             mom_z = (momentum - self.mom_stats['mean']) / (self.mom_stats['std'] + 1e-6)
             features_list.append(mom_z.values.reshape(-1, 1))
             
        features = np.column_stack(features_list)
        
        # Predict State Sequence (Viterbi)
        hidden_states = self.model.predict(features)
        
        # Posterior Probabilities
        probs = self.model.predict_proba(features)
        
        # Create Output
        final_res = pd.DataFrame(0.0, index=data.index, columns=self.final_labels)
        
        for i in range(self.n_components):
            label = self.regime_map.get(i, "Expansion") 
            final_res[label] += probs[:, i]
            
        # Hard Assignment
        pca_df = pd.DataFrame(pca_data, index=data.index, columns=['Growth', 'Inflation'])
        pca_df['Cluster'] = hidden_states
        pca_df['Regime'] = pca_df['Cluster'].map(self.regime_map)
        
        # Apply Duration Heuristic (Recovery > 24m -> Expansion)
        pca_df['Regime'] = self._apply_duration_filter(pca_df, max_recovery_months=24)
        
        return final_res, pca_df
