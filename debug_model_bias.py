
from src.data import DataLoader
from src.models import RegimeModel
import pandas as pd
import numpy as np

def analyze_bias():
    print("--- 1. Data Loading ---")
    loader = DataLoader('2025-11-MD.csv', 'FRED-MD_updated_appendix.csv')
    df = loader.run_pipeline()
    print(f"Data Shape: {df.shape}")
    
    # Check centering of key variables
    print("\n--- 2. Data Centering (Mean of Normalized Data) ---")
    key_vars = ['INDPRO', 'PAYEMS', 'CPIAUCSL', 'PCEPI']
    for var in key_vars:
        if var in df.columns:
            mean_val = df[var].mean()
            min_val = df[var].min()
            max_val = df[var].max()
            print(f"{var}: Mean={mean_val:.4f}, Range=[{min_val:.2f}, {max_val:.2f}]")
    
    print("\n--- 3. Model Training ---")
    model = RegimeModel()
    model.fit(df)
    
    # Get PCA Latent Space
    pca_data = model.pca.transform(df)
    # Apply signs manually to match model logic
    pca_data[:, 0] *= model.pc1_sign
    pca_data[:, 1] *= model.pc2_sign
    
    pca_df = pd.DataFrame(pca_data, index=df.index, columns=['Growth (PC1)', 'Inflation (PC2)'])
    
    print("\n--- 4. PCA Analysis ---")
    print(f"PC1 Sign Applied: {model.pc1_sign}")
    print(f"PC2 Sign Applied: {model.pc2_sign}")
    
    # Correlation Check
    if 'INDPRO' in df.columns:
        corr_growth = np.corrcoef(df['INDPRO'], pca_df['Growth (PC1)'])[0, 1]
        print(f"Corr(INDPRO, PC1_Growth): {corr_growth:.4f} (Expected Positive)")
        
    if 'CPIAUCSL' in df.columns:
        corr_infl = np.corrcoef(df['CPIAUCSL'], pca_df['Inflation (PC2)'])[0, 1]
        print(f"Corr(CPIAUCSL, PC2_Inflation): {corr_infl:.4f} (Expected Positive)")
        
    print("\n--- 5. Quadrant Distribution ---")
    q1 = ((pca_df['Growth (PC1)'] > 0) & (pca_df['Inflation (PC2)'] > 0)).sum()
    q2 = ((pca_df['Growth (PC1)'] < 0) & (pca_df['Inflation (PC2)'] > 0)).sum()
    q3 = ((pca_df['Growth (PC1)'] < 0) & (pca_df['Inflation (PC2)'] < 0)).sum()
    q4 = ((pca_df['Growth (PC1)'] > 0) & (pca_df['Inflation (PC2)'] < 0)).sum()
    total = len(pca_df)
    
    print(f"Q1 (Expansion +/+): {q1} ({q1/total:.1%})")
    print(f"Q2 (Stagflation -/+): {q2} ({q2/total:.1%})")
    print(f"Q3 (Contraction -/-): {q3} ({q3/total:.1%})")
    print(f"Q4 (Recovery +/-): {q4} ({q4/total:.1%})")

    # Check GMM Means
    print("\n--- 6. GMM Means ---")
    means = model.gmm.means_
    for i in range(4):
        print(f"Cluster {i}: ({means[i, 0]:.2f}, {means[i, 1]:.2f}) -> {model.regime_map[i]}")

if __name__ == "__main__":
    analyze_bias()
