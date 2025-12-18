import pandas as pd
import numpy as np
from src.data import DataLoader
from src.models import RegimeModel

def inspect_pca():
    print("--- Loading Data and Model ---")
    loader = DataLoader('2025-11-MD.csv', 'FRED-MD_updated_appendix.csv')
    df = loader.run_pipeline()
    
    # Calculate Inflation Proxy (CPI YoY)
    cpi_raw = loader.raw_df['CPIAUCSL']
    inflation_yoy = cpi_raw.pct_change(12) * 100
    inflation_yoy = inflation_yoy.reindex(df.index).fillna(method='ffill')

    model = RegimeModel()
    model.fit(df, inflation_series=inflation_yoy)
    _, pca_df = model.transform(df, inflation_series=inflation_yoy)
    
    print("\n--- 1990s Expansion Analysis (1995-2000) ---")
    subset_90s = pca_df['1995-01-01':'2000-12-01']
    print(subset_90s[['Growth', 'Inflation', 'Cluster', 'Regime']].describe())
    print("\nRegime Counts:")
    print(subset_90s['Regime'].value_counts())
    print("\nCluster Counts:")
    print(subset_90s['Cluster'].value_counts())
    
    print("\n--- 2010s Expansion Analysis ---")
    subset_10s = pca_df['2012-01-01':'2019-12-01']
    print(subset_10s[['Growth', 'Inflation', 'Cluster', 'Regime']].describe())
    print("\nRegime Counts:")
    print(subset_10s['Regime'].value_counts())
    print("\nCluster Counts:")
    print(subset_10s['Cluster'].value_counts())
    
    print("\n--- Centroids ---")
    means = model.gmm.means_
    n_features = means.shape[1]
    
    for i in range(len(means)):
        # Handle 4D: Growth, Accel, Infl, Mom. Or 3D.
        regime_label = model.regime_map.get(i, 'Unknown')
        
        if n_features == 4:
            # [Growth, Accel, Infl, Mom]
            # Mom is index 3
            g = means[i, 0]
            a = means[i, 1]
            infl = means[i, 2]
            mom = means[i, 3]
            print(f"Cluster {i}: Growth={g:.2f}, Accel={a:.2f}, Infl={infl:.2f}, Mom={mom:.2f} -> {regime_label}")
        elif n_features == 3:
             # Assumed [Growth, Accel, Level]
             print(f"Cluster {i}: Growth={means[i,0]:.2f}, Accel={means[i,1]:.2f}, Level={means[i,2]:.2f} -> {regime_label}")
        else:
             print(f"Cluster {i}: Growth={means[i,0]:.2f}, Inflation={means[i,1]:.2f} -> {regime_label}")

if __name__ == "__main__":
    inspect_pca()
