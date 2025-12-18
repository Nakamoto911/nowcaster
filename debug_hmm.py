
import pandas as pd
import numpy as np
from src.data import DataLoader
from src.models import RegimeHMM

def debug_hmm():
    print("--- Loading Data ---")
    loader = DataLoader('2025-11-MD.csv', 'FRED-MD_updated_appendix.csv')
    df = loader.run_pipeline()
    
    # Load Inflation
    cpi_raw = loader.raw_df['CPIAUCSL']
    inflation_yoy = cpi_raw.pct_change(12) * 100
    inflation_yoy = inflation_yoy.reindex(df.index).fillna(method='ffill')
    
    print("--- Fitting HMM (K=6) ---")
    hmm_model = RegimeHMM(n_components=6)
    hmm_model.fit(df, inflation_series=inflation_yoy)
    
    # Predict blocks
    _, pca_df = hmm_model.transform(df, inflation_series=inflation_yoy)
    pca_df['block_id'] = (pca_df['Cluster'] != pca_df['Cluster'].shift()).cumsum()
    
    # Handle index
    df_reset = pca_df.reset_index()
    # Force rename first col to Date to be safe
    df_reset.columns.values[0] = 'Date'
    
    blocks = df_reset.groupby('block_id').agg(
        Cluster=('Cluster', 'first'),
        Start=('Date', 'min'),
        End=('Date', 'max'),
        Count=('Date', 'count')
    )
    
    print("\n--- State 1 (Low Growth) Blocks ---")
    state_1_blocks = blocks[blocks['Cluster'] == 1]
    print(state_1_blocks[['Start', 'End', 'Count']].to_string())

if __name__ == "__main__":
    debug_hmm()
