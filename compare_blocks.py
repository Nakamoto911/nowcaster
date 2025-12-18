import pandas as pd
import numpy as np
from src.data import DataLoader
from src.models import RegimeModel, RegimeHMM

def start_end_blocks(series):
    """
    Convert a datetime-indexed series of regimes into a DataFrame of start/end blocks.
    """
    if series.empty:
        return pd.DataFrame(columns=['Start', 'End', 'Regime'])
        
    df = series.to_frame(name='Regime')
    df.index.name = 'Date' # specific name
    df['block_id'] = (df['Regime'] != df['Regime'].shift()).cumsum()
    
    blocks = df.reset_index().groupby('block_id').agg(
        Regime=('Regime', 'first'),
        Start=('Date', 'min'),
        End=('Date', 'max')
    ).reset_index(drop=True)
    
    return blocks

def compare_blocks():
    print("--- Loading Data and Model ---")
    loader = DataLoader('2025-11-MD.csv', 'FRED-MD_updated_appendix.csv')
    df = loader.run_pipeline()
    
    # Calculate Inflation Proxy for 3D Model
    cpi_raw = loader.raw_df['CPIAUCSL']
    inflation_yoy = cpi_raw.pct_change(12) * 100
    inflation_yoy = inflation_yoy.reindex(df.index).fillna(method='ffill')
    
    # --- MODEL 1: GMM ---
    print("--- Fitting GMM ---")
    model_gmm = RegimeModel()
    model_gmm.fit(df, inflation_series=inflation_yoy)
    _, pca_df_gmm = model_gmm.transform(df, inflation_series=inflation_yoy)
    
    # --- MODEL 2: HMM ---
    print("--- Fitting HMM (K=6) ---")
    model_hmm = RegimeHMM(n_components=6)
    model_hmm.fit(df, inflation_series=inflation_yoy)
    _, pca_df_hmm = model_hmm.transform(df, inflation_series=inflation_yoy)
    
    # 1. Get Predicted Blocks
    blocks_gmm = start_end_blocks(pca_df_gmm['Regime'])
    blocks_hmm = start_end_blocks(pca_df_hmm['Regime'])
    
    # 2. Get Ground Truth Blocks
    gt_df = pd.read_csv('ground_truth.csv')
    gt_df['Start Date'] = pd.to_datetime(gt_df['Start Date'])
    gt_df['End Date'] = pd.to_datetime(gt_df['End Date'])

    # 3. Create Comparison Text
    output_lines = []
    output_lines.append("--- Regime Comparison Report ---")
    
    output_lines.append("\n=== 1. GMM PREDICTED BLOCKS ===")
    output_lines.append(blocks_gmm.to_string())
    
    output_lines.append("\n\n=== 2. HMM PREDICTED BLOCKS ===")
    output_lines.append(blocks_hmm.to_string())
    
    output_lines.append("\n\n=== 3. GROUND TRUTH BLOCKS ===")
    output_lines.append(gt_df[['Start Date', 'End Date', 'Regime', 'Notes']].to_string())

    # 4. Accuracy Analysis
    # Expand GT to monthly
    gt_monthly = []
    for _, row in gt_df.iterrows():
        dates = pd.date_range(start=row['Start Date'], end=row['End Date'], freq='MS')
        regime = row['Regime']
        for d in dates:
            gt_monthly.append({'Date': d, 'Regime_GT': regime})
            
    gt_series_df = pd.DataFrame(gt_monthly).set_index('Date')
    
    # Align Both Models
    merged = gt_series_df.join(pca_df_gmm[['Regime']].rename(columns={'Regime': 'GMM'}), how='inner')
    merged = merged.join(pca_df_hmm[['Regime']].rename(columns={'Regime': 'HMM'}), how='inner')
    
    # Calculate Stats
    regimes = ["Contraction", "Expansion", "Stagflation", "Recovery"]
    stats_data = []
    
    total_matches_gmm = 0
    total_matches_hmm = 0
    total_months = len(merged)
    
    for r in regimes:
        # Filter for this regime in GT
        gt_subset = merged[merged['Regime_GT'] == r]
        gt_count = len(gt_subset)
        
        # GMM Stats
        gmm_count = len(merged[merged['GMM'] == r])
        gmm_correct = len(gt_subset[gt_subset['GMM'] == r])
        gmm_recall = gmm_correct / gt_count if gt_count > 0 else 0.0
        
        # HMM Stats
        hmm_count = len(merged[merged['HMM'] == r])
        hmm_correct = len(gt_subset[gt_subset['HMM'] == r])
        hmm_recall = hmm_correct / gt_count if gt_count > 0 else 0.0
        
        stats_data.append({
            'Regime': r,
            'GT Months': gt_count,
            'GMM Months': gmm_count,
            'GMM Acc': f"{gmm_recall:.1%}",
            'HMM Months': hmm_count,
            'HMM Acc': f"{hmm_recall:.1%}"
        })
        total_matches_gmm += gmm_correct
        total_matches_hmm += hmm_correct

    stats_df = pd.DataFrame(stats_data)
    
    output_lines.append("\n\n=== 4. Side-by-Side Performance Analysis ===")
    output_lines.append(f"Total Aligned Months: {total_months}")
    output_lines.append(f"GMM Overall Accuracy: {total_matches_gmm/total_months:.1%}")
    output_lines.append(f"HMM Overall Accuracy: {total_matches_hmm/total_months:.1%}")
    output_lines.append("\nPer-Regime Comparison (Recall):")
    output_lines.append(stats_df.to_string(index=False))

    # Write to file
    with open('regime_comparison.txt', 'w') as f:
        f.write('\n'.join(output_lines))
    
    print("Side-by-side comparison saved to regime_comparison.txt")

if __name__ == "__main__":
    compare_blocks()
