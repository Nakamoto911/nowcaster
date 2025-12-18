import pandas as pd
import numpy as np
from src.data import DataLoader
from src.models import RegimeModel, RegimeHMM
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

def load_ground_truth(filepath):
    """
    Expands the start/end date CSV into a monthly timeseries.
    """
    df = pd.read_csv(filepath)
    df['Start Date'] = pd.to_datetime(df['Start Date'])
    df['End Date'] = pd.to_datetime(df['End Date'])
    
    # Create a monthly index covering the full range
    min_date = df['Start Date'].min()
    max_date = pd.Timestamp.now() # Until now
    
    dates = pd.date_range(start=min_date, end=max_date, freq='MS')
    gt_series = pd.Series(index=dates, dtype='object')
    
    for _, row in df.iterrows():
        mask = (gt_series.index >= row['Start Date']) & (gt_series.index < row['End Date'])
        gt_series.loc[mask] = row['Regime']
        
    return gt_series.dropna()

def benchmark():
    print("--- Loading Data ---")
    loader = DataLoader('2025-11-MD.csv', 'FRED-MD_updated_appendix.csv')
    df_model = loader.run_pipeline()
    
    # Load Inflation for Override
    cpi_raw = loader.raw_df['CPIAUCSL']
    inflation_yoy = cpi_raw.pct_change(12) * 100
    inflation_yoy = inflation_yoy.reindex(df_model.index).fillna(method='ffill')

    print("--- Loading Ground Truth ---")
    gt_series = load_ground_truth('ground_truth.csv')

    print("\n=== MODEL 1: GMM (Benchmark) ===")
    gmm = RegimeModel()
    gmm.fit(df_model, inflation_series=inflation_yoy)
    _, pca_df_gmm = gmm.transform(df_model, inflation_series=inflation_yoy)
    pred_gmm = pca_df_gmm['Regime']
    
    # Align GMM
    common_idx = gt_series.index.intersection(pred_gmm.index)
    y_true = gt_series.loc[common_idx]
    y_gmm = pred_gmm.loc[common_idx]
    
    acc_gmm = accuracy_score(y_true, y_gmm)
    print(f"GMM Overall Accuracy: {acc_gmm:.2%}")

    print("\n=== MODEL 2: HMM (Challenger) ===")
    # Reverting to K=6 as it successfully found a "Recovery-like" state (State 1),
    # even though it conflated it with 2010s. This highlights the trade-off best.
    hmm_model = RegimeHMM(n_components=6)
    hmm_model.fit(df_model, inflation_series=inflation_yoy)
    _, pca_df_hmm = hmm_model.transform(df_model, inflation_series=inflation_yoy)
    pred_hmm = pca_df_hmm['Regime']
    
    # Align HMM
    y_hmm = pred_hmm.loc[common_idx]
    
    acc_hmm = accuracy_score(y_true, y_hmm)
    print(f"HMM Overall Accuracy: {acc_hmm:.2%}")
    
    print("\n=== SIDE-BY-SIDE COMPARISON ===")
    
    # Create combined DF
    comparison = pd.DataFrame({
        'Ground Truth': y_true,
        'GMM': y_gmm,
        'HMM': y_hmm
    })
    
    # Accuracy per Regime (Recall)
    print("\n--- Recall by Regime ---")
    regimes = ['Contraction', 'Expansion', 'Recovery', 'Stagflation']
    
    metrics = []
    for r in regimes:
        # Ground Truth Count
        gt_subset = comparison[comparison['Ground Truth'] == r]
        gt_count = len(gt_subset)
        
        # GMM Correct
        gmm_correct = len(gt_subset[gt_subset['GMM'] == r])
        gmm_recall = gmm_correct / gt_count if gt_count > 0 else 0
        
        # HMM Correct
        hmm_correct = len(gt_subset[gt_subset['HMM'] == r])
        hmm_recall = hmm_correct / gt_count if gt_count > 0 else 0
        
        metrics.append({
            'Regime': r,
            'Months': gt_count,
            'GMM Recall': f"{gmm_recall:.1%}",
            'HMM Recall': f"{hmm_recall:.1%}"
        })
        
    print(pd.DataFrame(metrics).to_string(index=False))
    
    # Specific Periods
    print("\n--- Era Deep Dive ---")
    eras = [
        ('1991-03-01', '1992-06-01', 'Jobless Recovery'),
        ('2009-06-01', '2011-09-01', 'Post-GFC Recovery'),
        ('2011-09-01', '2020-02-01', '2010s Expansion')
    ]
    
    for start, end, name in eras:
        print(f"\n{name} ({start} to {end}):")
        sub = comparison.loc[start:end]
        
        # Calculate Accuracy for this period
        gmm_acc = (sub['Ground Truth'] == sub['GMM']).mean()
        hmm_acc = (sub['Ground Truth'] == sub['HMM']).mean()
        
        print(f"  GMM Accuracy: {gmm_acc:.1%}")
        if gmm_acc < 0.8:
            print(f"  GMM Predictions: {sub['GMM'].value_counts().to_dict()}")
            
        print(f"  HMM Accuracy: {hmm_acc:.1%}")
        if hmm_acc < 0.8:
            print(f"  HMM Predictions: {sub['HMM'].value_counts().to_dict()}")

if __name__ == "__main__":
    benchmark()
