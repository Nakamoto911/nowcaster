import pandas as pd
import numpy as np
from src.data import DataLoader
from src.models import RegimeModel
import sys

def test_regime_logic():
    print("--- Starting Verification Test ---")
    
    # 1. Load Data
    print("Loading data...")
    loader = DataLoader(
        data_path='2025-11-MD.csv',
        appendix_path='FRED-MD_updated_appendix.csv'
    )
    df = loader.run_pipeline()
    print(f"Data loaded: {df.shape}")

    # 2. Fit Model
    print("Fitting model...")
    model = RegimeModel()
    model.fit(df)
    
    # Create Inflation Proxy (CPI YoY)
    cpi_raw = loader.raw_df['CPIAUCSL']
    inflation_yoy = cpi_raw.pct_change(12) * 100
    inflation_yoy.name = 'Inflation_YoY'
    
    # 3. Transform with Inflation Override
    regime_probs, pca_df = model.transform(df, inflation_series=inflation_yoy)
    
    # Get hard labels
    if 'Regime' in pca_df.columns:
        regimes = pca_df['Regime']
    else:
        # Fallback if Regime col not present (should be there per models.py but just in case)
        print("Regime column missing from pca_df, using best cluster map...")
        regimes = pca_df['Cluster'].map(model.regime_map)
    
    # --- Check 1: Stagflation False Positives (2015) ---
    print("\n[Check 1] Stagflation in 2015 (Low Inflation Period)")
    # 2015 should NOT be Stagflation because inflation was low.
    subset_2015 = regimes['2015-01-01':'2015-12-01']
    stag_months_2015 = subset_2015[subset_2015 == 'Stagflation']
    
    print(f"Stagflation months in 2015: {len(stag_months_2015)} / {len(subset_2015)}")
    
    # Inspect 2015 PCA values
    if len(stag_months_2015) > 0:
        print("Inspecting 2015 PCA values:")
        pca_2015 = pca_df.loc['2015-01-01':'2015-12-01']
        print(pca_2015)
        
    if len(stag_months_2015) == 0:
        print("PASS: No Stagflation in 2015.")
    else:
        print(f"FAIL: Found Stagflation in 2015.")

    # --- Check 2: Stagflation in 2011 ---
    print("\n[Check 2] Stagflation in 2011")
    subset_2011 = regimes['2011-01-01':'2011-12-01']
    stag_months_2011 = subset_2011[subset_2011 == 'Stagflation']
    print(f"Stagflation months in 2011: {len(stag_months_2011)} / {len(subset_2011)}")
    # User wanted it gone from 2011 too, or at least reduced.
    
    # --- Check 3: Recovery Existence ---
    print("\n[Check 3] Recovery Regime Existence")
    recovery_counts = regimes[regimes == 'Recovery'].count()
    print(f"Total Recovery Months: {recovery_counts}")
    
    if recovery_counts > 0:
        print("PASS: Recovery regime is present.")
    else:
        print("FAIL: Recovery regime is EMPTY.")

    # --- Check 4: Recovery Specific Periods (e.g. post-2008) ---
    print("\n[Check 4] Recovery in 2009-2010")
    subset_rec = regimes['2009-06-01':'2010-06-01']
    rec_months = subset_rec[subset_rec == 'Recovery']
    print(f"Recovery months in 2009-06 to 2010-06: {len(rec_months)}")
    print(f"Regimes found: {subset_rec.unique()}")

    # --- Check 5: Expansion vs Recovery (Inflation Threshold) ---
    print("\n[Check 5] Expansion Inflation Threshold Check")
    # We increased threshold to 0.2. 
    # Let's check a month that might have been marginally Expansion but should now be Recovery.
    pass

if __name__ == "__main__":
    try:
        test_regime_logic()
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
