import pandas as pd

def debug():
    data_path = '2025-11-MD.csv'
    print(f"Loading {data_path}...")
    
    # Read first few rows as string to inspect headers and transform row
    raw_head = pd.read_csv(data_path, nrows=5)
    print("First 5 rows (raw):")
    print(raw_head[['sasdate', 'UMCSENTx']])
    
    # Simulate loading logic
    df = pd.read_csv(data_path)
    # Row 0 is Transform codes
    df = df.iloc[1:].reset_index(drop=True)
    df['sasdate'] = pd.to_datetime(df['sasdate'])
    df.set_index('sasdate', inplace=True)
    
    target_date = pd.Timestamp('1960-01-01')
    
    if 'UMCSENTx' in df.columns:
        val = df.loc[target_date, 'UMCSENTx']
        print(f"\nValue at {target_date} for UMCSENTx: {val}")
        print(f"Is Na/NaN? {pd.isna(val)}")
        
        # Check surrounding
        print("\nSurrounding values:")
        print(df.loc['1959-10-01':'1960-04-01', 'UMCSENTx'])
    else:
        print("UMCSENTx not found in columns!")

if __name__ == "__main__":
    debug()
