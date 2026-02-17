import pandas as pd
import os
import numpy as np

files = ['Dashboard_Database.csv', 'Future_Forecast_Database.csv', 'history.csv']

print("--- ADVANCED DATA CHECK (DTYPES & COVERAGE) ---")

def check_file(filename):
    if not os.path.exists(filename):
        print(f"[MISSING] {filename}")
        return None
        
    print(f"\nAnalyzing: {filename}")
    try:
        df = pd.read_csv(filename)
        print(f"   Dtypes:\n{df.dtypes}")
        
        # Check Value column specifically
        if 'Value' in df.columns:
            if not pd.api.types.is_numeric_dtype(df['Value']):
                print("   [CRITICAL WARNING] 'Value' column is NOT numeric!")
                print(f"   Sample non-numeric values: {df[pd.to_numeric(df['Value'], errors='coerce').isna()]['Value'].head().tolist()}")
            else:
                print("   [OK] 'Value' column is numeric.")
                
        # Check Date column
        if 'Date' in df.columns:
             try:
                 pd.to_datetime(df['Date'])
                 print("   [OK] 'Date' column is parseable.")
             except:
                 print("   [CRITICAL WARNING] 'Date' column parsing failed!")

        return df
    except Exception as e:
        print(f"   [ERROR] Reading fail: {e}")
        return None

db_df = check_file('Dashboard_Database.csv')
fut_df = check_file('Future_Forecast_Database.csv')
hist_df = check_file('history.csv')

# Check Coverage Intersection
if db_df is not None and hist_df is not None:
    # Filter garbage out of hist first similarly to how app might fail to if not careful
    # App logic: hist_df = pd.read_csv; ... no explicit filtering of garbage rows before type conversion if read_csv guessed wrong.
    
    # Get set of P/L
    db_pairs = set(zip(db_df['Part'], db_df['Location']))
    hist_pairs = set(zip(hist_df['Part'], hist_df['Location']))
    
    missing_in_hist = db_pairs - hist_pairs
    if missing_in_hist:
        print(f"\n[WARNING] {len(missing_in_hist)} Part/Location pairs from Dashboard DB are missing in History!")
        print(list(missing_in_hist)[:5])
    else:
        print("\n[OK] All Dashboard Part/Locations have History data.")
