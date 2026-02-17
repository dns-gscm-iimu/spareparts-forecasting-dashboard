import pandas as pd
import os

files = ['Dashboard_Database.csv', 'Future_Forecast_Database.csv', 'history.csv']
expected_skus = [
    'PD2976', 'PD457', 'PD1399', 'PD3978', 'PD238', 
    'PD7820', 'PD391', 'PD112', 'PD293', 'PD2782', 'PD2801'
]

print("--- DATA FILE CHECK ---")
for f in files:
    if os.path.exists(f):
        size = os.path.getsize(f)
        print(f"[OK] {f} exists ({size/1024:.2f} KB)")
        try:
            df = pd.read_csv(f)
            print(f"   Rows: {len(df)}, Columns: {list(df.columns)}")
            
            if 'Part' in df.columns:
                unique_parts = df['Part'].unique()
                print(f"   Unique Parts ({len(unique_parts)}): {unique_parts}")
                
                missing = [sku for sku in expected_skus if sku not in unique_parts]
                if missing:
                    print(f"   [WARNING] Missing SKUs: {missing}")
                else:
                    print("   [OK] All expected SKUs present.")
            
            if 'Location' in df.columns:
                 print(f"   Locations: {df['Location'].unique()}")
                 
        except Exception as e:
            print(f"   [ERROR] Could not read file: {e}")
    else:
        print(f"[MISSING] {f} not found!")

