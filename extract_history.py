import pandas as pd
import os

INPUT_FILE = 'Spare-Part-Data-With-Summary.xlsx'
OUTPUT_FILE = 'history.csv'

def extract_history():
    print("Loading Excel (All Sheets)...")
    try:
        # Read all sheets
        dfs = pd.read_excel(INPUT_FILE, sheet_name=None)
        
        all_data = []
        for sheet_name, df_sheet in dfs.items():
            print(f"Processing sheet: {sheet_name}")
            # Check if columns exist
            cols = ['Part ID', 'Location', 'Month', 'Demand']
            if all(c in df_sheet.columns for c in cols):
                sub = df_sheet[cols].copy()
                all_data.append(sub)
            else:
                 print(f"Skipping {sheet_name} (Missing columns)")
        
        if not all_data:
            print("No valid data found.")
            return

        df = pd.concat(all_data, ignore_index=True)
        
        # Rename
        df.columns = ['Part', 'Location', 'Date', 'Value']
        
        # Formats
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Metadata
        df['Model'] = 'Actual'
        # We don't need Split for pure history visualization, but dashboard might expect it if we merge.
        # But we won't merge, we'll load separately.
        
        print(f"Extracted {len(df)} rows.")
        print(df.head())
        
        df.to_csv(OUTPUT_FILE, index=False)
        print(f"Saved to {OUTPUT_FILE}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    extract_history()
