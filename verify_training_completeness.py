import pandas as pd
import os

DB_FILE = 'Dashboard_Database.csv'

TARGET_PARTS = [
    'PD7820', 'PD391', 'PD112', 'PD293', # Medium
    'PD2782', 'PD2801' # Low
]

EXPECTED_SPLITS = [
    'Split 3y/1y',
    'Split 3.5y/0.5y',
    'Split 3.2y/0.8y'
]

EXPECTED_MODELS = [
    'ETS', 'SARIMA', 'Prophet', 'XGBoost', 'N-HiTS', 'Weighted Ensemble'
]

def verify():
    if not os.path.exists(DB_FILE):
        print("Database not found.")
        return

    df = pd.read_csv(DB_FILE)
    
    # Filter for target parts
    df = df[df['Part'].isin(TARGET_PARTS)]
    
    missing_log = []
    
    # Check each combination
    for part in TARGET_PARTS:
        # Check if part exists at all
        part_df = df[df['Part'] == part]
        if part_df.empty:
            missing_log.append(f"MISSING PART: {part} (No data found)")
            continue

        # Check locations (A and B usually, but let's see what's in data)
        locations = part_df['Location'].unique()
        
        for loc in locations:
            for split in EXPECTED_SPLITS:
                for model in EXPECTED_MODELS:
                    exists = not part_df[
                        (part_df['Location'] == loc) & 
                        (part_df['Split'] == split) & 
                        (part_df['Model'] == model)
                    ].empty
                    
                    if not exists:
                        missing_log.append(f"MISSING: {part} ({loc}) - {split} - {model}")

    if not missing_log:
        print("SUCCESS: All Medium and Low SKUs have been trained on all 3 Splits and all 6 Models.")
        # Print counts to be sure
        print("\nVerification Counts:")
        summary = df.groupby(['Part', 'Location', 'Split'])['Model'].nunique()
        print(summary)
    else:
        print("NON-COMPLIANCE FOUND:")
        for log in missing_log[:20]: # Limit output
            print(log)
        if len(missing_log) > 20:
            print(f"... and {len(missing_log)-20} more.")

if __name__ == "__main__":
    verify()
