import pandas as pd
import numpy as np

def analyze():
    try:
        df = pd.read_csv('Dashboard_Database.csv')
    except:
        print("No DB found")
        return

    # Filter out Actuals
    df = df[df['Model'] != 'Actual']
    
    # We want to see how XGBoost compares to others
    # Group by Part, Location, Model -> Average MAPE (across splits)
    summary = df.groupby(['Part', 'Location', 'Model'])['MAPE'].mean().reset_index()
    
    print("--- Model Comparison (Average Test MAPE %) ---")
    best_models = []
    
    parts = summary[['Part', 'Location']].drop_duplicates().values
    
    xgboost_wins = 0
    total_cases = 0
    
    for part, loc in parts:
        sub = summary[(summary['Part'] == part) & (summary['Location'] == loc)]
        # Sort by MAPE
        sub = sub.sort_values('MAPE')
        winner = sub.iloc[0]
        
        xgb_row = sub[sub['Model'] == 'XGBoost']
        xgb_mape = xgb_row['MAPE'].values[0] if not xgb_row.empty else 999
        
        best_mape = winner['MAPE']
        best_model = winner['Model']
        
        is_xgb_winner = (best_model == 'XGBoost')
        if is_xgb_winner: xgboost_wins += 1
        total_cases += 1
        
        print(f"\n{part} - {loc}:")
        print(f"  Winner: {best_model} ({best_mape:.1f}%)")
        print(f"  XGBoost: {xgb_mape:.1f}%")
        
        # Check rank of XGBoost
        rank = sub['Model'].tolist().index('XGBoost') + 1
        print(f"  XGBoost Rank: {rank}/{len(sub)}")

    print(f"\n\n--- Summary ---")
    print(f"XGBoost is the BEST model in {xgboost_wins}/{total_cases} cases.")
    
if __name__ == "__main__":
    analyze()
