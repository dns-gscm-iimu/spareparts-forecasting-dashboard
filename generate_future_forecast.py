
import pandas as pd
import numpy as np
import os
import warnings
from darts import TimeSeries
import logging

# Import Model Functions and Configuration from existing script
from generate_dashboard_data import (
    run_sarima, run_prophet, run_xgboost, run_nhits, run_ets,
    INPUT_FILE, WEIGHTS, TARGET_PARTS
)

warnings.filterwarnings('ignore')
logging.getLogger("darts").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

OUTPUT_DB = 'Dashboard_Database.csv'
FUTURE_DB = 'Future_Forecast_Database.csv'
FORECAST_HORIZON = 12 # Jan 2025 - Dec 2025 (If data ends Dec 2024? Need to check end date)
# Actually, data likely ends earlier. User said "upcoming year - jan25 - dec25".
# If data ends in Jun 2024, we need to forecast Jul 2024 - Dec 2025?
# Or maybe the data goes up to Dec 2024.
# I will check the max date in the data.

def get_best_models():
    """Identify the winning model for each Part/Location based on Composite Score."""
    print("Identifying Best Models...")
    if not os.path.exists(OUTPUT_DB):
        print("Error: Dashboard Database not found.")
        return {}
    
    df = pd.read_csv(OUTPUT_DB)
    df = df[df['Model'] != 'Actual']
    df = df[df['Model'] != 'N-BEATS'] # Exclude N-BEATS per user request
    
    # Calculate Score (Replicating Dashboard Logic)
    df['Score'] = 999.0
    
    # 1 row per model/split
    summary = df.drop_duplicates(subset=['Part', 'Location', 'Split', 'Model']).copy()
    
    # Normalize globally per Part/Loc (across splits)
    # We want to find the model architecture that is generally best.
    # Group by Part, Location
    best_models = {}
    
    for (part, loc), group in summary.groupby(['Part', 'Location']):
        # Min-Max Norm within this Part/Loc group
        try:
            # MAPE
            mn, mx = group['MAPE'].min(), group['MAPE'].max()
            d = mx - mn
            group['n_mape'] = (group['MAPE'] - mn) / d if d > 0 else 0
            
            # RMSE
            mn, mx = group['RMSE'].min(), group['RMSE'].max()
            d = mx - mn
            group['n_rmse'] = (group['RMSE'] - mn) / d if d > 0 else 0
            
            # Bias
            group['abs_bias'] = group['Bias'].abs()
            mn, mx = group['abs_bias'].min(), group['abs_bias'].max()
            d = mx - mn
            group['n_bias'] = (group['abs_bias'] - mn) / d if d > 0 else 0
            
            # Score
            group['Score'] = 0.7 * group['n_mape'] + 0.2 * group['n_rmse'] + 0.1 * group['n_bias']
            
            # Find Winner
            # We pick the model instance with the absolute lowest score.
            winner_row = group.loc[group['Score'].idxmin()]
            best_models[(part, loc)] = winner_row['Model']
            print(f"  {part} {loc} Winner: {winner_row['Model']} (Score: {winner_row['Score']:.3f})")
            
        except Exception as e:
            print(f"  Error scoring {part} {loc}: {e}")
            # Default to Ensemble or SARIMA if error
            best_models[(part, loc)] = 'Weighted Ensemble'
            
    return best_models

def main():
    print("Starting Future Forecast Generation (Jan 2025 - Dec 2025)...")
    
    # Load Raw Data
    print(f"Loading Raw Data from {INPUT_FILE}...")
    try:
        xls = pd.ExcelFile(INPUT_FILE)
        df_list = []
        for sheet in xls.sheet_names:
            try:
                d = pd.read_excel(INPUT_FILE, sheet_name=sheet, usecols=['Part ID', 'Location', 'Month', 'Demand'])
                df_list.append(d)
            except: pass
        df = pd.concat(df_list)
    except Exception as e:
        print(f"Error reading input file: {e}")
        return

    df['Month'] = pd.to_datetime(df['Month'])
    df['Demand'] = pd.to_numeric(df['Demand'], errors='coerce').fillna(0)
    df = df[df['Part ID'].isin(TARGET_PARTS)]
    
    records = []
    
    unique_skus = df[['Part ID', 'Location']].drop_duplicates().values
    
    # Models to run
    # Models to run
    MODELS = ['SARIMA', 'Prophet', 'XGBoost', 'N-HiTS', 'ETS', 'Weighted Ensemble']

    for part, loc in unique_skus:
        print(f"Forecasting {part} {loc}...")
        
        # Prepare Series
        sub = df[(df['Part ID']==part) & (df['Location']==loc)].set_index('Month').sort_index()
        sub = sub.resample('MS')['Demand'].sum()
        
        last_date = sub.index[-1]
        print(f"  Last Data Point: {last_date.date()}")
        
        # Calculate required steps to reach Dec 2025
        target_date = pd.Timestamp("2025-12-01")
        months_needed = (target_date.year - last_date.year) * 12 + (target_date.month - last_date.month)
        
        if months_needed <= 0:
            print("  Data already extends past 2025. No forecast needed?")
            continue
            
        print(f"  Forecasting {months_needed} steps forward...")
        
        full_series = TimeSeries.from_dataframe(sub.to_frame(), value_cols='Demand')
        
        # Helper to run specific model
        def run_single(name, series, steps):
            if name == 'SARIMA': return run_sarima(series, steps)
            if name == 'Prophet': return run_prophet(series, steps)
            if name == 'XGBoost': return run_xgboost(series, steps)
            if name == 'N-HiTS': return run_nhits(series, steps)
            if name == 'ETS': return run_ets(series, steps)
            return np.zeros(steps), 0, np.zeros(len(series))

        # Store individual preds for Ensemble calculation
        individual_preds = {}

        for model_name in MODELS:
            final_preds = None
            if model_name == 'Weighted Ensemble':
                # Calculate Ensemble from individuals
                w_map = WEIGHTS.get((part, loc), {'SARIMA':0.33, 'Prophet':0.33, 'XGBoost':0.33})
                total = sum(w_map.values())
                w_map = {k:v/total for k,v in w_map.items()}
                
                ensemble_preds = np.zeros(months_needed)
                valid_ens = True
                for m in ['SARIMA', 'Prophet', 'XGBoost']:
                    if m in individual_preds:
                        ensemble_preds += individual_preds[m] * w_map.get(m, 0)
                    else:
                        # Should not happen if loop order is preserved
                        valid_ens = False
                
                if valid_ens:
                    final_preds = ensemble_preds
                else: 
                    final_preds = np.zeros(months_needed)
            else:
                # Individual Models
                p, _, _ = run_single(model_name, full_series, months_needed)
                individual_preds[model_name] = p
                final_preds = p

            # Store Results
            # Generate Dates
            future_dates = [last_date + pd.DateOffset(months=i+1) for i in range(months_needed)]
            
            for i, val in enumerate(final_preds):
                d = future_dates[i]
                # Only keep 2025
                if d.year == 2025:
                    records.append({
                        'Part': part, 'Location': loc, 'Model': model_name,
                        'Date': d.strftime('%Y-%m-%d'), 'Value': float(max(0, val)) # No negative demand
                    })
                
    # Save
    if records:
        pd.DataFrame(records).to_csv(FUTURE_DB, index=False)
        print(f"Comparison generated and saved to {FUTURE_DB}")
    else:
        print("No records generated.")

if __name__ == "__main__":
    main()
