
import pandas as pd
import numpy as np
import warnings
from darts import TimeSeries
from darts.models import NBEATSModel, RNNModel
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_percentage_error
import logging

# Suppress warnings
warnings.filterwarnings('ignore')
logging.getLogger("darts").setLevel(logging.WARNING)
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

INPUT_FILE = 'Spare-Part-Data-With-Summary.xlsx'
TARGET_PARTS = ['PD457', 'PD2976', 'PD1399', 'PD3978', 'PD238']

# --- helper ---
def get_mape(y_true, y_pred):
    y_true_clean = np.where(y_true == 0, 1e-6, y_true)
    return np.mean(np.abs((y_true - y_pred) / y_true_clean))

def run_sarima(train_series, val_len):
    train_df = train_series.to_dataframe()[train_series.components[0]] # Get value column
    try:
        model = SARIMAX(
            train_df,
            order=(1, 1, 1),
            seasonal_order=(1, 1, 0, 12),
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        res = model.fit(disp=False)
        forecast = res.get_forecast(steps=val_len)
        return forecast.predicted_mean.values
    except:
        return np.zeros(val_len)

def run_nbeats(train_series, val_len):
    input_chunk = min(12, len(train_series)//2)
    model = NBEATSModel(
        input_chunk_length=input_chunk,
        output_chunk_length=val_len, 
        n_epochs=40,
        random_state=42,
        pl_trainer_kwargs={"accelerator": "cpu", "enable_progress_bar": False}
    )
    try:
        model.fit(train_series, verbose=False)
        pred = model.predict(val_len)
        return pred.values().flatten()
    except:
        return np.zeros(val_len)

def run_lstm(train_series, val_len):
    input_chunk = 12
    model = RNNModel(
        model='LSTM',
        hidden_dim=20,
        n_rnn_layers=1,
        input_chunk_length=input_chunk,
        n_epochs=40,
        random_state=42,
        pl_trainer_kwargs={"accelerator": "cpu", "enable_progress_bar": False}
    )
    try:
        model.fit(train_series, verbose=False)
        pred = model.predict(val_len)
        return pred.values().flatten()
    except:
         return np.zeros(val_len)

def main():
    print(f"--- Multi-Part Analysis for: {TARGET_PARTS} ---")
    
    # Load ALL Sheets to ensure coverage
    try:
        xls = pd.ExcelFile(INPUT_FILE)
        df_list = []
        # Iterate all sheets
        for sheet in xls.sheet_names:
            try:
                d = pd.read_excel(INPUT_FILE, sheet_name=sheet, usecols=['Part ID', 'Location', 'Month', 'Demand'])
                df_list.append(d)
            except:
                pass # Skip sheets that don't match structure
        df = pd.concat(df_list)
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    df['Month'] = pd.to_datetime(df['Month'])
    df['Demand'] = pd.to_numeric(df['Demand'], errors='coerce').fillna(0)
    
    # Splits definition
    splits = [
        {'name': 'Split A (3y Train / 1y Test)',   'train_end': '2023-12-01'},
        {'name': 'Split B (3.5y Train / 0.5y Test)', 'train_end': '2024-06-01'}
    ]
    
    summary = []

    for part in TARGET_PARTS:
        part_df = df[df['Part ID'] == part]
        
        if part_df.empty:
            print(f"Warning: {part} not found in file.")
            continue
            
        locs = part_df['Location'].unique()
        
        for loc in locs:
            print(f"\n> Processing {part} - {loc}...")
            
            loc_df = part_df[part_df['Location'] == loc].set_index('Month').sort_index()
            loc_df = loc_df.resample('MS')['Demand'].sum()
            
            # Skip if too short
            if len(loc_df) < 12: 
                print("Skipping: Not enough data.")
                continue
            
            series = TimeSeries.from_dataframe(loc_df.to_frame(), value_cols='Demand')

            for split in splits:
                train_end_dt = pd.Timestamp(split['train_end'])
                
                # Verify split point exists
                if train_end_dt > series.end_time() or train_end_dt < series.start_time():
                    continue

                train_series, val_series = series.split_after(train_end_dt)
                val_len = len(val_series)
                
                # Check for "Test 1 Year" - might not have full year if data ends early
                if split['name'].startswith('Split A') and val_len < 12:
                     # Just proceed with what we have, but note it
                     pass
                if val_len == 0: continue

                y_true = val_series.values().flatten()
                
                # Run Models
                # 1. SARIMA
                pred_s = run_sarima(train_series, val_len)
                mape_s = get_mape(y_true, pred_s)
                
                # 2. N-BEATS
                pred_n = run_nbeats(train_series, val_len)
                mape_n = get_mape(y_true, pred_n)
                
                # 3. LSTM
                pred_l = run_lstm(train_series, val_len)
                mape_l = get_mape(y_true, pred_l)
                
                # Winner
                results = {'SARIMA': mape_s, 'N-BEATS': mape_n, 'LSTM': mape_l}
                # Filter out crazy MAPEs (> 500%)
                valid_results = {k: v for k, v in results.items() if v < 5.0} # 500% cap for sanity
                
                if not valid_results:
                    best_model = "None (All Failed)"
                    best_mape = 9.99
                else:
                    best_model = min(valid_results, key=valid_results.get)
                    best_mape = valid_results[best_model]
                
                summary.append({
                    'Part': part,
                    'Location': loc,
                    'Split': split['name'],
                    'Best Model': best_model,
                    'MAPE': best_mape,
                    'SARIMA': mape_s,
                    'N-BEATS': mape_n,
                    'LSTM': mape_l
                })
                
                print(f"  {split['name'][:10]}... Best: {best_model} ({best_mape:.2%})")

    # Result Table
    print("\n" + "="*80)
    print(f"{'Part':<8} | {'Loc':<4} | {'Split':<30} | {'Best Model':<10} | {'MAPE':<8}")
    print("-" * 80)
    
    summary_df = pd.DataFrame(summary)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(by=['Part', 'Location', 'MAPE'])
        
        for _, row in summary_df.iterrows():
            print(f"{row['Part']:<8} | {row['Location']:<4} | {row['Split']:<30} | {row['Best Model']:<10} | {row['MAPE']:.2%}")
            
    print("="*80)

if __name__ == "__main__":
    main()
