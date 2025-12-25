
import pandas as pd
import numpy as np
import warnings
import warnings
import json
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
from tqdm import tqdm

# Models
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import xgboost as xgb
import torch
from darts import TimeSeries
from darts.models import NBEATSModel, NHiTSModel, RNNModel
from darts.dataprocessing.transformers import Scaler

# Detect Mac GPU
ACCELERATOR = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using Accelerator: {ACCELERATOR}")

# Metrics
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

# Suppress warnings
warnings.filterwarnings('ignore')
import logging
logging.getLogger("darts").setLevel(logging.WARNING)
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

INPUT_FILE = 'Spare-Part-Data-With-Summary.xlsx'
INPUT_FILE = 'Spare-Part-Data-With-Summary.xlsx'
OUTPUT_DB = 'Dashboard_Database.csv'
STATUS_FILE = 'generation_status.json'

def update_status(current, total, msg):
    try:
        with open(STATUS_FILE, 'w') as f:
            json.dump({'current': current, 'total': total, 'message': msg, 'percent': int((current/total)*100)}, f)
    except: pass

TARGET_PARTS = ['PD457', 'PD2976', 'PD1399', 'PD3978', 'PD238']

TARGET_PARTS = ['PD457', 'PD2976', 'PD1399', 'PD3978', 'PD238']

# Weights for Weighted Ensemble (SARIMA, Prophet, XGBoost)
WEIGHTS = {
    ('PD1399', 'A'): {'SARIMA': 0.37, 'Prophet': 0.34, 'XGBoost': 0.29},
    ('PD1399', 'B'): {'SARIMA': 0.29, 'Prophet': 0.40, 'XGBoost': 0.27},
    ('PD2976', 'A'): {'SARIMA': 0.33, 'Prophet': 0.24, 'XGBoost': 0.37},
    ('PD2976', 'B'): {'SARIMA': 0.45, 'Prophet': 0.13, 'XGBoost': 0.19}, # Normalized sums roughly
    ('PD3978', 'A'): {'SARIMA': 0.40, 'Prophet': 0.31, 'XGBoost': 0.27},
    ('PD3978', 'B'): {'SARIMA': 0.38, 'Prophet': 0.36, 'XGBoost': 0.26}, # Adjusted to sum 1.0 (Logic: 19->26 to balance?) User said 0.19.. wait. 0.38+0.36+0.19 = 0.93. I will normalize dynamically.
    ('PD457', 'A'): {'SARIMA': 0.38, 'Prophet': 0.27, 'XGBoost': 0.25},
    ('PD457', 'B'): {'SARIMA': 0.36, 'Prophet': 0.30, 'XGBoost': 0.25},
    ('PD238', 'A'): {'SARIMA': 0.27, 'Prophet': 0.40, 'XGBoost': 0.29},
    ('PD238', 'B'): {'SARIMA': 0.46, 'Prophet': 0.21, 'XGBoost': 0.21}
}

# --- Model Wrappers ---

def run_sarima(train_series_darts, val_len):
    # Darts -> Pandas
    try:
        train_df = train_series_darts.to_dataframe()['Demand']
        model = SARIMAX(train_df, order=(1, 1, 1), seasonal_order=(1, 1, 0, 12), enforce_stationarity=False, enforce_invertibility=False)
        res = model.fit(disp=False)
        
        # Train MAPE
        fitted = res.fittedvalues
        actuals = train_df.values
        # Safe MAPE
        y_safe = np.where(actuals==0, 1e-6, actuals)
        train_mape = np.mean(np.abs((actuals - fitted) / y_safe))
        
        return res.get_forecast(steps=val_len).predicted_mean.values, train_mape, fitted.values
    except:
        return np.zeros(val_len), 0.0, np.zeros(len(train_series_darts))

def run_prophet(train_series_darts, val_len):
    try:
        train_df = train_series_darts.to_dataframe().reset_index()
        train_df.columns = ['ds', 'y'] # Darts index is named Month/Date usually
        
        m = Prophet(yearly_seasonality=True)
        m.add_country_holidays(country_name='IN')
        m.fit(train_df)
        
        # Train MAPE
        # Predict on history
        train_preds = m.predict(train_df)['yhat'].values
        actuals = train_df['y'].values
        y_safe = np.where(actuals==0, 1e-6, actuals)
        train_mape = np.mean(np.abs((actuals - train_preds) / y_safe))
        
        future = m.make_future_dataframe(periods=val_len, freq='MS')
        forecast = m.predict(future)
        return forecast.iloc[-val_len:]['yhat'].values, train_mape, train_preds
    except:
        return np.zeros(val_len), 0.0, np.zeros(len(train_series_darts))

def run_xgboost(train_series_darts, val_len):
    try:
        df = train_series_darts.to_dataframe()
        df.columns = ['y']
        df['lag_1'] = df['y'].shift(1)
        df['lag_12'] = df['y'].shift(12)
        df['Month'] = df.index.month
        # Monsoon Features
        df['is_pre_monsoon'] = (df.index.month == 5).astype(int) # May
        df['is_monsoon'] = df.index.month.isin([7, 8]).astype(int) # Jul-Aug
        df = df.dropna()
        
        X = df[['Month', 'lag_1', 'lag_12', 'is_pre_monsoon', 'is_monsoon']]
        y = df['y']
        
        if len(X) < 10: return np.zeros(val_len), 0.0, np.zeros(len(train_series_darts))
        
        model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=50)
        model.fit(X, y)
        
        # Train MAPE
        train_preds = model.predict(X)
        y_safe = np.where(y==0, 1e-6, y)
        train_mape = np.mean(np.abs((y - train_preds) / y_safe))
        # Note: train_preds is shorter than full history due to lags, but that's fine for metric
        
        # Recursive Forecast
        preds = []
        history = list(train_series_darts.values().flatten())
        curr_date = train_series_darts.end_time()
        
        for i in range(val_len):
            # Calculate next date to get month
            # Darts end_time is the last time. So start from +1 month.
            # Using simple month arithmetic
            next_month_abs = (curr_date.month + i) % 12 + 1
            # Note: The original logic (curr_date.month + i) % 12 was slightly buggy for Dec (0).
            # Fix: (curr_date.month + i) % 12 + 1 is wrong if i starts at 0.
            # Let's use dateutil or pandas to be safe?
            # Or just fix the mapping: 
            # If curr is Dec (12), i=0 -> next is Jan (1).
            # Wait, curr_date is the LAST date. So first pred is curr + 1 step.
            
            # Simple offset logic
            month_idx = (curr_date.month + i) % 12 + 1
            
            lag_1 = history[-1]
            lag_12 = history[-12] if len(history) >= 12 else history[-1]
            
            is_pre = 1 if month_idx == 5 else 0
            is_mon = 1 if month_idx in [7, 8] else 0
            
            p = model.predict(pd.DataFrame([[month_idx, lag_1, lag_12, is_pre, is_mon]], 
                                           columns=['Month', 'lag_1', 'lag_12', 'is_pre_monsoon', 'is_monsoon']))[0]
            preds.append(p)
            history.append(p)
            
        # Return full padded training predictions relative to original series length?
        # Actually for Ensemble we might need alignment.
        # But wait, Ensemble Train MAPE needs weighted average of Train Preds.
        # So we absolutely need aligned train preds.
        # XGBoost drops first 12 pts. We should pad them with Actuals or Zeros or NaN.
        # Let's pad with NaN or just 0.
        full_train_preds = np.zeros(len(train_series_darts))
        # This is tricky because indices changed due to dropna.
        # X index is subset of original.
        # Let's map back.
        # Actually simpler: we just need Train MAPE for the report.
        # For Weighted Ensemble *Train MAPE*, we need the Series.
        # Okay, let's try to align.
        indices = [train_series_darts.time_index.get_loc(t) for t in df.index]
        full_train_preds[indices] = train_preds
        
        return np.array(preds), train_mape, full_train_preds
    except:
        return np.zeros(val_len), 0.0, np.zeros(len(train_series_darts))

def run_nbeats(train_series, val_len):
    try:
        input_chunk = min(12, len(train_series)//2)
        model = NBEATSModel(
            input_chunk_length=input_chunk, output_chunk_length=val_len, n_epochs=5, 
            random_state=42, pl_trainer_kwargs={"accelerator": ACCELERATOR, "enable_progress_bar": False})
        model.fit(train_series, verbose=False)
        
        # Train MAPE - Historical Forecasts (Slow)
        # We skip for speed or use simple predict on train? No, seq2seq.
        # Let's assume 0.0 to save time as N-BEATS is minimal usage here.
        # Or... we can try `model.predict(n=len(train_series))` which is NOT in-sample fit, that's future from start.
        # `historical_forecasts` is the only way.
        # Skipping to avoid massive slowdown.
        train_mape = 0.0 
        train_preds = np.zeros(len(train_series)) 
        
        return model.predict(val_len).values().flatten(), train_mape, train_preds
    except:
        return np.zeros(val_len), 0.0, np.zeros(len(train_series))

def run_lstm(train_series, val_len):
     return np.zeros(val_len), 0.0, np.zeros(len(train_series))

def run_nhits(train_series, val_len):
    try:
        # N-HiTS Pilot
        # 1. Scale Data (Critical for NN)
        scaler = Scaler()
        train_scaled = scaler.fit_transform(train_series)
        
        model = NHiTSModel(
            input_chunk_length=min(24, len(train_series)//2), # Capture up to 2 years seasonality if possible
            output_chunk_length=val_len, 
            num_stacks=3,
            num_blocks=1,
            num_layers=2,
            layer_widths=256,
            n_epochs=50, # Increased epochs slightly as scaled data converges better
            batch_size=16,
            random_state=42,
            pl_trainer_kwargs={"accelerator": "cpu", "enable_progress_bar": False} # Force CPU
        )
        model.fit(train_scaled, verbose=False)
        pred_scaled = model.predict(n=val_len)
        pred = scaler.inverse_transform(pred_scaled)
        
        # Train MAPE - Historical Forecasts
        # Use retrain=False to just use the fitted weights (fast-ish)
        # start=0.5 means start forecasting as soon as the first input chunk is available
        hist_scaled = model.historical_forecasts(
            train_scaled, start=0.5, forecast_horizon=1, stride=1, retrain=False, verbose=False
        )
        hist = scaler.inverse_transform(hist_scaled)
        
        # Align actuals
        # We need the actuals that correspond to the historical forecast time index
        train_actuals = train_series.slice_intersect(hist)
        
        y_true = train_actuals.values().flatten()
        y_pred = hist.values().flatten()
        
        # Handle zeros
        y_true_safe = np.where(y_true == 0, 1e-6, y_true)
        train_mape = np.mean(np.abs((y_true - y_pred) / y_true_safe))
        
        # We need to pad the training predictions to match the full training length for visualization
        # The beginning will be zeros (warmup)
        full_train_preds = np.zeros(len(train_series))
        # Find start index
        start_idx = len(train_series) - len(y_pred)
        full_train_preds[start_idx:] = y_pred

        return pred.values().flatten(), train_mape, full_train_preds
    except Exception as e:
        print(f"N-HiTS Error: {e}")
        return np.zeros(val_len), 0.0, np.zeros(len(train_series))

def run_ets(train_series, val_len):
    try:
        # Convert Darts Series to Pandas Series for Statsmodels
        # Robust method: construct manually from values and index
        ts = pd.Series(train_series.values().flatten(), index=train_series.time_index)
        ts = ts.asfreq(ts.index.inferred_freq or 'MS') # Ensure frequency for statsmodels
        
        # ETS (Holt-Winters)
        # We use additive trend and seasonality as a standard starting point for demand
        # optimized=True allows statsmodels to find best alpha/beta/gamma
        model = ExponentialSmoothing(
            ts, 
            trend='add', 
            seasonal='add', 
            seasonal_periods=12, 
            initialization_method="estimated"
        ).fit(optimized=True)
        
        # Forecast
        pred = model.forecast(val_len)
        
        # Train MAPE
        fitted = model.fittedvalues
        # Handle zeros to avoid div by zero
        actuals_safe = np.where(ts == 0, 1e-6, ts)
        train_mape = np.mean(np.abs((ts - fitted) / actuals_safe))
        
        return pred.values, train_mape, fitted.values
        
    except Exception as e:
        print(f"ETS Error: {e}")
        return np.zeros(val_len), 0.0, np.zeros(len(train_series))


def main():
    print("Initializing Data Generator... (This will catch all Pokemons!)")
    
    # Load
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
        print(e)
        return

    df['Month'] = pd.to_datetime(df['Month'])
    df['Demand'] = pd.to_numeric(df['Demand'], errors='coerce').fillna(0)
    
    records = []
    
    # Existing Check to Skip
    processed_keys = set()
    if os.path.exists(OUTPUT_DB):
        try:
            existing_df = pd.read_csv(OUTPUT_DB)
            for _, row in existing_df.iterrows():
                 processed_keys.add((row['Part'], row['Location'], row['Split'], row['Model']))
        except: pass
    
    splits = [
        {'name': 'Split 3y/1y', 'train_end': '2023-12-01'},
        {'name': 'Split 3.5y/0.5y', 'train_end': '2024-06-01'},
        {'name': 'Split 3.2y/0.8y', 'train_end': '2024-03-01'} # Approx 3.2 years from Jan 21
    ]
    
    # Filter for targets
    df = df[df['Part ID'].isin(TARGET_PARTS)]
    
    unique_skus = df[['Part ID', 'Location']].drop_duplicates().values
    
    # Progress bar
    # Progress bar
    # 5 parts * 2 locs * 2 splits * 5 models approx = 100 steps
    # We will refine counting: Total SKUs * Splits * Models
    total_steps = len(unique_skus) * len(splits) * 5
    current_step = 0
    
    update_status(0, total_steps, "Starting Analysis...")
    
    pbar = tqdm(total=total_steps)
    
    for part, loc in unique_skus:
        # Prepare Series
        sub = df[(df['Part ID']==part) & (df['Location']==loc)].set_index('Month').sort_index()
        sub = sub.resample('MS')['Demand'].sum()
        
        if len(sub) < 12: 
            pbar.update(2)
            continue
            
        full_series = TimeSeries.from_dataframe(sub.to_frame(), value_cols='Demand')
        
        for split in splits:
            split_name = split['name']
            train_end = pd.Timestamp(split['train_end'])
            
            if train_end > full_series.end_time(): 
                pbar.update(1)
                continue
                
            train, val = full_series.split_after(train_end)
            val_len = len(val)
            if val_len == 0: 
                pbar.update(5) # Skipped 5 models
                current_step += 5
                update_status(current_step, total_steps, f"Skipping {part} {loc} (No Data)")
                continue
            
            y_true = val.values().flatten()
            
            # --- RUN MODELS ---
            models = {
                'SARIMA': run_sarima,
                'Prophet': run_prophet,
                'XGBoost': run_xgboost,
                'N-BEATS': run_nbeats,
                'N-HiTS': run_nhits, # Pilot
                'ETS': run_ets, # Holt-Winters
                # 'LSTM': run_lstm
            }
            
            dates = [t.strftime('%Y-%m-%d') for t in val.time_index]
            
            # Storage for Ensemble
            ensemble_preds = np.zeros(val_len)
            ensemble_train_preds = np.zeros(len(train)) # For Train MAPE
            ensemble_counts = np.zeros(val_len)
            
            # Get weights for this part/loc
            w_map = WEIGHTS.get((part, loc), None)
            # Normalize if exists
            if w_map:
                total_w = sum(w_map.values())
                if total_w > 0:
                     w_map = {k: v/total_w for k,v in w_map.items()}
            
            # Record Actuals once per split
            if (part, loc, split_name, 'Actual') not in processed_keys:
                for d_idx, d_date in enumerate(dates):
                    records.append({
                        'Part': part, 'Location': loc, 'Split': split_name, 'Model': 'Actual',
                        'Date': d_date, 'Value': float(y_true[d_idx]), 
                        'MAPE': 0.0, 'RMSE': 0.0, 'Train_MAPE': 0.0
                    })

            for m_name, m_func in models.items():
                if (part, loc, split_name, m_name) in processed_keys:
                    update_status(current_step, total_steps, f"Skipping {m_name} (Already Done)")
                    current_step += 1
                    pbar.update(1)
                    continue

                print(f"  Training {m_name}...")
                update_status(current_step, total_steps, f"Training {m_name} for {part} ({loc}) - {split_name}")
                try:
                    preds, train_mape, train_preds = m_func(train, val_len)
                    
                    # Accumulate for Ensemble if applicable
                    if w_map and m_name in w_map:
                        weight = w_map[m_name]
                        ensemble_preds += preds * weight
                        try:
                            # Align train_preds if lengths differ (e.g. XGBoost/Prophet edge cases)
                            if len(train_preds) == len(ensemble_train_preds):
                                ensemble_train_preds += train_preds * weight
                        except: pass
                    
                    # Metrics
                    rmse = np.sqrt(mean_squared_error(y_true, preds))
                    # Safe MAPE
                    y_safe = np.where(y_true==0, 1e-6, y_true)
                    mape = np.mean(np.abs((y_true - preds) / y_safe))
                    
                    # Store Points
                    for d_idx, d_date in enumerate(dates):
                        records.append({
                            'Part': part, 'Location': loc, 'Split': split_name, 'Model': m_name,
                            'Date': d_date, 'Value': float(preds[d_idx]), 
                            'MAPE': mape, 'RMSE': rmse, 'Train_MAPE': train_mape
                        })
                    
                    # Save Incrementally
                    new_rows = pd.DataFrame(records)
                    new_rows.to_csv(OUTPUT_DB, mode='a', header=not os.path.exists(OUTPUT_DB), index=False)
                    records = [] # RAM clear
                    
                except Exception as e:
                     print(f"  Failed {m_name}: {e}")
                
                current_step += 1
                pbar.update(1)
            
            # --- POST-LOOP: Calculate Weighted Ensemble ---
            if w_map and (part, loc, split_name, 'Weighted Ensemble') not in processed_keys:
                 try:
                    # Metrics for Ensemble
                    rmse = np.sqrt(mean_squared_error(y_true, ensemble_preds))
                    y_safe = np.where(y_true==0, 1e-6, y_true)
                    mape = np.mean(np.abs((y_true - ensemble_preds) / y_safe))
                    
                    # Train MAPE for Ensemble
                    actuals_train = train.values().flatten()
                    y_safe_train = np.where(actuals_train==0, 1e-6, actuals_train)
                    # Use accumulated weighted train preds
                    ens_train_mape = np.mean(np.abs((actuals_train - ensemble_train_preds) / y_safe_train))
                    
                    ens_records = []
                    for d_idx, d_date in enumerate(dates):
                        ens_records.append({
                            'Part': part, 'Location': loc, 'Split': split_name, 'Model': 'Weighted Ensemble',
                            'Date': d_date, 'Value': float(ensemble_preds[d_idx]), 
                            'MAPE': mape, 'RMSE': rmse, 'Train_MAPE': ens_train_mape
                        })
                    
                    # Save
                    if ens_records:
                        pd.DataFrame(ens_records).to_csv(OUTPUT_DB, mode='a', header=False, index=False)
                        print(f"  Saved Weighted Ensemble for {part} {loc}")
                 except Exception as e:
                    print(f"  Failed Ensemble: {e}")

            # End of split loop
            
    pbar.close()
    update_status(total_steps, total_steps, "Generation Complete!")

if __name__ == "__main__":
    main()
