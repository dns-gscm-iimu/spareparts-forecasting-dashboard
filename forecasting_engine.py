
import pandas as pd
import numpy as np
import warnings
from datetime import timedelta
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet
from darts import TimeSeries
from darts.models import NBEATSModel, NHiTSModel
from darts.utils.timeseries_generation import datetime_attribute_timeseries
import logging

# Suppress Warnings
warnings.filterwarnings('ignore')
logging.getLogger("darts").setLevel(logging.WARNING)
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

REQUIRED_COLUMNS = ['Spare Part ID', 'Location', 'Month', 'Demand', 'Average Lead Time']

def validate_columns(df):
    """
    Validates that the dataframe has the required columns (Relaxed/Fuzzy Logic).
    Returns (True, None) or (False, error_message)
    """
    # Normalize columns to lower case strip
    df_cols_lower = [str(c).lower().strip() for c in df.columns]
    
    # Critical numeric columns we absolutely need
    critical_terms = ['demand', 'lead'] 
    
    found_critical = 0
    for term in critical_terms:
        if any(term in c for c in df_cols_lower):
            found_critical += 1
            
    if found_critical < 2:
        return False, "Could not identify 'Demand' or 'Average Lead Time' columns. Please ensure these numeric columns exist."
    
    return True, None

def clean_data(df):
    """
    Standardizes column names and formats using best-effort matching.
    """
    df_cols_map = {str(c).lower().strip(): c for c in df.columns}
    normalized_map = {}
    
    # Mapping heuristics
    # 1. Spare Part ID -> 'part', 'sku', 'id'
    # 2. Location -> 'location', 'city', 'site'
    # 3. Month -> 'month', 'date', 'period'
    # 4. Demand -> 'demand', 'sales', 'qty'
    # 5. Lead Time -> 'lead', 'time'
    
    heuristics = {
        'Spare Part ID': ['part', 'sku', 'id', 'item'],
        'Location': ['location', 'city', 'site', 'depot'],
        'Month': ['month', 'date', 'period', 'time'],
        'Demand': ['demand', 'sales', 'quantity', 'qty'],
        'Average Lead Time': ['lead', 'avg lead', 'lead time']
    }
    
    for target_col, keywords in heuristics.items():
        found = False
        # Try exact match first (case insensitive)
        for rc in [target_col.lower()]:
             if rc in df_cols_map:
                 normalized_map[df_cols_map[rc]] = target_col
                 found = True
                 break
        
        # Try keywords
        if not found:
            for kw in keywords:
                for actual_col in df_cols_map:
                    if kw in actual_col:
                        normalized_map[df_cols_map[actual_col]] = target_col
                        found = True
                        break
                if found: break
                
        # If not found, create dummy if it's not critical?
        # For Part/Location/Month we can default if missing
        if not found:
            if target_col == 'Spare Part ID':
                df[target_col] = 'Unknown_Part'
            elif target_col == 'Location':
                df[target_col] = 'Unknown_Loc'
            elif target_col == 'Month':
                # Create dummy sequence if missing?? Dangerous but requested "don't focus much"
                df[target_col] = pd.date_range(start='2020-01-01', periods=len(df), freq='MS')
    
    # Rename
    df = df.rename(columns=normalized_map)
    
    # Ensure columns exist (if they weren't in map and weren't created fallback)
    for rc in REQUIRED_COLUMNS:
        if rc not in df.columns:
            # Last ditch creation
            df[rc] = 0 if rc in ['Demand', 'Average Lead Time'] else 'Unknown'

    # Type Casting
    df['Month'] = pd.to_datetime(df['Month'])
    df['Demand'] = pd.to_numeric(df['Demand'], errors='coerce').fillna(0)
    df['Average Lead Time'] = pd.to_numeric(df['Average Lead Time'], errors='coerce').fillna(0)
    
    # Ensure Spare Part ID and Location are strings
    df['Spare Part ID'] = df['Spare Part ID'].astype(str)
    df['Location'] = df['Location'].astype(str)
    
    return df

# --- MODEL FUNCTIONS ---

def run_ets(train_series):
    # Statsmodels implementation
    try:
        # Check sufficient data for seasonality (typically needs 2x period)
        if len(train_series) < 24:
            # Force non-seasonal
            model = ExponentialSmoothing(train_series, trend='add', seasonal=None).fit()
        else:
            model = ExponentialSmoothing(
                train_series,
                seasonal_periods=12,
                trend='add',
                seasonal='add',
                damped_trend=True
            ).fit()
        return model
    except:
        try:
            # Fallback
            model = ExponentialSmoothing(train_series, trend='add').fit()
            return model
        except:
            return None

def run_sarima(train_series):
    try:
        # Check sufficient data for seasonality
        if len(train_series) < 24:
            # Force non-seasonal
            order = (1, 1, 1)
            seasonal_order = (0, 0, 0, 0)
        else:
            order = (1, 1, 1)
            seasonal_order = (1, 0, 0, 12)
            
        model = SARIMAX(
            train_series,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        ).fit(disp=False)
        return model
    except:
        return None

def run_prophet(train_df):
    try:
        # Prepare for Prophet: ds, y
        df = train_df.reset_index()
        df.columns = ['ds', 'y']
        m = Prophet(daily_seasonality=False, weekly_seasonality=False, yearly_seasonality=True)
        m.add_country_holidays(country_name='IN')
        m.fit(df)
        return m
    except:
        return None

def run_nhits(train_ts):
    try:
        model = NHiTSModel(
            input_chunk_length=12,
            output_chunk_length=6,
            num_stacks=3,
            num_blocks=1,
            num_layers=2,
            layer_widths=512,
            n_epochs=30, # Reduced for speed
            random_state=42,
            pl_trainer_kwargs={"accelerator": "cpu", "enable_progress_bar": False}
        )
        model.fit(train_ts, verbose=False)
        return model
    except:
        return None

# --- ENGINE ---

def analyze_part_location(df, part, location, progress_callback=None):
    """
    Main engine to analyze a specific Part/Location combo.
    df: Full cleaned dataframe
    """
    
    # Filter Data
    sub = df[(df['Spare Part ID'] == part) & (df['Location'] == location)].sort_values('Month').set_index('Month')
    sales = sub['Demand']
    
    if len(sales) < 12:
        return {'error': 'Insufficient data (<12 months) for analysis.'}
        
    # --- TRAINING ---
    # We will pick the Best Model based on the LAST split (Simulation of now) behavior
    # Dynamic Split logic for short data
    total_len = len(sales)
    
    if total_len < 18:
        # Short history (12-17 months): Use smaller test set to preserve training data
        test_len = 3 
    else:
        # Standard history (18+ months)
        test_len = 6
        
    train = sales.iloc[:-test_len]
    test = sales.iloc[-test_len:]
    
    if len(train) < 9:
        return {'error': 'Training set too small (< 9 months).'}
    
    # Results dictionary now stores detailed metrics
    results = {} # Key: ModelName, Value: {RMSE, MAPE, Bias}
    preds = {}
    
    # Helper to calc metrics
    def calc_metrics(y_true, y_pred):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        # Safe calc for MAPE
        y_true_safe = y_true.replace(0, 0.001)
        mape = np.mean(np.abs((y_true - y_pred) / y_true_safe))
        bias = np.mean(y_pred - y_true.values) # Bias = Mean(Forecast - Actual)
        return {'RMSE': rmse, 'MAPE': mape, 'Bias': bias, 'AbsBias': abs(bias)}

    # 1. ETS
    if progress_callback: progress_callback(10, "Training ETS...")
    ets_model = run_ets(train)
    if ets_model:
        pred_ets = ets_model.forecast(test_len)
        preds['ETS'] = pred_ets.values
        results['ETS'] = calc_metrics(test, pred_ets.values)
    
    # 2. SARIMA
    if progress_callback: progress_callback(30, "Training SARIMA...")
    sarima_model = run_sarima(train)
    if sarima_model:
        pred_sarima = sarima_model.get_forecast(steps=test_len).predicted_mean
        preds['SARIMA'] = pred_sarima.values
        results['SARIMA'] = calc_metrics(test, pred_sarima.values)
        
    # 3. Prophet
    if progress_callback: progress_callback(50, "Training Prophet...")
    pro_model = run_prophet(train)
    if pro_model:
        future = pro_model.make_future_dataframe(periods=test_len, freq='MS')
        forecast = pro_model.predict(future)
        pred_pro = forecast.iloc[-test_len:]['yhat'].values
        preds['Prophet'] = pred_pro
        results['Prophet'] = calc_metrics(test, pred_pro)
        
    # 4. N-HiTS (Darts)
    if progress_callback: progress_callback(70, "Training N-HiTS (Deep Learning)...")
    try:
        ts_train = TimeSeries.from_series(train)
        nhits_model = run_nhits(ts_train)
        if nhits_model:
            pred_ts = nhits_model.predict(test_len)
            pred_nhits = pred_ts.values().flatten()
            preds['N-HiTS'] = pred_nhits
            results['N-HiTS'] = calc_metrics(test, pred_nhits)
    except:
        pass
        
    # 5. Ensemble (Average of valid predictions)
    if preds:
        ens_pred = np.mean(list(preds.values()), axis=0)
        preds['Ensemble'] = ens_pred
        results['Ensemble'] = calc_metrics(test, ens_pred)
        
    if not results:
        return {'error': 'All models failed.'}
        
    # --- BEST MODEL SELECTION (COMPOSITE SCORE) ---
    # Score = 0.4*Norm_MAPE + 0.4*Norm_RMSE + 0.2*Norm_Bias
    
    # 1. Create DataFrame for Metrics
    metric_df = pd.DataFrame(results).T # Index: ModelName, Cols: RMSE, MAPE, Bias, AbsBias
    
    def normalize(series):
        if series.max() == series.min():
            return np.zeros(len(series))
        return (series - series.min()) / (series.max() - series.min())
        
    metric_df['Norm_RMSE'] = normalize(metric_df['RMSE'])
    metric_df['Norm_MAPE'] = normalize(metric_df['MAPE'])
    metric_df['Norm_Bias'] = normalize(metric_df['AbsBias']) # Use Abs Bias magnitude
    
    metric_df['Score'] = (0.4 * metric_df['Norm_RMSE']) + (0.4 * metric_df['Norm_MAPE']) + (0.2 * metric_df['Norm_Bias'])
    
    best_model_name = metric_df['Score'].idxmin()
    best_rmse = metric_df.loc[best_model_name, 'RMSE'] # Return RMSE for display context
    
    # --- FINAL FORECAST (Next 12 Months) ---
    if progress_callback: progress_callback(90, f"Generating Forecast using {best_model_name}...")
    
    forecast_horizon = 12
    full_data = sales
    final_forecast_values = []
    
    # Retrain Best Model on Full Data
    try:
        if best_model_name == 'ETS':
            m = run_ets(full_data)
            final_forecast_values = m.forecast(forecast_horizon).values
        elif best_model_name == 'SARIMA':
            m = run_sarima(full_data)
            final_forecast_values = m.get_forecast(steps=forecast_horizon).predicted_mean.values
        elif best_model_name == 'Prophet':
            m = run_prophet(full_data)
            future = m.make_future_dataframe(periods=forecast_horizon, freq='MS')
            f = m.predict(future)
            final_forecast_values = f.iloc[-forecast_horizon:]['yhat'].values
        elif best_model_name == 'N-HiTS':
            ts_full = TimeSeries.from_series(full_data)
            m = run_nhits(ts_full)
            final_forecast_values = m.predict(forecast_horizon).values().flatten()
        elif best_model_name == 'Ensemble':
             # Simple Avg of ETS/SARIMA/Prophet for stability on full data
             f_vals = []
             m1 = run_ets(full_data)
             if m1: f_vals.append(m1.forecast(forecast_horizon).values)
             m2 = run_sarima(full_data)
             if m2: f_vals.append(m2.get_forecast(steps=forecast_horizon).predicted_mean.values)
             if f_vals:
                 final_forecast_values = np.mean(f_vals, axis=0)
             else:
                 final_forecast_values = np.zeros(forecast_horizon)
                 
    except Exception as e:
        return {'error': f'Final Forecast Gen Failed: {str(e)}'}
        
    # Lead Time (Simple Moving Average or similar)
    lt_series = sub['Average Lead Time']
    avg_lt = lt_series.mean() # Simple Baseline
    # Add some variability
    if len(final_forecast_values) > 0:
        # Scale variability relative to demand ?? No, just simple for now
        # Or use historical std dev
        lt_std = lt_series.std() if len(lt_series) > 1 else 0
        lt_forecast = np.random.normal(avg_lt, lt_std, forecast_horizon)
        lt_forecast = np.maximum(lt_forecast, 0) # Non-negative
    else:
        lt_forecast = []

    # Dates
    last_date = sales.index[-1]
    future_dates = pd.date_range(start=last_date + pd.offsets.MonthBegin(1), periods=forecast_horizon, freq='MS')
    
    return {
        'status': 'success',
        'best_model': best_model_name,
        'best_rmse': best_rmse,
        'history_dates': sales.index.tolist(),
        'history_values': sales.values.tolist(),
        'forecast_dates': future_dates.tolist(),
        'forecast_values': final_forecast_values.tolist(),
        'lead_time_forecast': lt_forecast.tolist(),
        'total_forecast_demand': np.sum(final_forecast_values),
        'avg_forecast_lead_time': np.mean(lt_forecast)
    }
