
import pandas as pd
import numpy as np
import warnings
from datetime import timedelta
from dateutil.relativedelta import relativedelta

# Time Series Models
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
import xgboost as xgb

# Evaluation Metrics
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

# Progress monitoring (optional, implies terminal usage)
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ---------------- CONFIGURATION ----------------
INPUT_FILE = 'Spare-Part-Data-With-Summary.xlsx'
OUTPUT_FORECAST_FILE = 'Forecast_Results.xlsx'
OUTPUT_METRICS_FILE = 'Model_Performance.csv'

FORECAST_HORIZON = 12  # months
TEST_SIZE = 12         # months, for validation

# ---------------- HELPER FUNCTIONS ----------------

def evaluate_forecast(y_true, y_pred):
    """Calculates RMSE and MAPE."""
    # Avoid div by zero in MAPE by replacing 0 with small epsilon or handling it
    y_true_clean = y_true.replace(0, 1e-6)
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true_clean, y_pred)
    return rmse, mape

def get_future_dates(last_date, steps=12):
    """Generates next 12 month-start dates."""
    dates = []
    current = last_date
    for _ in range(steps):
        current = current + relativedelta(months=1)
        dates.append(current)
    return pd.to_datetime(dates)

# ---------------- MODELS ----------------

def train_predict_ets(train, steps):
    try:
        # Auto-find best trend/seasonal config or default to Additive
        # Using a robust default config: Trend=Add, Seasonal=Add, Seasonal_Periods=12
        model = ExponentialSmoothing(
            train, 
            trend='add', 
            seasonal='add', 
            seasonal_periods=12,
            initialization_method="estimated"
        ).fit(optimized=True)
        pred = model.forecast(steps)
    except:
        # Fallback for short series or errors
        try:
            model = ExponentialSmoothing(train, trend='add', seasonal=None).fit()
            pred = model.forecast(steps)
        except:
             return pd.Series([train.mean()] * steps, index=get_future_dates(train.index[-1], steps))
    return pred

def train_predict_sarima(train, steps):
    try:
        # Using a simple order or auto-arima logic would be better but keeping it fixed/simple for speed
        # as requested 'SARIMA' without auto-arima library specification in requirements (using statsmodels)
        # Order (1,1,1) x (1,1,0,12) is a decent starting point for monthly data
        model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 0, 12), enforce_stationarity=False, enforce_invertibility=False)
        results = model.fit(disp=False)
        pred = results.get_forecast(steps=steps).predicted_mean
    except:
        return pd.Series([train.mean()] * steps, index=get_future_dates(train.index[-1], steps))
    return pred

def train_predict_xgboost(train, steps):
    # Prepare data for ML
    df_train = train.to_frame(name='y')
    df_train['ds'] = df_train.index
    df_train['month'] = df_train['ds'].dt.month
    df_train['year'] = df_train['ds'].dt.year
    df_train['lag_1'] = df_train['y'].shift(1)
    df_train['lag_2'] = df_train['y'].shift(2)
    df_train['lag_12'] = df_train['y'].shift(12)
    df_train.dropna(inplace=True)
    
    if len(df_train) < 5:
        return pd.Series([train.mean()] * steps, index=get_future_dates(train.index[-1], steps))

    features = ['month', 'year', 'lag_1', 'lag_2', 'lag_12']
    X_train = df_train[features]
    y_train = df_train['y']
    
    try:
        model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
        model.fit(X_train, y_train)
        
        # Recursive forecasting
        last_date = train.index[-1]
        future_dates = get_future_dates(last_date, steps)
        preds = []
        
        # We need to construct input for each step based on history + previous preds
        # This is tricky without rebuilding the full lag structure iteratively
        # Simplified approach: Use last observed values to feed first prediction, then append
        
        # Reconstruct full history to easier picking of lags
        history = list(train.values)
        
        current_date = last_date
        for i in range(steps):
            current_date += relativedelta(months=1)
            # Construct features
            # lag1 is history[-1]
            lag_1 = history[-1]
            lag_2 = history[-2] if len(history) >= 2 else history[-1]
            lag_12 = history[-12] if len(history) >= 12 else history[-1]
            
            feat_vector = pd.DataFrame([{
                'month': current_date.month,
                'year': current_date.year,
                'lag_1': lag_1,
                'lag_2': lag_2,
                'lag_12': lag_12
            }])
            
            pred_val = model.predict(feat_vector)[0]
            preds.append(pred_val)
            history.append(pred_val)
            
        return pd.Series(preds, index=future_dates)
        
    except Exception as e:
        print(f"XGB Error: {e}")
        return pd.Series([train.mean()] * steps, index=get_future_dates(train.index[-1], steps))

def train_predict_prophet(train, steps):
    try:
        df_prophet = train.reset_index()
        df_prophet.columns = ['ds', 'y']
        
        m = Prophet(daily_seasonality=False, weekly_seasonality=False, yearly_seasonality=True)
        m.fit(df_prophet)
        
        future = m.make_future_dataframe(periods=steps, freq='MS')
        forecast = m.predict(future)
        
        # Return only the future part
        pred_vals = forecast.iloc[-steps:]['yhat'].values
        future_dates = forecast.iloc[-steps:]['ds'].values
        return pd.Series(pred_vals, index=future_dates)
    except:
        return pd.Series([train.mean()] * steps, index=get_future_dates(train.index[-1], steps))


# ---------------- MAIN LOGIC ----------------

def main():
    print("Loading data from first 5 sheets...")
    
    try:
        xls = pd.ExcelFile(INPUT_FILE)
        sheets_to_process = xls.sheet_names[:5] # First 5 sheets
        print(f"Processing sheets: {sheets_to_process}")
        
        all_data = []
        for sheet in sheets_to_process:
            # Read sheet
            df_sheet = pd.read_excel(INPUT_FILE, sheet_name=sheet, usecols=['Part ID', 'Location', 'Month', 'Demand'])
            all_data.append(df_sheet)
            
        df = pd.concat(all_data, ignore_index=True)
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Drop first few rows if they contain header noise
    df = df.dropna(subset=['Month', 'Demand'])
    df['Month'] = pd.to_datetime(df['Month'])
    df['Demand'] = pd.to_numeric(df['Demand'], errors='coerce').fillna(0)
    
    # Group by SKU + Location
    groups = df.groupby(['Part ID', 'Location'])
    
    all_metrics = []
    final_forecasts = []
    
    print(f"Found {len(groups)} unique combinations. Starting processing...")
    
    for (part_id, loc), group_df in tqdm(groups):
        # Prepare time series
        ts_data = group_df.set_index('Month').sort_index()['Demand']
        ts_data = ts_data.resample('MS').sum().fillna(0) # Ensure no missing months
        
        if len(ts_data) < 12:
            print(f"Skipping {part_id}-{loc}: Not enough data ({len(ts_data)} months)")
            continue
            
        # 1. EVALUATION PHASE (Train/Test Split)
        train_eval = ts_data.iloc[:-TEST_SIZE]
        test_eval = ts_data.iloc[-TEST_SIZE:]
        
        if len(train_eval) < 12:
            # If data is too short to split, metric is NaN but we still forecast
            # Just use full data for "training" loosely or skip metric
            train_eval = ts_data # Fallback
            test_eval = None
            
        
        models_perf = {}
        
        # Define model functions pointer
        model_funcs = {
            'ETS': train_predict_ets,
            'SARIMA': train_predict_sarima,
            'XGBoost': train_predict_xgboost,
            'Prophet': train_predict_prophet
        }
        
        best_model_name = 'ETS' # Default
        best_rmse = float('inf')
        
        if test_eval is not None:
            for name, func in model_funcs.items():
                try:
                    preds = func(train_eval, len(test_eval))
                    rmse, mape = evaluate_forecast(test_eval, preds)
                    
                    models_perf[name] = {'RMSE': rmse, 'MAPE': mape}
                    
                    if rmse < best_rmse:
                        best_rmse = rmse
                        best_model_name = name
                        
                    # Save metric for Dashboard
                    all_metrics.append({
                        'Part ID': part_id,
                        'Location': loc,
                        'Model': name,
                        'RMSE': round(rmse, 2),
                        'MAPE': round(mape, 4)
                    })
                except Exception as e:
                    print(f"Error evaluating {name} for {part_id}-{loc}: {e}")
                    models_perf[name] = {'RMSE': 99999, 'MAPE': 9.99} # Penalize
        else:
            # Cannot evaluate, just pick one (Prophet is robust)
            best_model_name = 'Prophet' 
            # Log empty metrics
            for name in model_funcs.keys():
                all_metrics.append({
                        'Part ID': part_id,
                        'Location': loc,
                        'Model': name,
                        'RMSE': 0,
                        'MAPE': 0
                    })

        # 2. MARK BEST MODEL
        # We add a flag to the metrics df to indicate which was best
        for m in all_metrics:
            if m['Part ID'] == part_id and m['Location'] == loc and m['Model'] == best_model_name:
                m['Best_Model'] = True
            elif m['Part ID'] == part_id and m['Location'] == loc:
                m['Best_Model'] = False

        # 3. FINAL FORECAST (Refit on Full Data)
        # Use the chosen best model to forecast next 12 months
        chosen_func = model_funcs[best_model_name]
        try:
            future_forecast = chosen_func(ts_data, FORECAST_HORIZON)
        except:
             future_forecast = pd.Series([0]*FORECAST_HORIZON) 
        
        # Prepare Result Row
        result_row = {
            'Part ID': part_id,
            'Location': loc,
            'Best Model': best_model_name,
            'RMSE (Test)': models_perf.get(best_model_name, {}).get('RMSE', 0),
            'MAPE (Test)': models_perf.get(best_model_name, {}).get('MAPE', 0),
        }
        
        # Add monthly columns
        for date, val in future_forecast.items():
            month_str = date.strftime('%b-%Y')
            result_row[month_str] = round(val, 2)
            
        final_forecasts.append(result_row)
        
    # ---------------- SAVE OUTPUTS ----------------
    
    # 1. Forecast Results
    df_forecasts = pd.DataFrame(final_forecasts)
    df_forecasts.to_excel(OUTPUT_FORECAST_FILE, index=False)
    print(f"Forecasts saved to {OUTPUT_FORECAST_FILE}")
    
    # 2. Metrics for Dashboard
    df_metrics = pd.DataFrame(all_metrics)
    df_metrics.to_csv(OUTPUT_METRICS_FILE, index=False)
    print(f"Metrics saved to {OUTPUT_METRICS_FILE}")

if __name__ == "__main__":
    main()
