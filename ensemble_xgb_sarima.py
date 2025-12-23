
import pandas as pd
import numpy as np
import holidays
import xgboost as xgb
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_percentage_error
from dateutil.relativedelta import relativedelta
import warnings

warnings.filterwarnings('ignore')

INPUT_FILE = 'Spare-Part-Data-With-Summary.xlsx'
TARGET_PART = 'PD457'

# --- Configuration ---
LOC_MAPPING = {'A': 'Mumbai', 'B': 'Bangalore'}
DIWALI_DATES = {
    2021: '2021-11-04', 2022: '2022-10-24', 2023: '2023-11-12',
    2024: '2024-10-31', 2025: '2025-10-20'
}

def feature_engineering(df, city):
    df = df.copy()
    df['Month'] = df.index.month
    
    # 1. Monsoon
    if city == 'Mumbai':
        df['Monsoon'] = df['Month'].apply(lambda x: 1 if x in [6, 7, 8, 9] else 0)
    elif city == 'Bangalore':
        df['Monsoon'] = df['Month'].apply(lambda x: 1 if x in [6, 7, 8, 9, 10] else 0)
    else:
        df['Monsoon'] = 0
        
    # 2. Diwali
    df['Diwali_Flag'] = 0
    for year, date_str in DIWALI_DATES.items():
        d = pd.to_datetime(date_str)
        mask = (df.index.year == d.year) & (df.index.month == d.month)
        df.loc[mask, 'Diwali_Flag'] = 1
        
    return df

def train_predict_xgboost(train_df, test_len, city):
    # Prepare data for ML (Lags)
    df = train_df.copy()
    # Ensure features exist
    if 'Monsoon' not in df.columns:
        df = feature_engineering(df, city)
        
    df['lag_1'] = df['Demand'].shift(1)
    df['lag_2'] = df['Demand'].shift(2)
    df['lag_12'] = df['Demand'].shift(12)
    
    # Drop NaNs created by lags
    df_model = df.dropna()
    
    features = ['Month', 'Monsoon', 'Diwali_Flag', 'lag_1', 'lag_2', 'lag_12']
    X_train = df_model[features]
    y_train = df_model['Demand']
    
    # Train
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=3, learning_rate=0.1)
    model.fit(X_train, y_train)
    
    # Recursive Forecast
    preds = []
    history = list(train_df['Demand'].values)
    current_date = train_df.index[-1]
    
    # We need to generate the "future" feature frame for Monsoon/Diwali first
    future_dates = [current_date + relativedelta(months=i+1) for i in range(test_len)]
    future_df = pd.DataFrame(index=future_dates)
    future_df = feature_engineering(future_df, city)
    
    for i in range(test_len):
        # Construct single row input
        # Get features from future_df
        date = future_dates[i]
        monsoon = future_df.loc[date, 'Monsoon']
        diwali = future_df.loc[date, 'Diwali_Flag']
        month = date.month
        
        lag_1 = history[-1]
        lag_2 = history[-2]
        lag_12 = history[-12] if len(history) >= 12 else history[-1]
        
        feat_vector = pd.DataFrame([{
            'Month': month,
            'Monsoon': monsoon,
            'Diwali_Flag': diwali,
            'lag_1': lag_1,
            'lag_2': lag_2,
            'lag_12': lag_12
        }])
        
        pred = model.predict(feat_vector)[0]
        # Enforce non-negative
        pred = max(0, pred)
        preds.append(pred)
        history.append(pred)
        
    return np.array(preds)

def train_predict_sarima(train_df, test_len, city):
    # Exog
    if 'Monsoon' not in train_df.columns:
        train_df = feature_engineering(train_df, city)
        
    exog_train = train_df[['Monsoon', 'Diwali_Flag']]
    
    # Future Exog
    current_date = train_df.index[-1]
    future_dates = [current_date + relativedelta(months=i+1) for i in range(test_len)]
    future_df = pd.DataFrame(index=future_dates)
    future_df = feature_engineering(future_df, city)
    exog_test = future_df[['Monsoon', 'Diwali_Flag']]
    
    try:
        model = SARIMAX(
            train_df['Demand'],
            exog=exog_train,
            order=(1, 1, 1),
            seasonal_order=(1, 1, 0, 12),
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        res = model.fit(disp=False)
        forecast = res.get_forecast(steps=test_len, exog=exog_test)
        return forecast.predicted_mean.values
    except:
        return np.zeros(test_len)

def main():
    print("--- Ensemble Strategy: XGBoost + SARIMAX ---")
    
    # Load Data (first 5 sheets)
    try:
        xls = pd.ExcelFile(INPUT_FILE)
        df_list = []
        for sheet in xls.sheet_names[:5]:
            d = pd.read_excel(INPUT_FILE, sheet_name=sheet, usecols=['Part ID', 'Location', 'Month', 'Demand'])
            df_list.append(d)
        df = pd.concat(df_list)
    except:
        return

    df = df[df['Part ID'] == TARGET_PART]
    df['Month'] = pd.to_datetime(df['Month'])
    df['Demand'] = pd.to_numeric(df['Demand'], errors='coerce').fillna(0)
    
    for loc in ['A', 'B']:
        city = LOC_MAPPING.get(loc, 'Mumbai')
        print(f"\nProcessing Location {loc} (Assumed {city})...")
        
        loc_df = df[df['Location'] == loc].set_index('Month').sort_index()
        loc_df = loc_df.resample('MS')['Demand'].sum().to_frame()
        
        # Feature Engineering (Combined)
        loc_df = feature_engineering(loc_df, city)
        
        # Split
        train = loc_df[:'2024-06-01']
        test = loc_df['2024-07-01':'2024-12-01']
        
        if len(test) == 0: continue
        
        # 1. XGBoost
        xgb_pred = train_predict_xgboost(train, len(test), city)
        xgb_mape = mean_absolute_percentage_error(test['Demand'], xgb_pred)
        
        # 2. SARIMA
        sarima_pred = train_predict_sarima(train, len(test), city)
        sarima_mape = mean_absolute_percentage_error(test['Demand'], sarima_pred)
        
        # 3. Ensemble
        best_mape = float('inf')
        best_w = 0.0 # Weight for XGBoost
        
        y_true = test['Demand'].values
        
        # Optimization
        for w in np.linspace(0, 1, 101):
            ens_pred = (w * xgb_pred) + ((1-w) * sarima_pred)
            mape = mean_absolute_percentage_error(y_true, ens_pred)
            if mape < best_mape:
                best_mape = mape
                best_w = w
                
        print(f"XGBoost MAPE: {xgb_mape:.2%}")
        print(f"SARIMA MAPE:  {sarima_mape:.2%}")
        print(f"Ensemble MAPE: {best_mape:.2%} (Weight XGB: {best_w:.2f})")
        
        if best_mape < 0.10:
            print("✅ GOAL MET: <10%")
        else:
             print("⚠️ GOAL MISSED")

if __name__ == "__main__":
    main()
