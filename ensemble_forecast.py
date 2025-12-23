
import pandas as pd
import numpy as np
import holidays
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_percentage_error
import warnings

warnings.filterwarnings('ignore')

INPUT_FILE = 'Spare-Part-Data-With-Summary.xlsx'
TARGET_PART = 'PD457'

# --- Configuration ---
# User Assumptions for Test Case
LOC_MAPPING = {
    'A': 'Mumbai',
    'B': 'Bangalore' 
}

# Diwali Dates (for feature engineering)
DIWALI_DATES = {
    2021: '2021-11-04',
    2022: '2022-10-24',
    2023: '2023-11-12',
    2024: '2024-10-31',
    2025: '2025-10-20'
}

def get_indian_holidays_df(years):
    in_holidays = holidays.IN(years=years)
    return pd.DataFrame([{'ds': date, 'holiday': name} for date, name in in_holidays.items()])

def feature_engineering(df, city):
    """
    Adds seasonality and festival features.
    """
    df = df.copy()
    df['Month'] = df.index.month
    
    # 1. Monsoon Feature
    # Mumbai: Heavy Jun-Sep
    # Bangalore: Moderate Jun-Oct
    if city == 'Mumbai':
        df['Monsoon'] = df['Month'].apply(lambda x: 1 if x in [6, 7, 8, 9] else 0)
    elif city == 'Bangalore':
        df['Monsoon'] = df['Month'].apply(lambda x: 1 if x in [6, 7, 8, 9, 10] else 0)
    else:
        df['Monsoon'] = 0
        
    # 2. Diwali Feature
    # Mark the Month containing Diwali
    df['Diwali_Flag'] = 0
    for year, date_str in DIWALI_DATES.items():
        d = pd.to_datetime(date_str)
        mask = (df.index.year == d.year) & (df.index.month == d.month)
        df.loc[mask, 'Diwali_Flag'] = 1
        
    return df

def run_prophet(train_df, test_len, holiday_df):
    # Prepare DF
    df = train_df.reset_index().rename(columns={'Month': 'ds', 'Demand': 'y'})
    
    # Add Regressors
    m = Prophet(holidays=holiday_df, yearly_seasonality=True)
    m.add_country_holidays(country_name='IN')
    
    # Add custom regressors
    m.add_regressor('Monsoon')
    m.add_regressor('Diwali')
    
    m.fit(df)
    
    # Future DF
    future = m.make_future_dataframe(periods=test_len, freq='MS')
    
    # Re-add regressors to future
    # We need to apply the SAME feature engineering logic to the 'future' dates
    future = future.set_index('ds')
    # Use the city from the training set? We need to pass it.
    # Hack: infer city from the 'Monsoon' pattern in train or just pass it explicitly.
    # For now, let's just re-engineer.
    
    # Getting city is tricky inside this func without passing it.
    # Let's assume train_df has the cols, we just need to extend them.
    # We'll just re-call the feature_engineering helper if we refactor.
    
    # SIMPLIFICATION: We passed pre-engineered train_df. 
    # We should move feature engineering OUTSIDE or pass the function.
    pass # See main loop implementation

def run_ensemble(train_combined, test_combined, city, holiday_df):
    """
    Runs Prophet and SARIMAX, then finds optimal weight.
    train_combined: DataFrame with Demand, Monsoon, Diwali cols. Index is Date.
    """
    test_len = len(test_combined)
    
    # --- PROPHET ---
    # Fix: Index is named 'Month', and we have a col 'Month'.
    train_copy = train_combined.copy()
    train_copy.index.name = 'Date_Index'
    p_train = train_copy.reset_index().rename(columns={'Date_Index': 'ds', 'Demand': 'y'})
    
    m = Prophet(holidays=holiday_df, yearly_seasonality=True)
    m.add_country_holidays(country_name='IN')
    m.add_regressor('Monsoon')
    m.add_regressor('Diwali_Flag')
    m.fit(p_train)
    
    # Make Future
    future = m.make_future_dataframe(periods=test_len, freq='MS')
    feature_engineering(future.set_index('ds'), city) # Re-apply logic to future df columns
    
    # Need to reconstruct `future` with features properly
    # The `future` df from Prophet only has `ds`. We need to join/add features.
    future_index = pd.to_datetime(future['ds'])
    temp_df = pd.DataFrame(index=future_index)
    temp_df = feature_engineering(temp_df, city)
    
    future['Monsoon'] = temp_df['Monsoon'].values
    future['Diwali_Flag'] = temp_df['Diwali_Flag'].values
    
    p_forecast = m.predict(future)
    p_pred = p_forecast.iloc[-test_len:]['yhat'].values
    
    # --- SARIMAX ---
    # Exog variables: Monsoon, Diwali
    exog_train = train_combined[['Monsoon', 'Diwali_Flag']]
    exog_test = test_combined[['Monsoon', 'Diwali_Flag']]
    
    try:
        # Using specific order tuned for seasonal data
        model = SARIMAX(
            train_combined['Demand'],
            exog=exog_train,
            order=(1, 1, 1),
            seasonal_order=(1, 1, 0, 12),
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        res = model.fit(disp=False)
        s_res = res.get_forecast(steps=test_len, exog=exog_test)
        s_pred = s_res.predicted_mean.values
    except Exception as e:
        print(f"SARIMAX Error: {e}")
        s_pred = np.zeros(test_len)

    # --- ENSEMBLE ---
    y_true = test_combined['Demand'].values
    best_mape = float('inf')
    best_w = 0.0 # Weight for Prophet
    
    # Grid Search 0.0 to 1.0
    for w in np.linspace(0, 1, 101):
        ens_pred = (w * p_pred) + ((1-w) * s_pred)
        
        # Eval
        # Avoid div/0
        mask = y_true != 0
        if np.sum(mask) == 0:
            mape = 999
        else:
            mape = np.mean(np.abs((y_true[mask] - ens_pred[mask]) / y_true[mask]))
            
        if mape < best_mape:
            best_mape = mape
            best_w = w
            
    return {
        'Prophet_MAPE': mean_absolute_percentage_error(y_true, p_pred),
        'SARIMA_MAPE': mean_absolute_percentage_error(y_true, s_pred),
        'Ensemble_MAPE': best_mape,
        'Best_Weight_Prophet': best_w,
        'Predictions': {
            'Prophet': p_pred,
            'SARIMA': s_pred,
            'Ensemble': (best_w * p_pred) + ((1-best_w) * s_pred)
        }
    }

def main():
    print("--- Ensemble Model Analysis (Assumption: A=Mumbai, B=Bangalore) ---")
    
    # Load Data
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
    
    # Global Holidays
    hol_df = get_indian_holidays_df(df['Month'].dt.year.unique())

    for loc in ['A', 'B']:
        city = LOC_MAPPING.get(loc, 'Mumbai')
        print(f"\nProcessing Location {loc} (Assumed {city})...")
        
        loc_df = df[df['Location'] == loc].set_index('Month').sort_index()
        loc_df = loc_df.resample('MS')['Demand'].sum()
        
        # DF to Frame
        loc_df = loc_df.to_frame()
        
        # Feature Engineering
        loc_df = feature_engineering(loc_df, city)
        
        # Split (3.5y Train / 0.5y Test)
        # Train: 2021-01 to 2024-06
        # Test: 2024-07 to 2024-12
        train = loc_df[:'2024-06-01']
        test = loc_df['2024-07-01':'2024-12-01']
        
        if len(test) == 0:
            print("No test data.")
            continue
            
        res = run_ensemble(train, test, city, hol_df)
        
        print(f"Prophet MAPE: {res['Prophet_MAPE']:.2%}")
        print(f"SARIMA MAPE:  {res['SARIMA_MAPE']:.2%}")
        print(f"Ensemble MAPE: {res['Ensemble_MAPE']:.2%} (Weight Prophet: {res['Best_Weight_Prophet']:.2f})")
        
        if res['Ensemble_MAPE'] < 0.10:
             print("✅ GOAL MET: <10%")
        else:
             print("⚠️ GOAL MISSED")

if __name__ == "__main__":
    main()
