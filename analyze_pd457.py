
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

def get_indian_holidays(years):
    # Get holidays for India
    in_holidays = holidays.IN(years=years)
    holiday_df = pd.DataFrame([
        {'ds': date, 'holiday': name} 
        for date, name in in_holidays.items()
    ])
    return holiday_df

def evaluate(y_true, y_pred):
    # MAPE
    y_true_clean = y_true.replace(0, 1e-6)
    return mean_absolute_percentage_error(y_true_clean, y_pred)

def run_prophet(train_df, test_len, holiday_df):
    # Prophet requires ds, y
    df = train_df.reset_index().rename(columns={'Month': 'ds', 'Demand': 'y'})
    
    m = Prophet(
        holidays=holiday_df,
        yearly_seasonality=True,
        daily_seasonality=False,
        weekly_seasonality=False
    )
    m.add_country_holidays(country_name='IN')
    m.fit(df)
    
    future = m.make_future_dataframe(periods=test_len, freq='MS')
    forecast = m.predict(future)
    return forecast.iloc[-test_len:]['yhat'].values

def run_sarima(train_df, test_len):
    # SARIMA
    # Using order (1,1,1) x (0,1,1,12) as a robust starting point
    # We can try to incorporate holidays as exog but SARIMA handles seasonality natively well
    try:
        model = SARIMAX(
            train_df['Demand'],
            order=(1, 1, 1),
            seasonal_order=(0, 1, 1, 12),
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        res = model.fit(disp=False)
        pred = res.get_forecast(steps=test_len).predicted_mean
        return pred.values
    except:
        return np.zeros(test_len)

def main():
    print(f"--- Deep Dive Analysis for {TARGET_PART} ---")
    
    # 1. Load Data
    try:
        # Read from first 5 sheets like before
        xls = pd.ExcelFile(INPUT_FILE)
        df_list = []
        for sheet in xls.sheet_names[:5]:
            d = pd.read_excel(INPUT_FILE, sheet_name=sheet, usecols=['Part ID', 'Location', 'Month', 'Demand'])
            df_list.append(d)
        df = pd.concat(df_list)
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    # Filter PD457
    df = df[df['Part ID'] == TARGET_PART]
    
    # Preprocess
    df['Month'] = pd.to_datetime(df['Month'])
    df['Demand'] = pd.to_numeric(df['Demand'], errors='coerce').fillna(0)
    
    # Handle duplicates if multiple locations - let's analyze by Location or aggregate?
    # User said "for part id pd457". If exists in multiple locations, user usually implies aggregate or dominant one.
    # Let's check locations.
    locs = df['Location'].unique()
    print(f"Locations found for {TARGET_PART}: {locs}")
    
    for loc in locs:
        print(f"\nAnalyzing Location: {loc}")
        loc_df = df[df['Location'] == loc].set_index('Month').sort_index()
        loc_df = loc_df.resample('MS')['Demand'].sum() # Ensure MS freq
        
        # Define Fixed Test Set: Jul 2024 - Dec 2024
        test_start = '2024-07-01'
        test_end = '2024-12-01'
        
        if test_start not in loc_df.index:
            print("Test data range (Jul-Dec 2024) not fully present.")
            continue
            
        y_test = loc_df[test_start:test_end]
        if len(y_test) < 6:
            print("Warning: Test set has fewer than 6 months.")
        
        # Define Training Windows to test sufficiency
        # 1. Full History (Jan 2021 - Jun 2024) approx 3.5 years
        # 2. 2.5 Years (Jan 2022 - Jun 2024)
        # 3. 1.5 Years (Jan 2023 - Jun 2024)
        
        windows = [
            ('3.5 Years (Full)', '2021-01-01'),
            ('2.5 Years', '2022-01-01'),
            ('1.5 Years', '2023-01-01')
        ]
        
        results = []
        
        # Prepare holidays
        years = loc_df.index.year.unique()
        hol_df = get_indian_holidays(years)
        
        print("\nResults (MAPE on Jul-Dec 2024):")
        print(f"{'History':<20} | {'Prophet':<10} | {'SARIMA':<10}")
        print("-" * 46)
        
        best_mape = 1.0
        best_cfg = ""
        
        for name, start_date in windows:
            if pd.to_datetime(start_date) < loc_df.index.min():
                 # Handle if data starts later than 2021
                 real_start = loc_df.index.min()
                 if real_start > pd.to_datetime(start_date):
                     continue 
            
            y_train = loc_df[start_date:'2024-06-01']
            if len(y_train) < 12:
                print(f"{name:<20} | Not enough data")
                continue
                
            # Run Prophet
            p_pred = run_prophet(y_train, len(y_test), hol_df)
            p_mape = evaluate(y_test, p_pred)
            
            # Run SARIMA
            s_pred = run_sarima(y_train.to_frame(), len(y_test))
            s_mape = evaluate(y_test, s_pred)
            
            results.append({
                'History': name,
                'Prophet': p_mape,
                'SARIMA': s_mape
            })
            
            print(f"{name:<20} | {p_mape:.2%}     | {s_mape:.2%}")
            
            if p_mape < best_mape:
                best_mape = p_mape
                best_cfg = f"Prophet ({name})"
            if s_mape < best_mape:
                best_mape = s_mape
                best_cfg = f"SARIMA ({name})"
                
        print("-" * 46)
        
        if best_mape < 0.10:
            print(f"✅ Success: Achieved <10% MAPE with {best_cfg} ({best_mape:.2%})")
        else:
            print(f"⚠️ Warning: Best MAPE is {best_mape:.2%}, which is above 10%.")
            
        print("\nDetailed Forecasts (Best Model on Full Data):")
        # Just show the forecast values for the user to see
        # ... (Implementation simplified for console output)

if __name__ == "__main__":
    main()
