
import pandas as pd
import numpy as np
import warnings
from darts import TimeSeries
from darts.models import NBEATSModel, RNNModel
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_percentage_error
import logging
import torch

# Suppress warnings
warnings.filterwarnings('ignore')
logging.getLogger("darts").setLevel(logging.WARNING)
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

INPUT_FILE = 'Spare-Part-Data-With-Summary.xlsx'
TARGET_PART = 'PD457'

LOC_MAPPING = {'A': 'Mumbai', 'B': 'Bangalore'}

# --- helper ---
def get_mape(y_true, y_pred):
    y_true_clean = np.where(y_true == 0, 1e-6, y_true)
    return np.mean(np.abs((y_true - y_pred) / y_true_clean))

def run_sarima(train_series, val_len):
    # Convert Darts series to pandas
    train_df = train_series.to_dataframe()['Demand']
    
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
    # N-BEATS requires a bit of lookback
    input_chunk = min(12, len(train_series)//2)
    model = NBEATSModel(
        input_chunk_length=input_chunk,
        output_chunk_length=val_len, # Forecast full horizon at once
        n_epochs=50,
        random_state=42,
        force_reset=True,
        pl_trainer_kwargs={"accelerator": "cpu", "enable_progress_bar": False}
    )
    try:
        model.fit(train_series, verbose=False)
        pred = model.predict(val_len)
        return pred.values().flatten()
    except Exception as e:
        print(print(f"NBEATS Error: {e}"))
        return np.zeros(val_len)

def run_lstm(train_series, val_len):
    input_chunk = 12
    model = RNNModel(
        model='LSTM',
        hidden_dim=25,
        n_rnn_layers=1,
        dropout=0.1,
        input_chunk_length=input_chunk,
        n_epochs=50,
        random_state=42,
        force_reset=True,
        pl_trainer_kwargs={"accelerator": "cpu", "enable_progress_bar": False}
    )
    try:
        model.fit(train_series, verbose=False)
        pred = model.predict(val_len)
        return pred.values().flatten()
    except Exception as e:
         print(f"LSTM Error: {e}")
         return np.zeros(val_len)

def main():
    print(f"--- Advanced Ensemble Exploration for {TARGET_PART} ---")
    
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
    
    # Define Splits
    # Split A: Train 3 Years (Jan 21 - Dec 23) -> Test 1 Year (Jan 24 - Dec 24)
    # Split B: Train 3.5 Years (Jan 21 - Jun 24) -> Test 0.5 Year (Jul 24 - Dec 24)
    
    splits = [
        {'name': 'Split A (3y Train / 1y Test)', 'train_end': '2023-12-01', 'test_start': '2024-01-01', 'test_end': '2024-12-01'},
        {'name': 'Split B (3.5y Train / 0.5y Test)', 'train_end': '2024-06-01', 'test_start': '2024-07-01', 'test_end': '2024-12-01'}
    ]
    
    for loc in ['A', 'B']:
        print(f"\nLocation {loc} (Assumed {LOC_MAPPING.get(loc)})")
        print("="*60)
        
        loc_df = df[df['Location'] == loc].set_index('Month').sort_index()
        loc_df = loc_df.resample('MS')['Demand'].sum()
        
        # Convert to Darts TimeSeries
        series = TimeSeries.from_dataframe(loc_df.to_frame(), value_cols='Demand')
        
        for split in splits:
            print(f"\nRunning {split['name']}...")
            
            # Divide Data
            train_end_date = pd.Timestamp(split['train_end'])
            train_series, val_series = series.split_after(train_end_date)
            
            # Ensure val_series matches test_end goal
            # Darts split_after takes everything after.
            # We must clip val_series to test_end just in case data extends beyond (it generally doesn't)
            
            val_len = len(val_series)
            if val_len == 0:
                print("Not enough data for this split.")
                continue
                
            y_true = val_series.values().flatten()
            
            # 1. SARIMA
            p_sarima = run_sarima(train_series, val_len)
            m_sarima = get_mape(y_true, p_sarima)
            
            # 2. N-BEATS
            p_nbeats = run_nbeats(train_series, val_len)
            m_nbeats = get_mape(y_true, p_nbeats)
            
            # 3. LSTM
            p_lstm = run_lstm(train_series, val_len)
            m_lstm = get_mape(y_true, p_lstm)
            
            # 4. Ensemble (Average of All 3)
            p_ens = (p_sarima + p_nbeats + p_lstm) / 3
            m_ens = get_mape(y_true, p_ens)
            
            # 5. Ensemble (Weighted: 50% Sarima, 25% DLs)
            p_ens_w = (0.5 * p_sarima) + (0.25 * p_nbeats) + (0.25 * p_lstm)
            m_ens_w = get_mape(y_true, p_ens_w)
            
            print(f"{'Model':<15} | MAPE")
            print("-" * 25)
            print(f"{'SARIMA':<15} | {m_sarima:.2%}")
            print(f"{'N-BEATS':<15} | {m_nbeats:.2%}")
            print(f"{'LSTM':<15} | {m_lstm:.2%}")
            print(f"{'Ens (Avg)':<15} | {m_ens:.2%}")
            print(f"{'Ens (Wtd)':<15} | {m_ens_w:.2%}")
            
            if m_sarima < 0.1: print(">> SARIMA met <10%")
            if m_nbeats < 0.1: print(">> N-BEATS met <10%")
            if m_lstm < 0.1: print(">> LSTM met <10%")
            if m_ens_w < 0.1: print(">> Weighted Ensemble met <10%")

if __name__ == "__main__":
    main()
