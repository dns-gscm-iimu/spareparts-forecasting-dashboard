
import pandas as pd
import matplotlib.pyplot as plt

INPUT_FILE = 'Spare-Part-Data-With-Summary.xlsx'
TARGET_PART = 'PD457'

def main():
    print(f"Profiling Seasonality for {TARGET_PART}...")
    
    # Load Data (first 5 sheets)
    try:
        xls = pd.ExcelFile(INPUT_FILE)
        df_list = []
        for sheet in xls.sheet_names[:5]:
            d = pd.read_excel(INPUT_FILE, sheet_name=sheet, usecols=['Part ID', 'Location', 'Month', 'Demand'])
            df_list.append(d)
        df = pd.concat(df_list)
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    df = df[df['Part ID'] == TARGET_PART]
    df['Month'] = pd.to_datetime(df['Month'])
    df['Demand'] = pd.to_numeric(df['Demand'], errors='coerce').fillna(0)
    
    # Extract Month Index (1-12)
    df['Month_Num'] = df['Month'].dt.month
    
    locs = df['Location'].unique()
    
    for loc in locs:
        loc_df = df[df['Location'] == loc]
        
        # Calculate Average Demand per Month
        monthly_avg = loc_df.groupby('Month_Num')['Demand'].mean()
        
        # Normalize for easier comparison (0-1 scale)
        norm_avg = (monthly_avg - monthly_avg.min()) / (monthly_avg.max() - monthly_avg.min())
        
        print(f"\n--- Location {loc} Seasonality Profile ---")
        # Print sort of ascii chart
        for m in range(1, 13):
            val = norm_avg.get(m, 0)
            bar = '#' * int(val * 20)
            print(f"Month {m:02d}: {bar} ({monthly_avg.get(m, 0):.1f})")
            
        # Heuristic Inference
        # Mumbai: High Jun-Sep (Months 6-9)
        # Chennai: High Oct-Dec (Months 10-12)
        # Bangalore: Mixed/Moderate
        
        avg_jjas = monthly_avg.loc[6:9].mean() if all(x in monthly_avg.index for x in range(6,10)) else 0
        avg_ond = monthly_avg.loc[10:12].mean() if all(x in monthly_avg.index for x in range(10,13)) else 0
        
        ratio = avg_jjas / (avg_ond + 1e-6)
        print(f"JJAS (Jun-Sep) Avg: {avg_jjas:.1f}")
        print(f"OND (Oct-Dec) Avg: {avg_ond:.1f}")
        print(f"Ratio JJAS/OND: {ratio:.2f}")
        
        if ratio > 1.2:
            print(">> Hints at MUMBAI (SW Monsoon dominance)")
        elif ratio < 0.8:
            print(">> Hints at CHENNAI (NE Monsoon dominance)")
        else:
            print(">> Ambiguous / Balanced (Could be Bangalore/Pune/Delhi)")
            
if __name__ == "__main__":
    main()
