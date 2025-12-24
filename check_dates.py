import pandas as pd
try:
    df = pd.read_csv('Dashboard_Database.csv')
    actuals = df[df['Model'] == 'Actual']
    print(f"Min Date: {actuals['Date'].min()}")
    print(f"Max Date: {actuals['Date'].max()}")
    print(f"Row count: {len(actuals)}")
except Exception as e:
    print(e)
