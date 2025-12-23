
import pandas as pd

try:
    df = pd.read_excel('Spare-Part-Data-With-Summary.xlsx')
    print("Columns:", df.columns.tolist())
    print("First 5 rows:")
    print(df.head())
    print("\nData Types:")
    print(df.dtypes)
except Exception as e:
    print(f"Error reading file: {e}")
