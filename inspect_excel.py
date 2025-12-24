import pandas as pd
try:
    df = pd.read_excel('Spare-Part-Data-With-Summary.xlsx')
    print("Columns:", df.columns.tolist())
    print("Head:\n", df.head())
except Exception as e:
    print(e)
