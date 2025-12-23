
import pandas as pd
import openpyxl

try:
    xls = pd.ExcelFile('Spare-Part-Data-With-Summary.xlsx')
    print("Sheet names:", xls.sheet_names)
except Exception as e:
    print(f"Error: {e}")
