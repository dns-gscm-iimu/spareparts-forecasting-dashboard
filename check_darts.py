
import pandas as pd
from darts import TimeSeries

df = pd.DataFrame({'a': [1,2,3]})
ts = TimeSeries.from_dataframe(df, value_cols='a')
print(dir(ts))
