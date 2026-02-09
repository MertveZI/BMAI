import pandas as pd
import numpy as np

np.random.seed(42)

# Time index
dates = pd.date_range('2025-01-01', periods=1000, freq='H')  # 1000 часов
df = pd.DataFrame(index=dates)

# Value
t_hours = (df.index - df.index[0]).total_seconds() / 3600
df['value'] = ___

# Category
df['category'] = np.random.choice(['___', '___', '___'], len(df))

# Resample
daily = df['value'].resample('___').agg(['___', '___', '___']) 
print("Daily stats:")
print(daily.head())

# Rolling
df['rolling_mean'] = df['value'].rolling(window=___).___()  # 7 дней = сколько часов?
df['rolling_std'] = df['value'].rolling(window=___).___()

# Max volatility
max_vol_day = df['rolling_std'].idxmax()
print("День с максимальной волатильностью:", max_vol_day)
print("Std в этот день:", df['rolling_std'].loc[max_vol_day])