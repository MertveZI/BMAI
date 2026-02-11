import pandas as pd
import numpy as np

np.random.seed(42)

# Time index
dates = pd.date_range('2025-01-01', periods=1000, freq='h')  # 1000 часов
df = pd.DataFrame(index=dates)

# Value
t_hours = (df.index - df.index[0]).total_seconds() / 3600
df['value'] = np.sin(2 * np.pi * t_hours / 24) + np.random.normal(0, 0.1, len(df)) + 0.001 * t_hours

# Category
df['category'] = np.random.choice(['A', 'B', 'C'], len(df))

# Resample
daily = df['value'].resample('d').agg(['mean', 'max', 'min']) 
print("Daily stats:")
print(daily.head())

# Rolling
df['rolling_mean'] = df['value'].rolling(window=168).mean()  # 7 дней = 168 часов?
df['rolling_std'] = df['value'].rolling(window=168).std()
# Max volatility
max_vol_day = df['rolling_std'].idxmax()
print("День с максимальной волатильностью:", max_vol_day)
print("Std в этот день:", df['rolling_std'].loc[max_vol_day])