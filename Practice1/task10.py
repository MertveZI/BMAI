import pandas as pd
import numpy as np

np.random.seed(42)

# Создаем датафрейм df_measure
df_measure = pd.DataFrame({
    'sample_id': range(1, 501),
    'voltage': np.random.uniform(0, 10, 500),
    'current': np.random.uniform(0, 2, 500)
})

# df_calib
df_calib = pd.DataFrame({
    'calib_id': range(1, 101),
    'factor': np.random.uniform(1.0, 1.5, 100)
})

# Добавьте calib_id в df_measure
df_measure['calib_id'] = df_measure['sample_id'] % 100 + 1

# Merge
df_merged = pd.merge(df_measure, df_calib, on='calib_id', how='left')

# Corrected voltage
df_merged['corrected_voltage'] = df_merged['voltage'] * df_merged['factor']

# Группировка
grouped_calib = df_merged.groupby('calib_id')['corrected_voltage'].mean()  
print("Средняя corrected_voltage по calib_id:")
print(grouped_calib.head(10))