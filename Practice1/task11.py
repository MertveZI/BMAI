import pandas as pd
import numpy as np

np.random.seed(42)

# Данные
df = pd.DataFrame({
    'particle_type': np.random.choice(['alpha', 'beta', 'gamma'], 2000),
    'energy': np.random.normal(50, 20, 2000),  
    'angle': np.random.uniform(0, 180, 2000),   
    'time': np.random.uniform(0, 100, 2000)     
})

# Биннинг
energy_bins = pd.cut(df['energy'], bins=5, labels=[f'E{i}' for i in range(5)])  # нарезаем по бинам столбец с энергией
angle_bins = pd.cut(df['angle'], bins=4, labels=[f'A{i}' for i in range(4)])

# Pivot
pivot = pd.pivot_table(
    df,
    values='time',
    index='particle_type',
    columns=[energy_bins, angle_bins],
    aggfunc=['mean', 'count']  
)

pivot = pivot.fillna(0)

print("Pivot table:")
print(pivot.head())

# Максимум
max_mean_time = pivot['mean'].max().max()
max_idx = pivot['mean'].stack().idxmax()
print("Тип с максимальным средним временем в высоком energy_bin:", max_idx)