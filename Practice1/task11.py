import pandas as pd
import numpy as np

np.random.seed(42)

# Данные
df = pd.DataFrame({
    'particle_type': np.random.choice(['alpha', 'beta', 'gamma'], 2000),
    'energy': ___,
    'angle': ___,
    'time': ___})

# Биннинг
energy_bins = pd.cut(df['energy'], bins=___, labels=[f'E{i}' for i in range(5)])  # нарезаем по бинам столбец с энергией
angle_bins = pd.cut(df['angle'], bins=___, labels=[f'A{i}' for i in range(4)])

# Pivot
pivot = pd.pivot_table(
    df,
    values=___,
    index=___,
    columns=[energy_bins, angle_bins],
    aggfunc=[___, ___]  
)

pivot = pivot.fillna(___)

print("Pivot table:")
print(pivot.head())

# Максимум
max_mean_time = pivot['mean'].max().max()
max_idx = pivot['mean'].stack().idxmax()
print("Тип с максимальным средним временем в высоком energy_bin:", max_idx)