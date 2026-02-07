import pandas as pd
import numpy as np

np.random.seed(42)

#Генерация данных (самостоятельно создайте 1000 строк)
n = 1000
groups = np.random.choice(['A', 'B'], size=n, p=[0.6, 0.4])
temps_A = np.random.normal(300, 20, sum(groups=='A'))
temps_B = np.random.normal(400, 30, sum(groups=='B'))
temps = np.concatenate([temps_A, temps_B])
pressures = np.random.uniform(1, 10, n)
energies = temps * pressures + np.random.normal(loc=0.0, scale=50)

df = pd.DataFrame({'group': groups, 'temperature': temps, 'pressure': pressures,'energy': energies})

# Добавьте пропуски (10% случайных)
df.iloc[np.random.choice(df.index, 100), :] = np.nan

# Очистка
df_clean = df.dropna()

# Группировка и аггрегация
grouped = df_clean.groupby('group').agg({
    'temperature': 'mean',     # mean
    'pressure': 'median',        # median
    'energy': 'sum'           # sum
})
print("Группировка:")
print(grouped)

# Новая колонка
df_clean['efficiency'] = (df_clean['energy'] / df_clean['temperature']).replace(0, np.nan)
# Фильтрация и топ
high_eff = df_clean.groupby('group')['efficiency'].mean()
high_groups = high_eff[high_eff > 8].index
top3 = df_clean[df_clean['group'].isin(high_groups)].nlargest(3, 'energy')
print("Топ-3 по энергии в эффективных группах:")
print(top3)