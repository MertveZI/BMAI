import numpy as np


np.random.seed(42)
data = np.random.normal(0, 1.5, 200)  

mask = np.abs(data) > 2

print("Количество выбросов |x| > 2:", mask.sum())
data[mask] = np.nan

mean_clean = np.nanmean(data)
print("Среднее без выбросов:", mean_clean)