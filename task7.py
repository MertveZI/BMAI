import numpy as np

np.random.seed(42)


data = np.random.exponential(scale = 2, size = 5000)

hist, bin_edges = np.histogram(data, bins=20)
bin_width = bin_edges[1] - bin_edges[0]
prob_hist = hist / (len(data) * bin_edges[0])  # нормализация
print("Нормализованная гистограмма (первые 5):", prob_hist[:5])


percentiles = np.percentile(data, [25, 50, 75])  # создаем лист с нужными перцентилями
print("Перцентили:", percentiles)

#Среднее и std
mean = np.mean(data)
std = np.std(data)
print("Среднее и std:", mean, std)

# Отбираем >5
num_above_5 = np.sum(data > 5)
print("Значений >5:", num_above_5)