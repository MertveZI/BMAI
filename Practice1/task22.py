import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

np.random.seed(42)

vars = ['T', 'n_e', 'B', 'pressure', 'E_field', 'v_x', 'v_y', 'v_z', 'Z_eff', 'impurity']
n = 300
# ── ваш код здесь ──

# Базовая случайная матрица
data = ___
corr = ___

# Добавим искусственную структуру
corr[0,1] = corr[1,0] = 0.82   # T ~ n_e
corr[2,3] = corr[3,2] = 0.75   # B ~ pressure
corr[5:8,5:8] += 0.4           # скорости коррелированы между собой

df_corr = pd.DataFrame(corr, index=vars, columns=vars)

# Обычный heatmap
plt.figure(figsize=(9, 7))
sns.heatmap(___, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.4, cbar_kws={'label': 'корреляция'})
plt.title("Матрица корреляций параметров плазмы")
plt.show()

# Clustermap
g = sns.clustermap( ___, vmin=-1, vmax=1, figsize=(10, 10), dendrogram_ratio=0.1, cbar_pos=(0.02, 0.8, 0.03, 0.18))

g.ax_heatmap.set_title("Кластеризованная корреляционная матрица", pad=40)
plt.show()