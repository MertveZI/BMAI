import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

n = 800
types = np.random.choice(['pi_pos', 'pi_neg', 'K_pos', 'p'], size=n, p=[0.4, 0.4, 0.1, 0.1])

energy = np.where(types == 'p', np.random.lognormal(4.2, 0.6, n), np.random.lognormal(3.1, 0.5, n))

# ваш код здесь 

theta = np.random.uniform(0, 180, n)
px = energy * np.cos(np.deg2rad(theta)) * (1 + np.random.normal(0, 0.08, n))
py = energy * np.sin(np.deg2rad(theta)) * (1 + np.random.normal(0, 0.08, n))


df = pd.DataFrame({
    'particle_type': types,
    'energy': energy,
    'theta': theta,
    'px': px,
    'py': py
})

sns.pairplot(
    df,
    vars=['energy', 'theta', 'px', 'py'],
    plot_kws={'s': 18, 'alpha': 0.7, 'linewidth': 0},
)

plt.suptitle("Парные распределения характеристик частиц", y=1.02, fontsize=14)

plt.show()