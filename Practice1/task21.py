import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

np.random.seed(42)

n = 1200
event_types = np.random.choice(['background', 'single-photon', 'two-photon'], n, p=[0.5, 0.3, 0.2])

energy = np.where(event_types == 'background', 
                  np.random.exponential(1.8, n),
                  np.where(event_types == 'single-photon',
                           np.random.gamma(3.2, 2.1, n),
                           np.random.gamma(5.5, 1.4, n)))

df = pd.DataFrame({
    'event_type': event_types,
    'energy_deposited': energy
})

plt.figure(figsize=(10, 6))

sns.violinplot(
    data=df, x='event_type', y='energy_deposited',
    palette='Set2', inner=None, linewidth=1.1, saturation=0.85
)

sns.boxenplot(
    data=df, x='event_type', y='energy_deposited',
    width=0.18, color='0.3', linewidth=1.4
)

sns.stripplot(
    data=df, x='event_type', y='energy_deposited',
    color='0.1', size=2.8, alpha=0.45, jitter=0.22
)

plt.title("Распределение отложенной энергии по типам событий")
plt.xlabel("Тип события")
plt.ylabel("Отложенная энергия (МэВ)")
plt.grid(alpha=0.15, axis='y')
plt.show()