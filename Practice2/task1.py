
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn import metrics

pd.options.mode.chained_assignment = None
sns.set(font_scale=1.3, palette='Set2')

np.random.seed(42)

n = 50
I = np.random.uniform(0, 10, n)         # ток, А
R_true = 9.7                            # настоящее сопротивление
noise = np.random.normal(0, 10, n)      # шум измерения напряжения
V = I * R_true + noise                  # измеренное напряжение
Isq = I ** 2                            # квадрат тока

df = pd.DataFrame({'Ток_I': I, 'Напряжение_V': V})
df.head()

plt.figure(figsize=(10,6))
plt.scatter(df['Ток_I'], df['Напряжение_V'], alpha=0.7, edgecolor='w')
plt.xlabel('Ток, А')
plt.ylabel('Напряжение, В')
plt.title('Закон Ома + шум измерений')
plt.grid(True, alpha=0.3)
plt.show()

train, test = train_test_split(df, test_size=0.25, random_state=42)

X_train = train[['Ток_I']].values
y_train = train['Напряжение_V'].values

X_test  = test[['Ток_I']].values
y_test  = test['Напряжение_V'].values