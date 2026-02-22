import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn import metrics


pd.options.mode.chained_assignment = None
sns.set_theme(font_scale=1.3, palette='Set2')
np.random.seed(42)

n = 50                                  # количество точек
I = np.random.uniform(0, 10, n)         # ток, А
noise = np.random.normal(0, 10, n)      # шум измерения напряжения, В
# При увеличении шума предсказание напряжения стремится к шуму
Isq = I ** 2                            # квадрат тока
V = 3 * (I ** 2) + 2 * I + noise        # измеренное напряжение с нелинейной зависимостью, В

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


#Способ 1 - аналитическое решение

# Добавляем столбец единиц для свободного члена
X_train_b = np.c_[np.ones(len(X_train)), X_train]
X_test_b  = np.c_[np.ones(len(X_test)),  X_test]

# Аналитическая формула МНК
theta_best = np.linalg.inv(X_train_b.T @ X_train_b) @ X_train_b.T @ y_train

print(f"Аналитическое решение:\n  theta_0 (intercept) = {theta_best[0]:.4f} В")
print(f"  theta_1 (коэффициент) = {theta_best[1]:.4f} В/А  →  R = {theta_best[1]:.4f} Ом")


#Способ 2 - Sklearn (градиентный спуск внутри)
lr = SGDRegressor(loss='squared_error', penalty=None)
lr.fit(X_train, y_train)

print(f"Sklearn:\n  intercept = {lr.intercept_[0]:.4f} В")
print(f"  coef_  = {lr.coef_[0]:.4f} В/А  →  R = {lr.coef_[0]:.4f} Ом")

y_pred_analytic = X_test_b @ theta_best
y_pred_sklearn  = lr.predict(X_test)

rmse_analytic = np.sqrt(metrics.mean_squared_error(y_test, y_pred_analytic))
rmse_sklearn  = np.sqrt(metrics.mean_squared_error(y_test, y_pred_sklearn))

print(f"RMSE (аналитическое решение) = {rmse_analytic:.3f} В")
print(f"RMSE (sklearn)              = {rmse_sklearn:.3f} В")

# Отображение результатов
plt.figure(figsize=(10,6))
plt.scatter(X_test, y_test, alpha=0.6, label='тестовые данные')
plt.plot(X_test, y_pred_sklearn, 'r-', lw=2.5, label='sklearn')
plt.plot(X_test, y_pred_analytic, '--', color='lime', lw=2.5, label='аналитическое МНК')
plt.xlabel('Ток, А')
plt.ylabel('Напряжение, В')
plt.title(f'Предсказания линейной регрессии\nRMSE ≈ {rmse_sklearn:.2f} В')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()




