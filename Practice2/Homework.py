import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# Установка параметров для визуализации
plt.rcParams['figure.figsize'] = (12, 6)
sns.set_theme(style="whitegrid")


def analyze_correlations(df):
    correlation_matrix = df.corr()
    
    # Визуализация матрицы корреляций
    plt.figure(figsize=(14, 10))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', center=0)
    plt.title('Матрица корреляций')
    plt.tight_layout()
    plt.show()
    
    # Анализ корреляции с целевой переменной
    target_correlations = correlation_matrix['MedHouseVal'].sort_values(ascending=False)
    print("\nКорреляция признаков с целевой переменной (MedHouseVal):")
    print(target_correlations)
    
    return target_correlations


def select_features(target_correlations):
    selected_features = target_correlations[1:6].index.tolist()
    print("\nВыбранные признаки для модели:")
    
    print(selected_features)
    
    return selected_features


def visualize_features(df, selected_features):
    """Визуализируем распределение выбранных признаков"""
    plt.figure(figsize=(16, 12))
    for i, feature in enumerate(selected_features):
        plt.subplot(2, 3, i+1)
        sns.scatterplot(x=df[feature], y=df['MedHouseVal'])
        plt.title(f'{feature} vs MedHouseVal')
        plt.xlabel(feature)
        plt.ylabel('MedHouseVal')
    plt.tight_layout()
    plt.show()


def prepare_data(df, selected_features):
    """Разделяем данные на признаки и целевую переменную"""
    X = df[selected_features]
    y = df['MedHouseVal']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Масштабируем данные
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\nРазмер обучающей выборки: {X_train.shape}")
    print(f"Размер тестовой выборки: {X_test.shape}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def mean_absolute_percentage_error(y_true, y_pred):
    """Функция для вычисления MAPE"""
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def train_linear_model(X_train, X_test, y_train, y_test, fit_intercept=True):
    """Обучаем линейную модель"""
    model = LinearRegression(fit_intercept=fit_intercept)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    metrics = {
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'MAE': mean_absolute_error(y_test, y_pred),
        'MAPE': mean_absolute_percentage_error(y_test, y_pred),
        'R²': r2_score(y_test, y_pred)
    }
    
    return model, y_pred, metrics


def compare_linear_models(X_train, X_test, y_train, y_test, selected_features):
    """Сравниваем модели с и без intercept"""
    model_with, y_pred_with, metrics_with = train_linear_model(
        X_train, X_test, y_train, y_test, fit_intercept=True
    )
    model_without, y_pred_without, metrics_without = train_linear_model(
        X_train, X_test, y_train, y_test, fit_intercept=False
    )
    
    # Выводим коэффициенты
    print("\nМодель С intercept:")
    print(f"Intercept: {model_with.intercept_:.4f}")
    for feature, coef in zip(selected_features, model_with.coef_):
        print(f"{feature}: {coef:.4f}")
    
    print("\nМодель БЕЗ intercept:")
    for feature, coef in zip(selected_features, model_without.coef_):
        print(f"{feature}: {coef:.4f}")
    
    # Сравниваем метрики
    print("\nСравнение метрик моделей:")
    print("Модель С intercept:")
    for metric, value in metrics_with.items():
        print(f"{metric}: {value:.4f}")
    
    print("\nМодель БЕЗ intercept:")
    for metric, value in metrics_without.items():
        print(f"{metric}: {value:.4f}")
    
    # Визуализация предсказаний
    plt.figure(figsize=(12, 6))
    plt.scatter(y_test, y_pred_with, alpha=0.5, label='С intercept')
    plt.scatter(y_test, y_pred_without, alpha=0.5, label='Без intercept')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 
             'r--', label='Идеальное предсказание')
    plt.xlabel('Фактическая цена')
    plt.ylabel('Предсказанная цена')
    plt.title('Сравнение моделей с и без intercept')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return model_with, y_pred_with, metrics_with, y_pred_without, metrics_without


def train_polynomial_model(X_train, X_test, y_train, y_test):
    """Обучаем модель с полиномиальными признаками"""
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    
    print(f"\nКоличество признаков после добавления полиномиальных: {X_train_poly.shape[1]}")
    
    model_poly = LinearRegression(fit_intercept=True)
    model_poly.fit(X_train_poly, y_train)
    y_pred_poly = model_poly.predict(X_test_poly)
    
    metrics_poly = {
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_poly)),
        'MAE': mean_absolute_error(y_test, y_pred_poly),
        'MAPE': mean_absolute_percentage_error(y_test, y_pred_poly),
        'R²': r2_score(y_test, y_pred_poly)
    }
    
    return model_poly, y_pred_poly, metrics_poly, X_train_poly


def compare_all_models(metrics_with, metrics_without, metrics_poly):
    """Сравниваем все три модели"""
    print("\nСравнение всех моделей:")
    print("Модель С intercept (линейная):")
    for metric, value in metrics_with.items():
        print(f"{metric}: {value:.4f}")
    
    print("\nМодель БЕЗ intercept (линейная):")
    for metric, value in metrics_without.items():
        print(f"{metric}: {value:.4f}")
    
    print("\nПолиномиальная модель:")
    for metric, value in metrics_poly.items():
        print(f"{metric}: {value:.4f}")


def visualize_model_predictions(y_test, y_pred_with, y_pred_poly):
    """Визуализация всех моделей"""
    plt.figure(figsize=(12, 6))
    plt.scatter(y_test, y_pred_with, alpha=0.5, label='Линейная (с intercept)')
    plt.scatter(y_test, y_pred_poly, alpha=0.5, label='Полиномиальная')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 
             'r--', label='Идеальное предсказание')
    plt.xlabel('Фактическая цена')
    plt.ylabel('Предсказанная цена')
    plt.title('Сравнение линейной и полиномиальной моделей')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def check_overfitting(model_poly, X_train_poly, y_train, metrics_poly):
    """Проверка на переобучение"""
    y_train_pred_poly = model_poly.predict(X_train_poly)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred_poly))
    test_rmse = metrics_poly['RMSE']
    
    print("\nПроверка на переобучение полиномиальной модели:")
    print(f"RMSE на обучающей выборке: {train_rmse:.4f}")
    print(f"RMSE на тестовой выборке: {test_rmse:.4f}")
    print(f"Разница: {test_rmse - train_rmse:.4f}")


def analyze_feature_importance(model_poly, selected_features):
    """Анализируем важность признаков в полиномиальной модели"""
    feature_names = []
    for i, feature in enumerate(selected_features):
        feature_names.append(feature)
        for j in range(i, len(selected_features)):
            feature_names.append(f"{feature}*{selected_features[j]}")
    
    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': model_poly.coef_
    })
    
    coef_df['Abs_Coefficient'] = np.abs(coef_df['Coefficient'])
    coef_df = coef_df.sort_values('Abs_Coefficient', ascending=False).head(10)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Coefficient', y='Feature', data=coef_df, palette='viridis')
    plt.title('Топ-10 самых важных признаков в полиномиальной модели')
    plt.tight_layout()
    plt.show()


def main():
    
    california = fetch_california_housing()
    df = pd.DataFrame(california.data, columns=california.feature_names)
    df['MedHouseVal'] = california.target
    
    # Анализируем корреляции
    target_correlations = analyze_correlations(df)
    
    # Выбираем признаки
    selected_features = select_features(target_correlations)
    
    # Визуализируем признаки
    visualize_features(df, selected_features)
    
    # Подготавливаем данные
    X_train, X_test, y_train, y_test, scaler = prepare_data(df, selected_features)
    
    # Сравниваем линейные модели
    model_with, y_pred_with, metrics_with, y_pred_without, metrics_without = \
        compare_linear_models(X_train, X_test, y_train, y_test, selected_features)
    
    # Обучаем полиномиальную модель
    model_poly, y_pred_poly, metrics_poly, X_train_poly = \
        train_polynomial_model(X_train, X_test, y_train, y_test)
    
    # Сравниваем все модели
    compare_all_models(metrics_with, metrics_without, metrics_poly)
    
    # Визуализация предсказаний
    visualize_model_predictions(y_test, y_pred_with, y_pred_poly)
    
    # Проверка на переобучение
    check_overfitting(model_poly, X_train_poly, y_train, metrics_poly)
    
    # Анализ важности признаков
    analyze_feature_importance(model_poly, selected_features)


if __name__ == "__main__":
    main()