import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler



# Установка параметров для визуализации
plt.rcParams['figure.figsize'] = (12, 6)
sns.set_theme(style="whitegrid")

def analyze_missing_values(df):
    # процент пропусков
    missing_percentage = df.isna().mean().sort_values(ascending=False)
    
    # визуализация пропусков
    plt.figure(figsize=(12, 6))
    sns.heatmap(df.isna(), cbar=False, yticklabels=False, cmap='viridis')
    plt.title('Пропуски в данных (жёлтый = пропуск)')
    plt.tight_layout()
    plt.show()
    
    return missing_percentage

def clean_weather_data(df):

    df['Precip'] = df['Precip'].replace('T', '0.01').astype(float)
    df['Snowfall'] = df['Snowfall'].replace('#VALUE!', '0.0').astype(float)
    df['PRCP'] = df['PRCP'].replace('T', '0.01').astype(float)
    df['Date'] = pd.to_datetime(df['Date'])
    df['YR'] = df['Date'].dt.year
    df['MO'] = df['Date'].dt.month
    df['DA'] = df['Date'].dt.day

    df = df[['STA', 'Date', 'Precip', 'MaxTemp', 'MinTemp', 'MeanTemp','Snowfall', 'YR', 'MO', 'DA']]

    print("\nОбнаружены пропущенные значения. Заполняем их...")
    
    # создаем imputer
    imputer = SimpleImputer(strategy='mean')
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # заполняем пропуски
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    
    return df

def visualize_temperature_data(df):
    # Распределение средней температуры
    plt.figure(figsize=(10, 5))
    sns.histplot(df['MeanTemp'], bins=60, kde=True)
    plt.title('Распределение средней суточной температуры')
    plt.xlabel('Mean Temperature (°F)')
    plt.tight_layout()
    plt.show()
    
    # Средняя температура по годам и станциям
    year_station_temp = df.groupby(['YR', 'STA'])['MeanTemp'].mean().unstack()
    
    plt.figure(figsize=(14, 8))
    sns.heatmap(year_station_temp, cmap='coolwarm')
    plt.title('Средняя температура по годам и станциям')
    plt.tight_layout()
    plt.show()

def prepare_features(df):
    # Выбираем признаки, не относящиеся напрямую к температуре
    non_temp_features = ['Precip', 'Snowfall', 'YR', 'MO', 'DA']
    
    # Создаем матрицу признаков и целевую переменную
    X = df[non_temp_features]
    y = df['MeanTemp']
    
    return X, y

def train_and_evaluate_model(X, y):
    # Разделяем данные на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    # Масштабируем данные
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Обучаем модель
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    
    # Делаем предсказания
    y_pred = model.predict(X_test_scaled)
    
    # Оцениваем качество
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Анализ коэффициентов
    coefficients = dict(zip(X.columns, model.coef_))
    
    print(f"\nRMSE: {rmse:.3f}")
    print(f"MAE: {mae:.3f}")
    print(f"R²: {r2:.4f}")
    
    print("\nКоэффициенты модели:")
    for feature, coef in coefficients.items():
        print(f"{feature}: {coef:.4f}")
    
    # Визуализация предсказаний
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
    plt.xlabel('Фактическая температура')
    plt.ylabel('Предсказанная температура')
    plt.title('Фактические vs Предсказанные значения')
    plt.tight_layout()
    plt.show()
    
    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'coefficients': coefficients
    }

def train_model_without_intercept(X, y):
    # Разделяем данные на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    # Масштабируем данные
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Обучаем модель без intercept
    model = LinearRegression(fit_intercept=False)
    model.fit(X_train_scaled, y_train)
    
    # Делаем предсказания
    y_pred = model.predict(X_test_scaled)
    
    # Оцениваем качество
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Анализ коэффициентов
    coefficients = dict(zip(X.columns, model.coef_))
    
    print(f"\nRMSE (без intercept): {rmse:.3f}")
    print(f"MAE (без intercept): {mae:.3f}")
    print(f"R² (без intercept): {r2:.4f}")
    
    print("\nКоэффициенты модели (без intercept):")
    for feature, coef in coefficients.items():
        print(f"{feature}: {coef:.4f}")
    
    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'coefficients': coefficients
    }

def compare_models(X, y):
    # Обучаем модели
    model_with = train_and_evaluate_model(X, y)
    model_without = train_model_without_intercept(X, y)
    
    # Сравниваем метрики
    print("\nСравнение моделей:")
    print(f"RMSE с intercept: {model_with['rmse']:.3f}, без intercept: {model_without['rmse']:.3f}")
    print(f"R² с intercept: {model_with['r2']:.4f}, без intercept: {model_without['r2']:.4f}")

def main():
    df = pd.read_csv('Summary of weather.csv')
    # анализ пропусков
    analyze_missing_values(df)
    df_clean = clean_weather_data(df)
    
    # Визуализация 
    visualize_temperature_data(df_clean)

    X, y = prepare_features(df_clean)
    
    # Обучение модели
    train_and_evaluate_model(X, y)
    
    print("\nСравнение моделей с и без intercept...")
    compare_models(X, y)

if __name__ == "__main__":
    main()