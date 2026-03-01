# Импорт библиотек
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# ==========================================
# 1. Загрузка данных и анализ
# ==========================================

df = pd.read_csv('star_classification.csv')

print("\nLДатасет star_classification")
print(df.head())

print("\nИнформация о датасете:")
print(df.info())

print("\nОсновная статистика:")
print(df.describe())

print("\nКоличество объектов по классам:")
print(df['class'].value_counts())

plt.figure()
sns.countplot(data=df, x='class')
plt.title('Распределение классов (GALAXY, STAR, QSO)')
plt.show()

plt.figure()
sns.histplot(data=df, x='redshift', hue='class', element='step', alpha=0.5)
plt.title('Распределение redshift по классам')
plt.show()

# Цвета для признаков
df['u_g'] = df['u'] - df['g']
df['g_r'] = df['g'] - df['r']

plt.figure()
sns.scatterplot(data=df, x='u_g', y='g_r', hue='class', alpha=0.5, s=50)
plt.title('Диаграмма рассеяния: u-g vs g-r')
plt.show()

numeric_cols = ['u', 'g', 'r', 'i', 'z', 'redshift', 'u_g', 'g_r']
plt.figure(figsize=(10, 8))
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Корреляционная матрица признаков')
plt.show()

# ==========================================
# 2. Обработка данных
# ==========================================
 
# Удаляю колонки, где нет физических данных, это может внести шум
cols_to_drop = ['obj_ID', 'alpha', 'delta', 'run_ID', 'rerun_ID', 
                'cam_col', 'field_ID', 'spec_obj_ID', 'plate', 'MJD', 'fiber_ID']

data = df.drop(columns=cols_to_drop)

# Доп цвета для признаков
data['r_i'] = data['r'] - data['i']
data['i_z'] = data['i'] - data['z']

print("\nПропуски в данных после очистки:")
print(data.isnull().sum())

# Кодирование целевой переменной
le = LabelEncoder()
data['class_encoded'] = le.fit_transform(data['class'])
print("\nКодировка классов:", dict(zip(le.classes_, le.transform(le.classes_))))

# Разделение на признаки и таргет
X = data.drop(columns=['class', 'class_encoded'])
y = data['class_encoded']

# Разбиение на train / val / test (70% / 15% / 15%)
# Сначала отделим тест
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)
# Теперь валидацию от обучающей
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.176, random_state=42, stratify=y_train_val
)

print(f"\nРазмеры выборок: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")

# Нормализация признаков
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# 3. Обучение моделей
# ==========================================

# a) Без регуляризации
model_no_reg = LogisticRegression(C=1e10, solver='lbfgs', max_iter=1000, random_state=42)
model_no_reg.fit(X_train_scaled, y_train)

# b) L1 регуляризация
model_l1 = LogisticRegression(C=1.0, penalty='l1', solver='saga', max_iter=1000, random_state=42)
model_l1.fit(X_train_scaled, y_train)

# c) L2 регуляризация (стандартная)
model_l2 = LogisticRegression(C=1.0, penalty='l2', solver='lbfgs', max_iter=1000, random_state=42)
model_l2.fit(X_train_scaled, y_train)

# Функция для оценки
def evaluate(model, X, y, name):
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    print(f"\n{name} на тесте:")
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y, y_pred, target_names=le.classes_))
    return acc

# Оцениваем на тесте
evaluate(model_no_reg, X_test_scaled, y_test, "Без регуляризации")
evaluate(model_l1, X_test_scaled, y_test, "L1 регуляризация")
evaluate(model_l2, X_test_scaled, y_test, "L2 регуляризация")

# Матрица ошибок для лучшей модели
plt.figure(figsize=(8, 6))
y_pred_l2 = model_l2.predict(X_test_scaled)
sns.heatmap(confusion_matrix(y_test, y_pred_l2), annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Матрица ошибок (L2 регуляризация)')
plt.ylabel('Истинный класс')
plt.xlabel('Предсказанный класс')
plt.show()

