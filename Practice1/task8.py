import pandas as pd

# Создайте DataFrame с колонками 'mass' и 'velocity'
data = {'mass': [1, 2, 3], 'velocity': [4, 5, 6]}
df = pd.DataFrame(data)  

# Добавьте колонку 'kinetic_energy'  с помощью лямбда функции
df['kinetic_energy'] = df.apply(lambda row:0.5 * row[data['mass']] * row[data['velocity']]**2, axis=1)  
print(df)

# Отфильтруйте строки, где kinetic_energy > 20
filtered = df.loc[20]  
print('Фильтрованные данные:', filtered)
