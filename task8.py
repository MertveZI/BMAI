import pandas as pd

# Создайте DataFrame с колонками 'mass' и 'velocity'
data = {'mass': [1, 2, 3], 'velocity': [4, 5, 6]}
df = pd.DataFrame(___)  # TODO: data

# Добавьте колонку 'kinetic_energy'  с помощью лямбда функции
df['kinetic_energy'] = df.apply(lambda row:___ * row[___] * row[___]**___, axis=1)  
print(df)

# Отфильтруйте строки, где kinetic_energy > 20
filtered = df.loc[]  # TODO: 20
print('Фильтрованные данные:', filtered)
