import pandas as pd
import numpy as np

np.random.seed(42)

# Данные
reactants = np.random.choice(['A','B','C'], (300, 2))
df = pd.DataFrame({
    'reactant1': reactants[:,0],
    'reactant2': reactants[:,1],
    'rate': np.exp(-np.random.uniform(1,5,300)) * np.random.uniform(0.5,2,300),
    'T': np.random.uniform(200, 400, 300)
})

# Reaction_type
def get_type(row):
    return 'good' if row['reactant1'] == row['reactant2'] else 'bad'

df['reaction_type'] = df.apply(___, axis=1) 

# Нормализация rate по группе
df['normalized_rate'] = df.groupby(___)[___].transform(lambda x: ___)

# Кастомная функция
def activation_energy(row):
    return ___

df['E_act'] = df.apply(___, axis=1)

print("Средняя E_act по типу реакции:")
print(df.groupby('reaction_type')['E_act'].mean())