import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


np.random.seed(42)
# Создайте случайный DataFrame 4x4
random_data = pd.DataFrame(
    np.random.randn(100, 4),  
    columns=['A', 'B', 'C', 'D']
)

# Вычислите корреляцию
corr = random_data.corr()  

# Постройте heatmap
sns.heatmap(corr, annot=True)  
plt.show()