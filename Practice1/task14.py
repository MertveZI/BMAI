import matplotlib.pyplot as plt
import numpy as np  

# Создайте массив x от 0 до 2*pi с 100 точками
x = np.linspace(0, 2 * np.pi, 100)

# Вычислите y = sin(x)
y = np.sin(x)

# Постройте график
plt.plot(x, y)
plt.title('Синусоидальная волна')
plt.xlabel('Время')
plt.ylabel('Амплитуда')
plt.show()  # show it