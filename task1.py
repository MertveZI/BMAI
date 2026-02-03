import numpy as np


velocities = np.array([1, 2, 3, 4, 5]) 
average_velocity = np.mean(velocities)
transformation_matrix = np.eye(2)
transformed = np.dot(transformation_matrix, velocities[:2])
print('Трансформированные скорости:', transformed)
# На будущее: как избавиться от точек в выводе?
# Изменение типа переменных в массиве на 'int64' ни к чему не приводит