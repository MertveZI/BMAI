import numpy as np


arr = np.arange(48).reshape(6, 8)

print("Исходный массив:")
print(arr)

print("\nЭлемент [3,4] (нумерация с 0):", arr[3,4])

print("\nТретий столбец:")
print(arr[:,3])

print("\nПравая нижняя подматрица 3×4:")
print(arr[3:, 4:])