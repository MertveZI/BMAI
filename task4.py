import numpy as np


np.random.seed(42)

points = np.random.normal(0, 1.5, (1, 3))

cov = np.cov(points, rowvar=False)

eigvals, eigvecs = np.linalg.eigh()

print("Собственные значения (дисперсии по главным осям):")
print(np.sort(eigvals)[::-1])

print("\nГлавные направления (собственные векторы, отсортированные по убыванию):")
idx = np.argsort(eigvals)[::-1]
print(eigvecs[:, idx])