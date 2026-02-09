import numpy as np


np.random.seed(42)

points = np.random.normal(loc=0.0, scale=4.0, size=(150, 3))
cov = np.cov(points, rowvar=False)

eigvals, eigvecs = np.linalg.eigh(cov)

print("Собственные значения (дисперсии по главным осям):")
print(np.sort(eigvals)[::-1])

print("\nГлавные направления (собственные векторы, отсортированные по убыванию):")
idx = np.argsort(eigvals)[::-1]
print(eigvecs[:, idx])
