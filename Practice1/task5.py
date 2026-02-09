import numpy as np


R = 0.0821 # газовая постоянная в л·атм/(моль·K)

temps = np.array([290, 300, 310, 320])
pressures = np.array([0.8, 1.0, 1.2, 1.5, 2.0])

density_inv = pressures / (R * temps[:, np.newaxis]) 

print("Обратная молярная плотность (л/моль):")
print(np.round(density_inv, 3))