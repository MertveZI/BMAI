import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

t = np.linspace(0, 10, 1000)
x = np.cos(2 * t)          
y = np.sin(2 * t)
z = 0.5 * t                

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# ваш код здесь
sc = ax.scatter(x, y, z, 
                c=z, 
                cmap='viridis', 
                s=10, 
                alpha=0.9)

ax.set_title('Спиральная траектория в магнитном поле')
ax.set_xlabel('x (мм)')
ax.set_ylabel('y (мм)')
ax.set_zlabel('z (мм)')

ax.view_init(elev=30, azim=45)  
fig.colorbar(sc, ax=ax, label='z (мм)')  

ax.grid(True, alpha=0.3)  
plt.tight_layout()
plt.show()