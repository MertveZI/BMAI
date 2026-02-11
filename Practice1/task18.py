import numpy as np
import matplotlib.pyplot as plt


x = np.linspace(-4, 4, 80)
y = np.linspace(-4, 4, 80)
X, Y = np.meshgrid(x, y)

r1 = np.sqrt((X + 1.5)**2 + Y**2 + 0.3**2)
r2 = np.sqrt((X - 1.5)**2 + Y**2 + 0.3**2)
V = 1 / r1 - 1.5 / r2  

dV_dx = np.gradient(V, x, axis=1)  
dV_dy = np.gradient(V, y, axis=0) 
Ex, Ey = -dV_dx, -dV_dy  

fig, ax = plt.subplots(figsize=(10, 8))

im = ax.pcolormesh(X, Y, V, cmap='RdBu_r', shading='auto', vmin=-3, vmax=3)
fig.colorbar(im, ax=ax, label='Потенциал V')

ax.contour(X, Y, V, levels=np.linspace(-3,3,15), colors='k', linewidths=0.8, alpha=0.6)

skip = 6
ax.quiver(X[::skip,::skip], Y[::skip,::skip],
          Ex[::skip,::skip], Ey[::skip,::skip],
          scale=40, color='darkgreen', alpha=0.7)

ax.set_aspect('equal')
ax.set_title('Потенциал двух зарядов (+1 и -1.5) + электрическое поле', fontsize=14)
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()