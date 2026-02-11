import numpy as np
import matplotlib.pyplot as plt


# ваш код здесь
t = np.linspace(0, 20, 300)
theory = 100 * np.exp(-t / 5.2)
exp = theory + np.random.normal(0, np.sqrt(4 + 0.3 * t))

plt.figure(figsize=(11, 6))

plt.plot(t, theory, 'r-', lw=2.5, label='Теория')
plt.plot(t, exp, 'bo', ms=4, alpha=0.7, label='Эксперимент')

plt.axhline(50, color='gray', ls='--', lw=1.2, alpha=0.6)

idx = np.argmin(np.abs(t - 8))
plt.annotate(r'$T_{1/2} \approx 3.6$', xy=(t[idx], theory[idx]), 
             xytext=(12, 60),
             arrowprops=dict(arrowstyle='->', color='darkgreen', lw=1.5),
             fontsize=11, color='darkgreen', fontweight='bold')

plt.xlabel('Время (с)')
plt.ylabel('Количество частиц')
plt.legend(loc='upper right', frameon=True, shadow=True)
plt.grid(alpha=0.3)
plt.title('Распад радиоактивного образца')
plt.tight_layout()
plt.show()