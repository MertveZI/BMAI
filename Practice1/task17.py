import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

t = np.linspace(0, 30, 1500)
x = np.exp(-0.15 * t) * np.sin(3.5 * t + 0.8)
v = np.gradient(x, t)

# FFT 
N = len(t)
yf = fft(x)  
xf = fftfreq(N, d=t[1] - t[0])[:N//2]
power = 2.0 / N * np.abs(yf[:N//2])**2 

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Гармонический осциллятор с затуханием', fontsize=16)

# [0,0] — график x(t)
axes[0,0].plot(t, x, 'b-', lw=1.4)
axes[0,0].set_title('Положение x(t)')

# [0,1] — график v(t)
axes[0,1].plot(t, v, 'orange', lw=1.4)
axes[0,1].set_title('Скорость v(t)')

# [1,0] — фазовый портрет v(x)
axes[1,0].plot(x, v, 'k.', ms=2, alpha=0.5)
axes[1,0].plot(x, v, 'k-', lw=0.6, alpha=0.3)   # спираль
axes[1,0].set_title('Фазовый портрет v(x)')
axes[1,0].set_xlabel('x')
axes[1,0].set_ylabel('v')

# [1,1] — спектр мощности
axes[1,1].stem(xf, power, linefmt='b-', markerfmt='bo', basefmt='k-')
axes[1,1].set_title('Спектр мощности')
axes[1,1].set_xlim(0, 2.0)  
axes[1,1].set_xlabel('Частота (Гц)')


plt.tight_layout()
plt.show()