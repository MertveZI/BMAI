import numpy as np

np.random.seed(42)

# ваша задача здесь

t = np.linspace(1000)  # 1000 точек
s = np.sin(2*np.pi*5*t) + 0.5*np.sin(2*np.pi*12*t) + шум?

fft_s = np.fft.fft(___) # смотрим документацию

freq = np.fft.fftfreq(___, d=___)  # TODO: len(t)

abs_fft = np.abs(fft_s[:len(fft_s)//2])
freq_pos = freq[:len(freq)//2]

peak1_idx = ___  # первый пик
peak1_freq = freq_pos[peak1_idx]

abs_fft_masked = abs_fft.copy()
abs_fft_masked[peak1_idx-5:peak1_idx+5] = 0  # маскируем окрестность
peak2_idx = ___  # второй пик
peak2_freq = ___

print("Пиковые частоты:", peak1_freq, peak2_freq)