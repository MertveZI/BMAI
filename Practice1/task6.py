import numpy as np


np.random.seed(42)

t = np.linspace(0, 10, num=1000) 
s = np.sin(2*np.pi*5*t) + 0.5*np.sin(2*np.pi*12*t) + np.random.normal(loc=0.0, scale=0.1)

fft_s = np.fft.fft(s, n=None, axis=-1, norm=None, out=None) 

freq = np.fft.fftfreq(len(t), d=t[1]-t[0]) 

abs_fft = np.abs(fft_s[:len(fft_s)//2])
freq_pos = freq[:len(freq)//2]

peak1_idx = np.argmax(abs_fft)  # первый пик
peak1_freq = freq_pos[peak1_idx]

abs_fft_masked = abs_fft.copy()
abs_fft_masked[peak1_idx-5:peak1_idx+5] = 0  
peak2_idx = np.argmax(abs_fft_masked)  # второй пик
peak2_freq = freq_pos[peak2_idx]

print("Пиковые частоты:", peak1_freq, peak2_freq)