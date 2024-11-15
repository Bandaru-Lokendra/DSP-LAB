import numpy as np
import matplotlib.pyplot as plt

# Generate a sample signal
time = np.linspace(0, 10, 1000)
signal = np.sin(2 * np.pi * 5 * time) + np.sin(2 * np.pi * 10 * time)

# Perform FFT
fft_result = np.fft.fft(signal)

# Calculate frequencies
freq = np.fft.fftfreq(len(signal), d=time[1] - time[0])

# Plot the time-domain signal
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(time, signal)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Time Domain Signal')

# Plot the frequency-domain spectrum
plt.subplot(2, 1, 2)
plt.plot(freq, np.abs(fft_result))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.title('Frequency Domain Spectrum')
plt.xlim(0, 20)  # Adjust x-axis limits to focus on relevant frequencies

plt.tight_layout()
plt.show()
