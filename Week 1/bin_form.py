import numpy as np
import matplotlib.pyplot as plt

# Step 1: Generate Sine Wave

fs1 = 8000  # initial sampling rate
duration = 0.5  # signal duration
freq = 200  # signal frequency

t = np.arange(0, duration, 1/fs1)
x = np.sin(2 * np.pi * freq * t)

# Step 2: Resample Signal

fs2 = 1000  # resampling rate
t_resampled = np.arange(0, duration, 1/fs2)
x_resampled = np.interp(t_resampled, t, x)

# Step 3: Quantize Signal

levels = 8  # number of quantization levels
x_quantized = np.round(x_resampled * (levels - 1) / np.max(np.abs(x_resampled))) / (levels - 1)

# Step 4: Save Signal in Binary Form

x_quantized_int = x_quantized * (levels - 1)
x_quantized_int.astype(np.int8).tofile('signal.bin')

# Step 5: Plot Signals

plt.figure(figsize=(12, 6))

plt.subplot(121)
plt.plot(t, x)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Original Signal')

plt.subplot(122)
plt.plot(t_resampled, x_quantized)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Resampled and Quantized Signal')

plt.tight_layout()
plt.show()
