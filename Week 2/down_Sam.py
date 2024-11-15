import numpy as np
import matplotlib.pyplot as plt

# Original signal
original_signal = np.array([1, 2, 3, 4, 5, 6, 7])

# Downsampling factor
downsampling_factor = 2

# Downsampled signal
downsampled_signal = original_signal[::downsampling_factor]
print(original_signal)
print(downsampled_signal)

# Plot the original and downsampled signals
plt.figure(figsize=(10, 6))
plt.stem(np.arange(len(original_signal)), original_signal, linefmt='b-', markerfmt='bo', basefmt='b-', label='Original Signal')
plt.stem(np.arange(len(downsampled_signal)), downsampled_signal, linefmt='r-', markerfmt='ro', basefmt='r-', label='Downsampled Signal')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)
plt.show()
