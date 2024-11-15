import numpy as np
import matplotlib.pyplot as plt

# Parameters
fs = 1000  # Sampling rate
frequency = 200  # Signal frequency
duration = 0.5  # Signal duration
bits = 4  # Quantization bits
amplitude = 1  # Signal amplitude

# Generate sine wave
t = np.arange(0, duration, 1/fs)
signal = amplitude * np.sin(2 * np.pi * frequency * t)

# 4-bit Quantization
quantization_levels = 2**bits
step_size = 2 * amplitude / quantization_levels
quantized_signal = np.round(signal / step_size) * step_size

# Digital Bit Stream
digital_bit_stream = []
for sample in quantized_signal:
    digital_sample = int(np.round(sample / step_size))
    binary_sample = format(digital_sample, '04b')  # 4-bit binary representation
    digital_bit_stream.extend([int(bit) for bit in binary_sample])

# Plot original and quantized signals
plt.figure(figsize=(12, 6))

plt.subplot(121)
plt.plot(t, signal)
plt.title('Original Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

plt.subplot(122)
plt.plot(t, quantized_signal)
plt.title('Quantized Signal (4-bit)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

plt.tight_layout()
plt.show()

print("Digital Bit Stream:", digital_bit_stream)
``

This code:

1.  Generates a sine wave.
2.  Performs 4-bit quantization.
3.  Converts the quantized signal to a digital bit stream.
4.  Plots the original and quantized signals.

The digital bit stream is printed at the end.

Would you like any further clarification or modifications?

