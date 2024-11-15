import numpy as np
import matplotlib.pyplot as plt
from cmath import exp, pi

def dft(x):
    """
    Performs the Discrete Fourier Transform (DFT) on the input signal x.

    Args:
        x: The input signal as a numpy array.

    Returns:
        The DFT of the input signal as a numpy array.
    """

    N = len(x)
    X = np.zeros(N, dtype=complex)
    for k in range(N):
        for n in range(N):
            X[k] += x[n] * exp(-2j * pi * k * n / N)
    return X

# Generate a sample signal
t = np.linspace(0, 1, 100)
x =[1,2,3,4,5]

# Compute the DFT
X = dft(x)

# Calculate frequencies
f_s = 1 / np.diff(t)[0]  # Sampling frequency
f = np.linspace(0, f_s / 2, len(X))  # Frequency range

# Calculate magnitudes and phases
magnitude = np.abs(X)
phase = np.angle(X, deg=True)

# Plot magnitude and phase spectra
#plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(f, magnitude)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.title('Magnitude Spectrum')
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(f, phase)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Phase (degrees)')
plt.title('Phase Spectrum')
#plt.grid(True)

#plt.tight_layout()
plt.show()
