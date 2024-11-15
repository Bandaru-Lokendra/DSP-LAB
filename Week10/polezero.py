import numpy as np
import matplotlib.pyplot as plt

# Define the transfer function coefficients
num = [1, -2, 2, -1]  # Numerator coefficients (zeros)
den = [1, -1.5, 0.7, -0.1]  # Denominator coefficients (poles)

# Find the zeros and poles
zeros = np.roots(num)
poles = np.roots(den)

# Plot the zeros and poles
plt.figure(figsize=(8, 6))
plt.scatter(np.real(zeros), np.imag(zeros), marker='o', color='red', label='Zeros')
plt.scatter(np.real(poles), np.imag(poles), marker='x', color='blue', label='Poles')
plt.grid(True)
plt.xlabel('Real')
plt.ylabel('Imaginary')
plt.title('Pole-Zero Plot')
plt.legend()
plt.axis('equal')
plt.show()

# Calculate the frequency response manually
def freq_response(num, den, w):
    h = np.polyval(num, np.exp(-1j*w)) / np.polyval(den, np.exp(-1j*w))
    return h

# Frequency range
w = np.linspace(0, np.pi, 1000)

# Calculate magnitude and phase response
h = freq_response(num, den, w)

# Plot the magnitude response
plt.figure(figsize=(8, 6))
plt.plot(w, 20*np.log10(abs(h)))
plt.xlabel('Frequency (rad/sample)')
plt.ylabel('Magnitude (dB)')
plt.title('Magnitude Response')
plt.grid(True)
plt.show()

# Plot the phase response
plt.figure(figsize=(8, 6))
plt.plot(w, np.angle(h))
plt.xlabel('Frequency (rad/sample)')
plt.ylabel('Phase (radians)')
plt.title('Phase Response')
plt.grid(True)
plt.show()
