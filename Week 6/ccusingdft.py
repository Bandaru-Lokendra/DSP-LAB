import numpy as np
import matplotlib.pyplot as plt

def circular_convolution(x, h):
    """
    Performs circular convolution of two sequences x and h.

    Args:
        x: First sequence as a numpy array.
        h: Second sequence as a numpy array.

    Returns:
        The circular convolution of x and h as a numpy array.
    """

    N = len(x)
    y = np.zeros(N)

    for n in range(N):
        for k in range(N):
            y[n] += x[k] * h[(n-k) % N]

    return y

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
            X[k] += x[n] * np.exp(-2j * np.pi * k * n / N)
    return X

def idft(X):
    """
    Performs the Inverse Discrete Fourier Transform (IDFT) on the input signal X.

    Args:
        X: The DFT of the input signal as a numpy array.

    Returns:
        The IDFT of the input signal as a numpy array.
    """

    N = len(X)
    x = np.zeros(N, dtype=complex)
    for n in range(N):
        for k in range(N):
            x[n] += X[k] * np.exp(2j * np.pi * k * n / N)
        x[n] /= N
    return x

# Example usage
x = np.array([1, 2, 3, 4])
h = np.array([5, 6, 7, 8])

# Direct circular convolution
#y1 = circular_convolution(x, h)

# Circular convolution using DFT properties
X = dft(x)
H = dft(h)
Y = X * H
y2 = idft(Y)

# Plot the results
plt.figure(figsize=(10, 6))

''''plt.subplot(2, 1, 1)
plt.stem(y1)
plt.xlabel('n')
plt.ylabel('y1[n]')
plt.title('Circular Convolution (Direct)')
plt.grid(True)'''

plt.subplot(2, 1, 2)
plt.stem(np.real(y2))  # Take the real part to remove numerical errors
plt.xlabel('n')
plt.ylabel('y2[n]')
plt.title('Circular Convolution (DFT Properties)')
plt.grid(True)

plt.tight_layout()
plt.show()
