import numpy as np
import matplotlib.pyplot as plt

def z_transform(x, z):
  """
  Calculates the Z-transform of a discrete-time signal x[n].

  Args:
    x: The input signal as a numpy array.
    z: The complex variable z.

  Returns:
    The Z-transform of x at the given value of z.
  """

  X = np.sum(x * z**(-np.arange(len(x))))
  return X

# Example usage:
# Define a discrete-time signal
x = np.array([1, 2, 3, 4, 5])

# Define the range of z values for plotting
z_values = np.linspace(0.5, 2, 100) + 1j * np.linspace(-1, 1, 100)

# Calculate the Z-transform for each z value
X = np.array([z_transform(x, z) for z in z_values])

# Plot the magnitude and phase response
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.plot(np.real(z_values), np.abs(X))
plt.xlabel('Real(z)')
plt.ylabel('Magnitude')
plt.title('Magnitude Response')

plt.subplot(1, 2, 2)
plt.plot(np.real(z_values), np.angle(X))
plt.xlabel('Real(z)')
plt.ylabel('Phase (radians)')
plt.title('Phase Response')

plt.tight_layout()
plt.show()
