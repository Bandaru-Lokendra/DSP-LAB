import numpy as np
import matplotlib.pyplot as plt

def cyclic_delay(signal, delay):
  """
  Applies a cyclic delay to a given signal.

  Args:
    signal: The input signal as a NumPy array.
    delay: The number of positions to shift the signal cyclically.

  Returns:
    The cyclically delayed signal.
  """

  delayed_signal = np.roll(signal, delay)
  return delayed_signal
  #np.roll() is a NumPy function that shifts elements of an array along a specified axis. Think of it as rotating the array elements.

# Example usage:
# Generate a sample signal
signal = np.array([1, 2, 3, 4, 5, 6, 7, 8])

# Specify the desired delay
delay = 3

# Apply the cyclic delay
delayed_signal = cyclic_delay(signal, delay)

# Plot the original and delayed signals
plt.figure(figsize=(10, 6))
plt.plot(signal, label='Original Signal')
plt.plot(delayed_signal, label='Delayed Signal')
plt.xlabel('Sample Index')
plt.ylabel('Signal Value')
plt.legend()
plt.grid(True)
plt.show()
