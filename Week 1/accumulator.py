import numpy as np
import matplotlib.pyplot as plt

# Create a sample array of numbers
numbers = np.array([1, 2, 3, 4, 5])

# Calculate the cumulative sum using NumPy's cumsum function
cumulative_sum = np.cumsum(numbers)
#cumsum= a sequence of partial sums of a given sequence. It represents the sum of all elements up to a specific point in the sequence
# Plot the original numbers and their cumulative sum
plt.figure(figsize=(8, 5))
plt.plot(numbers, label='Original Numbers')
plt.plot(cumulative_sum, label='Cumulative Sum')
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Cumulative Sum')
plt.legend()
plt.grid(True)
plt.show()
