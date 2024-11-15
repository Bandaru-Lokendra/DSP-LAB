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

# Example usage
x = np.array([1, 2, 3,0,0,0])
h = np.array([1,2,3,0,0,0])

y = circular_convolution(x, h)
signal=[1,2,3]
kernel=[1,2,3]


convolved_length=len(signal)+len(kernel)-1
convolved_signal=np.zeros(convolved_length)


for i in range(convolved_length):
	for j in range(len(kernel)):
		if i-j>=0 and i-j<len(signal):
			convolved_signal[i]+=signal[i-j]*kernel[j]
print(convolved_signal) 
print(y)


