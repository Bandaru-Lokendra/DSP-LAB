import numpy as np

# Initialize empty lists
x = []
h = []

# Get the size of the sequences from the user
l = int(input("Enter size of x[n]: "))
m = int(input("Enter size of h[n]: "))

# Get values for x[n]
print("Enter values for x[n]:")
for _ in range(l):
    value = int(input())  # Convert input to integer
    x.append(value)

# Get values for h[n]
print("Enter values for h[n]:")
for _ in range(m):
    value = int(input())  # Convert input to integer
    h.append(value)

def convolve_with_padding(x, h):
    # Convert lists to numpy arrays
    x = np.array(x)
    h = np.array(h)
    
    # Calculate the length of the output sequence
    N = len(x) + len(h) - 1
    
    # Zero pad x and h to the length of the output sequence
    x_padded = np.pad(x, (0, N - len(x)), 'constant')
    h_padded = np.pad(h, (0, N - len(h)), 'constant')
    
    print(f"Padded x[n]: {x_padded}")
    print(f"Padded h[n]: {h_padded}")
    
    # Perform convolution
    y = np.convolve(x_padded, h_padded)
    
    # Slice the output to keep only the first N elements
    y = y[:N]
    
    return y

# Perform convolution with zero padding
output = convolve_with_padding(x, h)

# Print the result
print(f"Output y[n]: {output}")
