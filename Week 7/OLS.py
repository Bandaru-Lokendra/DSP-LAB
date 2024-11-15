import numpy as np

def dft(x):
    N = len(x)
    X = np.zeros(N, dtype=complex)
    for k in range(N):
        for n in range(N):
            X[k] += x[n] * np.exp(-2j * np.pi * k * n / N)
    return X

def idft(X):
    N = len(X)
    x = np.zeros(N, dtype=complex)
    for n in range(N):
        for k in range(N):
            x[n] += X[k] * np.exp(2j * np.pi * k * n / N)
    return x / N

def overlap_save(x, h):
    L = len(h)
    N = 2 * L  # Typically N = L + M - 1, but here we use 2L for simplicity
    M = len(x)
    
    # Zero-pad the filter to length N
    h_padded = np.zeros(N)
    h_padded[:L] = h
    
    # Compute the DFT of the filter
    H = dft(h_padded)
    
    # Initialize the output array
    y = np.zeros(M + L - 1)
    
    # Split the input into overlapping segments
    for i in range(0, M, L):
        # Extract the current segment and zero-pad
        x_segment = np.zeros(N)
        if i == 0:
            x_segment[:L] = x[i:i+L]
        else:
            x_segment[:L] = x[i-L:i]
        
        # Compute the DFT of the segment
        X = dft(x_segment)
        
        # Multiply in the frequency domain
        Y = X * H
        
        # Compute the inverse DFT
        y_segment = np.real(idft(Y))
        
        # Overlap-save: discard the first L-1 points and add the rest to the output
        y[i:i+L] += y_segment[L:]
    
    return y[:M]

# Example usage
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
h = np.array([1, 2, 1])
y = overlap_save(x, h)
print("Output:", y)
