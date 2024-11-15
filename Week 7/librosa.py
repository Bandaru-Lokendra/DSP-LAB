import numpy as np
import matplotlib.pyplot as plt
import librosa

filename = '/home/lokendra/n.wave'
sampling_rate = 16000

# Check if librosa.load is available
if hasattr(librosa, 'load'):
    y, sr = librosa.load(filename, sr=sampling_rate)
else:
    raise ImportError("librosa.load function is not available. Please check your librosa installation.")

duration = 0.1  
sample_length = int(duration * sampling_rate)
sample = y[:sample_length]

t = np.arange(sample_length) / sampling_rate

def dft(x):
    N = len(x)
    X = np.zeros(N, dtype=complex)
    for k in range(N):
        X[k] = np.sum(x * np.exp(-2j * np.pi * k * np.arange(N) / N))
    return X

X = dft(sample)

new_sampling_rate = 8000
y_resampled = librosa.resample(y, orig_sr=sampling_rate, target_sr=new_sampling_rate)

sample_length_resampled = int(duration * new_sampling_rate)
sample_resampled = y_resampled[:sample_length_resampled]

X_resampled = dft(sample_resampled)

zero_padding = int(sample_length * 1.5)
sample_resampled_padded = np.pad(sample_resampled, (0, zero_padding - len(sample_resampled)), 'constant')

def convolve(x, h):
    N = len(x) + len(h) - 1
    X = np.fft.fft(x, N)
    H = np.fft.fft(h, N)
    Y = np.fft.ifft(X * H)
    return Y

result = convolve(sample_resampled_padded, sample)

plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(np.abs(X))
plt.title('Magnitude Spectrum (16kHz)')
plt.xlabel('Frequency Bin')
plt.ylabel('Magnitude')
plt.grid()

plt.subplot(3, 1, 2)
plt.plot(np.abs(X_resampled))
plt.title('Magnitude Spectrum (8kHz)')
plt.xlabel('Frequency Bin')
plt.ylabel('Magnitude')
plt.grid()

plt.subplot(3, 1, 3)
plt.plot(result.real) 
plt.title('Convolution Result')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.grid()

plt.tight_layout()
plt.show()

