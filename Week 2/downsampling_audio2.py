import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

# Load audio file
fs, audio = wavfile.read('audio_file.wav')

# Define downsampling factor
downsampling_factor = 2

# Downsample audio signal
fs_downsampled = fs // downsampling_factor
audio_downsampled = audio[::downsampling_factor]

# Plot original and downsampled audio signals
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(audio)
plt.title('Original Audio Signal')
plt.xlabel('Time Index')
plt.ylabel('Amplitude')

plt.subplot(2, 1, 2)
plt.plot(audio_downsampled)
plt.title('Downsampled Audio Signal')
plt.xlabel('Time Index')
plt.ylabel('Amplitude')

plt.tight_layout()
plt.show()

# Plot frequency spectra
freq = np.fft.fftfreq(len(audio), d=1/fs)
freq_downsampled = np.fft.fftfreq(len(audio_downsampled), d=1/fs_downsampled)

plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.stem(freq, np.abs(np.fft.fft(audio)))
plt.title('Original Frequency Spectrum')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')

plt.subplot(2, 1, 2)
plt.stem(freq_downsampled, np.abs(np.fft.fft(audio_downsampled)))
plt.title('Downsampled Frequency Spectrum')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')

plt.tight_layout()
plt.show()

# Save downsampled audio file
wavfile.write('downsampled_audio.wav', fs_downsampled, audio_downsampled)

'''
This code:

1. Loads an audio file using wavfile.read.
2. Defines a downsampling factor.
3. Downsamples the audio signal using array slicing (audio[::downsampling_factor]).
4. Plots the original and downsampled audio signals using Matplotlib.
5. Plots the frequency spectra of the original and downsampled audio signals using np.fft.fft.
6. Saves the downsampled audio file using wavfile.write.

Note:

- The downsampling process reduces the sampling rate of the audio signal.
- The wavfile module from SciPy is used for reading and writing WAV files.
- The np.fft module is used for computing the Fast Fourier Transform (FFT).

Replace 'audio_file.wav' with the path to your audio file.

Make sure to install required libraries using pip install numpy matplotlib scipy.'''
