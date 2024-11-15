import numpy as np
import matplotlib.pyplot as plt

# Function to compute DTFT and plot the spectra
def plot_dtft_and_spectra(signal):
    # Define the DTFT
    def dtft(x, omega):
        N = len(x)
        X = np.zeros(len(omega), dtype=complex)
        for k, w in enumerate(omega):
            X[k] = np.sum(x * np.exp(-1j * w * np.arange(N)))
        return X

    # Generate frequency vector
    omega = np.linspace(-np.pi, np.pi, 1024)  # Frequency range from -π to π
    X = dtft(signal, omega)
    
    # Compute magnitude and phase spectra
    magnitude_spectrum = np.abs(X)
    phase_spectrum = np.angle(X)
    
    # Plot the input signal, DTFT magnitude, and phase spectra
    plt.figure(figsize=(12, 12))
    
    # Plot the input signal
    plt.subplot(3, 1, 1)
    n = np.arange(len(signal))
    plt.plot(n, signal, marker='o', linestyle='-')
    plt.title('Input Discrete-Time Signal')
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    
    # Plot the magnitude spectrum
    plt.subplot(3, 1, 2)
    plt.plot(omega, magnitude_spectrum)
    plt.title('Magnitude Spectrum of DTFT')
    plt.xlabel('Frequency (rad/sample)')
    plt.ylabel('Magnitude')
    
    # Plot the phase spectrum
    plt.subplot(3, 1, 3)
    plt.plot(omega, phase_spectrum)
    plt.title('Phase Spectrum of DTFT')
    plt.xlabel('Frequency (rad/sample)')
    plt.ylabel('Phase (radians)')
    
    plt.tight_layout()
    plt.show()

# Define a discrete-time signal
signal = np.array([1, 0, -1, 0, 1, 0, -1, 0])

# Call the function with the predefined signal
plot_dtft_and_spectra(signal)

