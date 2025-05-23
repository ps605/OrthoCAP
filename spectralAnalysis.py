# spectralAnalysis.py

import numpy as np
import matplotlib.pyplot as plt

def spectralAnalysis(data, dt):
    "Run an FFT to complete a power density analysis for data. Useful when designing filters."
    
    length = len(data)
    data = data #- np.average(data) # Removes average as signal is likely not a zero average
    fhat = np.fft.fft(data,length)
    power_spectral_density = fhat*np.conj(fhat) / length
    freq = (1/(dt*length)) * np.arange(length)
    L = np.arange(1, np.floor(length/2), dtype=int)

    # Plot power spectrum
    plt.plot(freq[L], power_spectral_density[L]) 
    plt.show()

    return

# # Example data
# dt = 0.001 # Define time step
# t = np.arange(0,1,dt) # Define time vector
# data_in = np.sin(2*np.pi*50*t) + np.sin(2*np.pi*150*t) # Create sinusoidal 
# data_in = data_in + 2.5*np.random.randn(len(t)) # Add noise

# spectralAnalysis(data_in)
