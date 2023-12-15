import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Generate a sample channel impulse response (CIR) waveform
t = np.linspace(0, 10, 1000)  # Time vector
cir_waveform = np.exp(-0.5 * t) * np.cos(2 * np.pi * 1 * t)  # Example CIR waveform

# Define an exponential decay function for fitting
def exponential_decay(t, A, decay_constant):
    return A * np.exp(-decay_constant * t)

# Use curve_fit to fit the exponential decay function to the CIR waveform
popt, pcov = curve_fit(exponential_decay, t, cir_waveform)

# Extract the decay constant from the fitted parameters
decay_constant = popt[1]

# Plot the original CIR waveform and the fitted exponential decay
plt.plot(t, cir_waveform, label='Original CIR')
plt.plot(t, exponential_decay(t, *popt), 'r--', label='Exponential Decay Fit')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.legend()
plt.show()

# Print the decay constant
print(f"Decay Constant: {decay_constant}")