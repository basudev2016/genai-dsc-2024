import numpy as np
from scipy.io.wavfile import write

# Parameters for the sound wave
sample_rate = 44100  # Samples per second
duration = 2.0       # Duration in seconds
frequency = 440.0    # Frequency in Hz (A4 note)

# Generate the time points
t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

# Generate the sound wave (sine wave)
audio_data = 0.5 * np.sin(2 * np.pi * frequency * t)

# Save the sound to a file
write("simple_tone.wav", sample_rate, audio_data.astype(np.float32))

print("Generated a simple tone saved as 'simple_tone.wav'")
