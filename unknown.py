import sounddevice as sd
from scipy.io.wavfile import write
import os

def record_sample(speaker="unknown", num_samples=3, duration=3, fs=16000):
    folder = f"VoiceSamples/{speaker}"
    os.makedirs(folder, exist_ok=True)

    for i in range(1, num_samples + 1):
        filename = f"{folder}/{speaker}{i}.wav"
        print(f"\nRecording sample {i} for '{speaker}'...")
        audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
        sd.wait()
        write(filename, fs, audio)
        print(f"Saved: {filename}")

# Call the function to record unknown speaker samples
record_sample()
