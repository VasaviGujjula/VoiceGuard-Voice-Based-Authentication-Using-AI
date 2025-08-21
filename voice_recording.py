import sounddevice as sd
from scipy.io.wavfile import write

def record_voice(filename="output.wav", duration=5, fs=44100):
    print("Recording...")
    voice = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    write(filename, fs, voice)
    print(f"Recording saved as {filename}")

record_voice()
