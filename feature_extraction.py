import librosa
import numpy as np

def extract_mfcc(filename):
    y, sr = librosa.load(filename, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfccs.T, axis=0)

features = extract_mfcc("VoiceSamples/vasavi/output1.wav")
print("MFCC features:", features)
