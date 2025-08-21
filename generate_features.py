import os
import librosa
import numpy as np

def extract_features_from_folder(folder):
    features = []
    labels = []
    print(f"Looking into: {folder}")
    
    for speaker in os.listdir(folder):
        speaker_folder = os.path.join(folder, speaker)
        if not os.path.isdir(speaker_folder):
            continue
        print(f"Reading speaker: {speaker}")
        
        for file in os.listdir(speaker_folder):
            if file.endswith(".wav"):
                filepath = os.path.join(speaker_folder, file)
                print(f"Extracting from: {filepath}")
                y, sr = librosa.load(filepath, sr=None)
                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                mfcc_mean = np.mean(mfcc.T, axis=0)
                features.append(mfcc_mean)
                labels.append(speaker)
    
    print(f"Speakers found: {set(labels)}")
    print(f"Total samples: {len(labels)}")
    return np.array(features), np.array(labels)

X, y = extract_features_from_folder("VoiceSamples")
np.save("X.npy", X)
np.save("y.npy", y)
print("✅ Feature extraction complete. Saved X.npy and y.npy")
