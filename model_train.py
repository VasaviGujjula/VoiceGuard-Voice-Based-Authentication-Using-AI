import numpy as np
from sklearn.svm import SVC
import joblib

X = np.load("X.npy")
y = np.load("y.npy")

print("Training on classes:", set(y))

model = SVC(kernel='linear', probability=True)
model.fit(X, y)

joblib.dump(model, "voice_model.pkl")
print("✅ Model trained and saved as voice_model.pkl")
