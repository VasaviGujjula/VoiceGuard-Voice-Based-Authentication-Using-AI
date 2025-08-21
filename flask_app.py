from flask import Flask, request, render_template, url_for
import joblib
import librosa
import numpy as np
from datetime import datetime
import os

app = Flask(__name__)
model = joblib.load("voice_model.pkl")

def extract_mfcc(filename):
    y, sr = librosa.load(filename, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfccs.T, axis=0)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    confidence = None
    audio_file = None
    if request.method == "POST":
        upload_path = os.path.join("static", "temp.wav")
        file = request.files["audio"]
        file.save(upload_path)
        features = extract_mfcc(upload_path).reshape(1, -1)
        prediction = model.predict(features)[0]
        proba = model.predict_proba(features)[0]
        confidence = round(np.max(proba) * 100, 2)
        result = prediction
        audio_file = upload_path

        # Log access
        with open("access_log.txt", "a") as f:
            f.write(f"{datetime.now()} - {prediction} ({confidence}%)\n")

    return render_template("index.html", result=result, confidence=confidence, audio_file=audio_file)

if __name__ == "__main__":
    app.run(debug=True)
