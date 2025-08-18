from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
import os
import soundfile as sf
import resampy
from python_speech_features import mfcc
import uuid

app = Flask(__name__)


CORS(app, resources={r"/": {"origins": ""}})


try:
    svm_model = joblib.load("svm_model.pkl")
    rf_model = joblib.load("rf_model.pkl")
except Exception as e:
    raise RuntimeError(f"Model loading failed: {e}")


def extract_mfcc(file_path):
    data, samplerate = sf.read(file_path)

    # Convert Stereo → Mono
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)

    # Normalize safely
    if np.max(np.abs(data)) > 0:
        data = data / np.max(np.abs(data))

    # Resample to 16kHz
    TARGET_SR = 16000
    if samplerate != TARGET_SR:
        data = resampy.resample(data, samplerate, TARGET_SR)
        samplerate = TARGET_SR

    # Extract MFCC
    nfft_val = 2048 if len(data) >= 2048 else 512
    mfcc_features = mfcc(signal=data, samplerate=samplerate, numcep=26, nfft=nfft_val)

    return np.mean(mfcc_features, axis=0).reshape(1, -1)


def get_alert_level(file_path):
    features = extract_mfcc(file_path)
    svm_pred = svm_model.predict(features)[0]
    rf_pred = rf_model.predict(features)[0]

    if svm_pred == 1 and rf_pred == 1:
        return "High Alert"
    elif svm_pred == 1 or rf_pred == 1:
        return "Moderate Alert"
    else:
        return "Normal"


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    f = request.files['file']
    file_path = f"temp_{uuid.uuid4().hex}{os.path.splitext(f.filename)[1]}"
    f.save(file_path)

    try:
        alert = get_alert_level(file_path)
        return jsonify({'alert_level': alert})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))