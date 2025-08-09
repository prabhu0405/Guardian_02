from flask import Flask, request, jsonify
import numpy as np
import joblib
import os
import soundfile as sf
import resampy
from python_speech_features import mfcc

app = Flask(__name__)

#Load pre-trained models
svm_model = joblib.load("svm_model.pkl")
rf_model = joblib.load("rf_model.pkl")

#Feature Extraction
def extract_mfcc(file_path):
    # Load audio using soundfile
    data, samplerate = sf.read(file_path)

    # Convert stereo to mono if needed
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)

    # Normalize audio
    if data.dtype != np.float32:
        data = data / np.max(np.abs(data))

    # Resample to match training data sample rate
    TARGET_SR = 16000
    if samplerate != TARGET_SR:
        data = resampy.resample(data, samplerate, TARGET_SR)
        samplerate = TARGET_SR

    # Extract MFCC features
    mfcc_features = mfcc(
        signal=data,
        samplerate=samplerate,
        numcep=26,   # match training
        nfft=2048
    )

    # Take mean across time frames
    mfccs_mean = np.mean(mfcc_features, axis=0)

    return mfccs_mean.reshape(1, -1)

# -------------------------------
# 🚨 Alert Prediction
# -------------------------------
def get_alert_level(file_path):
    features = extract_mfcc(file_path)

    # Predict using SVM and RF
    svm_pred = svm_model.predict(features)[0]
    rf_pred = rf_model.predict(features)[0]

    # Decide alert level
    if svm_pred == 1 and rf_pred == 1:
        return "High Alert"
    elif svm_pred == 1 or rf_pred == 1:
        return "Moderate Alert"
    else:
        return "Normal"

#API Endpoint
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    f = request.files['file']
    file_path = 'temp_audio' + os.path.splitext(f.filename)[1]  # keep extension
    f.save(file_path)

    try:
        alert = get_alert_level(file_path)
        return jsonify({'alert_level': alert})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

# Run App
if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)