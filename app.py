import os
import subprocess
import sys

from flask import Flask, request, render_template, jsonify
import numpy as np
import joblib

app = Flask(__name__)

# Load the trained model, scaler, and PCA
def train_model_if_needed():
    if not os.path.exists('best_model.pkl') or not os.path.exists('scaler.pkl') or not os.path.exists('pca.pkl'):
        python_path = sys.executable
        subprocess.run([python_path, 'wine_quality.py'])

train_model_if_needed()

model = joblib.load('best_model.pkl')
scaler = joblib.load('scaler.pkl')
pca = joblib.load('pca.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from form
    data = [float(x) if x else 0 for x in request.form.values()]
    data_array = np.array([data])

    # Scale the data
    data_scaled = scaler.transform(data_array)

    # Apply PCA transformation
    data_pca = pca.transform(data_scaled)

    # Make prediction
    prediction = model.predict(data_pca)[0]

    result = 'Bună (calitate > 5)' if prediction == 1 else 'Rea (calitate <= 5)'

    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(debug=True)
