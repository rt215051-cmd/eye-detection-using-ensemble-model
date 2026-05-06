import os
import io
import json
import sqlite3
from datetime import datetime
import numpy as np
from PIL import Image

# -------------------------------------
# FIX for Keras 3 (quantization_config error)
# -------------------------------------
import tensorflow as tf
from tensorflow import keras

original_dense_init = keras.layers.Dense.__init__

def custom_dense_init(self, *args, **kwargs):
    kwargs.pop('quantization_config', None)  # remove problematic key
    original_dense_init(self, *args, **kwargs)

keras.layers.Dense.__init__ = custom_dense_init

# -------------------------------------
# Now import models AFTER patch
# -------------------------------------
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import DenseNet121, ResNet50, MobileNetV2
from tensorflow.keras.applications.densenet import preprocess_input as preprocess_densenet
from tensorflow.keras.applications.resnet import preprocess_input as preprocess_resnet
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as preprocess_mobilenet

from flask import Flask, request, jsonify, render_template

# Reduce TF warnings
tf.get_logger().setLevel('ERROR')

app = Flask(__name__)

# -----------------------------
# Database Setup
# -----------------------------
def get_db():
    conn = sqlite3.connect('eyeguard.db')
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            prediction TEXT NOT NULL,
            confidence REAL NOT NULL,
            all_results TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# -----------------------------
# Class Labels
# -----------------------------
CLASS_NAMES = ["amd", "cataract", "diabetic_retinopathy", "glaucoma", "normal"]

CLASS_LABELS = {
    "amd": "Age-related Macular Degeneration",
    "cataract": "Cataract",
    "diabetic_retinopathy": "Diabetic Retinopathy",
    "glaucoma": "Glaucoma",
    "normal": "Normal"
}

# -----------------------------
# Load Models
# -----------------------------
print("Loading models...")

# Base models
densenet_base = DenseNet121(weights="imagenet", include_top=False, pooling="avg", input_shape=(224,224,3))
resnet_base = ResNet50(weights="imagenet", include_top=False, pooling="avg", input_shape=(224,224,3))
mobilenet_base = MobileNetV2(weights="imagenet", include_top=False, pooling="avg", input_shape=(224,224,3))

# Classifier models
model_vgg = load_model("eye_disease_classifier_vgg16.keras", compile=False)
model_resnet = load_model("eye_disease_classifier_Resnet50.h5", compile=False)
model_mob = load_model("eye_disease_classifier_mobilenetv2 (1).h5", compile=False)
model_dense = load_model("eye_disease_classifier_Densenet121.h5", compile=False)
model_inception = load_model("eye_disease_classifier_inception.h5", compile=False)

print("All models loaded successfully.")

# -----------------------------
# Prediction Function
# -----------------------------
def predict_ensemble(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    img = image.resize((224, 224))
    arr = np.expand_dims(np.array(img, dtype=np.float32), axis=0)

    # -------- DenseNet (1024 features) --------
    arr_den = preprocess_densenet(arr.copy())
    feat_den = densenet_base.predict(arr_den, verbose=0)

    pred1 = model_vgg.predict(feat_den, verbose=0)
    pred4 = model_dense.predict(feat_den, verbose=0)
    pred5 = model_inception.predict(feat_den, verbose=0)

    # -------- ResNet (2048 features) --------
    arr_res = preprocess_resnet(arr.copy())
    feat_res = resnet_base.predict(arr_res, verbose=0)
    pred2 = model_resnet.predict(feat_res, verbose=0)

    # -------- MobileNet (1280 features) --------
    arr_mob = preprocess_mobilenet(arr.copy())
    feat_mob = mobilenet_base.predict(arr_mob, verbose=0)
    pred3 = model_mob.predict(feat_mob, verbose=0)

    # -------- Ensemble --------
    ensemble_pred = (pred1 + pred2 + pred3 + pred4 + pred5) / 5

    return ensemble_pred[0]

# -----------------------------
# Routes
# -----------------------------
@app.route('/')
def home():
    return render_template('landing.html')

@app.route('/dashboard')
def dashboard():
    return render_template('index.html')

@app.route('/history')
def history():
    conn = get_db()
    rows = conn.execute('SELECT * FROM predictions ORDER BY created_at DESC').fetchall()
    conn.close()
    records = []
    for row in rows:
        records.append({
            'id': row['id'],
            'filename': row['filename'],
            'prediction': row['prediction'],
            'confidence': round(row['confidence'], 2),
            'all_results': json.loads(row['all_results']),
            'created_at': row['created_at']
        })
    return render_template('history.html', records=records)

@app.route('/history/delete/<int:record_id>', methods=['POST'])
def delete_record(record_id):
    conn = get_db()
    conn.execute('DELETE FROM predictions WHERE id = ?', (record_id,))
    conn.commit()
    conn.close()
    return jsonify({'success': True})

@app.route('/predict', methods=['POST'])
def predict():

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    filename = file.filename

    if filename == '':
        return jsonify({'error': 'Empty file'}), 400

    try:
        probs = predict_ensemble(file.read())

        sorted_indices = np.argsort(probs)[::-1]

        results = [
            {
                "disease": CLASS_LABELS[CLASS_NAMES[i]],
                "probability": float(probs[i]) * 100
            }
            for i in sorted_indices
        ]

        # Save to database
        conn = get_db()
        conn.execute(
            'INSERT INTO predictions (filename, prediction, confidence, all_results) VALUES (?, ?, ?, ?)',
            (filename, results[0]['disease'], results[0]['probability'], json.dumps(results))
        )
        conn.commit()
        conn.close()

        return jsonify({
            "success": True,
            "prediction": results[0]["disease"],
            "confidence": results[0]["probability"],
            "all_results": results
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -----------------------------
# Run Server
# -----------------------------
if __name__ == '__main__':
    print("Server running at http://127.0.0.1:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)