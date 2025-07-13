from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras import layers, models
import os

# ───── Config ─────
IMG_SIZE = (224, 224)
NUM_CLASSES = 4
CLASS_NAMES = ['bolt', 'locatingpin', 'nut', 'washer']
WEIGHTS_PATH = "effnet_weights_final.h5"

# ───── Flask App Setup ─────
app = Flask(__name__, template_folder="templates", static_folder="static")
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10 MB limit
CORS(app)

# ───── Model Initialization ─────
print("🚀 Initializing EfficientNetB0 model...")
base_model = EfficientNetB0(include_top=False, weights="imagenet", input_shape=(*IMG_SIZE, 3))
base_model.trainable = False

inputs = layers.Input(shape=(*IMG_SIZE, 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
model = models.Model(inputs, outputs)

# Load weights
if os.path.exists(WEIGHTS_PATH):
    try:
        model.load_weights(WEIGHTS_PATH)
        print("✅ Model weights loaded successfully.")
    except Exception as e:
        print(f"❌ Failed to load weights: {e}")
else:
    print("🚨 Model weights file not found! Please upload 'effnet_weights_final.h5'.")

# ───── Routes ─────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        print("❌ No file uploaded")
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    if not file.filename.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
        print("❌ Unsupported file format:", file.filename)
        return jsonify({"error": "Unsupported file format"}), 400

    try:
        # Load image
        img = Image.open(file).convert("RGB").resize(IMG_SIZE)
        arr = np.array(img, dtype=np.float32)
        arr = preprocess_input(arr)
        arr = np.expand_dims(arr, axis=0)

        print(f"📐 Image shape: {arr.shape}")

        # Predict
        prediction = model.predict(arr)[0]
        print(f"🔢 Raw prediction vector: {prediction}")

        if np.allclose(prediction, prediction[0]):
            print("⚠️ Flat prediction vector. Model may not be trained or weights not loaded properly.")
            return jsonify({"error": "Model is not properly initialized. Try re-uploading weights."}), 500

        class_id = int(np.argmax(prediction))
        confidence = float(prediction[class_id]) * 100
        result = {
            "class": CLASS_NAMES[class_id],
            "confidence": round(confidence, 2)
        }

        print(f"✅ Prediction result: {result}")
        return jsonify(result)

    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        return jsonify({"error": "Prediction failed"}), 500

# ───── Run Server ─────
if __name__ == "__main__":
    print("🟢 Flask API running on http://localhost:5000")
    app.run(debug=True)
