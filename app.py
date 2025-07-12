from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras import layers, models

# ───── Config ─────
IMG_SIZE = (224, 224)
NUM_CLASSES = 4
CLASS_NAMES = ['bolt', 'locatingpin', 'nut', 'washer']
WEIGHTS_PATH = "effnet_weights_final.h5"

# ───── Load Model ─────
base = EfficientNetB0(include_top=False, weights="imagenet", input_shape=(*IMG_SIZE, 3))
base.trainable = False
inputs = layers.Input(shape=(*IMG_SIZE, 3))
x = base(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
model = models.Model(inputs, outputs)
model.load_weights(WEIGHTS_PATH)

# ───── Flask Setup ─────
app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    img = Image.open(file).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    arr = preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)

    pred = model.predict(arr)[0]
    class_id = int(np.argmax(pred))
    confidence = float(pred[class_id]) * 100

    return jsonify({
        "class": CLASS_NAMES[class_id],
        "confidence": round(confidence, 2)
    })

if __name__ == '__main__':
    app.run(debug=True)
