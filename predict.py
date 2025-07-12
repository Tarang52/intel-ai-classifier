import tensorflow as tf
import numpy as np
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras import layers, models
from PIL import Image
import matplotlib.pyplot as plt

# ───────── CONFIG ─────────
IMG_SIZE = (224, 224)
NUM_CLASSES = 4
CLASS_NAMES = ['bolt', 'locatingpin', 'nut', 'washer']
WEIGHTS_PATH = "effnet_weights_final.h5"
IMAGE_PATH = "C:\\Users\\chira\\Desktop\\INTEL AI\\datasets\\mechanical_parts_split\\val\\washer\\WM0804_3_10.png"  # 👈 Replace this with your image filename

# ───────── MODEL STRUCTURE (Matches training!) ─────────
base = EfficientNetB0(include_top=False, weights="imagenet", input_shape=(*IMG_SIZE, 3))
base.trainable = False

inputs = layers.Input(shape=(*IMG_SIZE, 3))
x = layers.RandomFlip("horizontal")(inputs)
x = layers.RandomRotation(0.1)(x)
x = layers.RandomZoom(0.1)(x)
x = base(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

model = models.Model(inputs, outputs)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.load_weights(WEIGHTS_PATH)
print("✅ Model loaded successfully.")

# ───────── LOAD & PREDICT ─────────
def load_image(path):
    img = Image.open(path).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img)
    arr = preprocess_input(arr.astype(np.float32))
    return np.expand_dims(arr, axis=0), img

img_tensor, img_display = load_image(IMAGE_PATH)
pred = model.predict(img_tensor, verbose=0)[0]
class_idx = np.argmax(pred)
confidence = pred[class_idx] * 100

# ───────── OUTPUT ─────────
print(f"\n🧠 Prediction: {CLASS_NAMES[class_idx]} ({confidence:.2f}%)")

plt.imshow(img_display)
plt.axis("off")
plt.title(f"Prediction: {CLASS_NAMES[class_idx]} ({confidence:.2f}%)")
plt.show()
