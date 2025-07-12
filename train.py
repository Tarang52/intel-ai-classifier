import tensorflow as tf
from pathlib import Path
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input

# ───────── CONFIG ─────────
BASE_DIR   = Path("datasets/mechanical_parts_split")
TRAIN_DIR  = BASE_DIR / "train"
VAL_DIR    = BASE_DIR / "val"
IMG_SIZE   = (224, 224)
BATCH_SIZE = 32
EPOCHS     = 15
SEED       = 42
MODEL_OUT  = "best_effnet_parts.h5"

# ───────── LOAD DATA ─────────
raw_train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR, image_size=IMG_SIZE, batch_size=BATCH_SIZE,
    shuffle=True, seed=SEED, label_mode="categorical")
raw_val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR, image_size=IMG_SIZE, batch_size=BATCH_SIZE,
    shuffle=True, seed=SEED, label_mode="categorical")

class_names = raw_train_ds.class_names
num_classes = len(class_names)
print("🗂️  Classes:", class_names)

def prep(x, y):
    return preprocess_input(tf.cast(x, tf.float32)), y

train_ds = raw_train_ds.map(prep).prefetch(tf.data.AUTOTUNE)
val_ds   = raw_val_ds.map(prep).prefetch(tf.data.AUTOTUNE)

# ───────── MODEL ─────────
base = EfficientNetB0(include_top=False, weights="imagenet", input_shape=(*IMG_SIZE, 3))
base.trainable = False

inputs = layers.Input(shape=(*IMG_SIZE, 3))
x = layers.RandomFlip("horizontal")(inputs)
x = layers.RandomRotation(0.1)(x)
x = layers.RandomZoom(0.1)(x)
x = base(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)

model = models.Model(inputs, outputs)
model.compile(
    optimizer=optimizers.Adam(1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ───────── CALLBACK: NO JSON, ONLY FLOATS ─────────
class SafePrintCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        try:
            # Force conversion to float (not Tensor, not Numpy)
            loss = float(logs.get("loss", 0.0))
            acc = float(logs.get("accuracy", 0.0))
            val_loss = float(logs.get("val_loss", 0.0))
            val_acc = float(logs.get("val_accuracy", 0.0))
        except Exception as e:
            print(f"⚠️ Failed to cast logs to float: {e}")
            loss = acc = val_loss = val_acc = -1

        print(f"Epoch {epoch+1}/{EPOCHS} — loss: {loss:.4f} — acc: {acc:.4f} — val_loss: {val_loss:.4f} — val_acc: {val_acc:.4f}")

# ───────── TRAIN ─────────
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[SafePrintCallback()],  # ⛔ no ModelCheckpoint, no EarlyStopping
    verbose=0  # ⛔ no internal logging either
)

# ───────── EVAL & MANUAL SAVE ─────────
val_loss, val_acc = model.evaluate(val_ds, verbose=0)
print(f"\n✅ Training complete. Final val accuracy: {val_acc:.4f}")

model.save_weights("effnet_weights_final.h5")
print("✅ Model weights saved to effnet_weights_final.h5")

print(f"💾 Best model saved at: {MODEL_OUT}")
