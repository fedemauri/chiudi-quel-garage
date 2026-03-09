#!/usr/bin/env python3
"""Train a binary classifier (open/closed) for garage door detection.

Uses MobileNetV2 transfer learning with aggressive data augmentation
to handle the small dataset (~140 images).

Usage:
    python scripts/train_model.py --train
    python scripts/train_model.py --predict <image_path>
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np

# ── Paths ────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parent.parent / "data/images"
MODEL_DIR = Path(__file__).resolve().parent.parent / "model"
CLASSES = ("closed", "open")  # index 0=closed, 1=open
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

IMG_SIZE = (224, 224)  # MobileNetV2 input size
BATCH_SIZE = 8
EPOCHS = 30
LEARNING_RATE = 0.0005
FINE_TUNE_EPOCHS = 20
FINE_TUNE_LR = 0.00005


def load_dataset():
    """Load all labeled images from data/images/{open,closed}/ directories."""
    from PIL import Image

    images = []
    labels = []
    filenames = []

    for class_idx, class_name in enumerate(CLASSES):
        class_dir = DATA_DIR / class_name
        if not class_dir.exists():
            print(f"  Warning: {class_dir} not found, skipping")
            continue
        for path in sorted(class_dir.iterdir()):
            if path.suffix.lower() not in IMAGE_EXTS:
                continue
            img = Image.open(path).convert("RGB").resize(IMG_SIZE)
            img_array = np.array(img, dtype=np.float32) / 255.0
            images.append(img_array)
            labels.append(class_idx)
            filenames.append(path.name)

    if not images:
        print(f"No images found in {DATA_DIR}/open/ and {DATA_DIR}/closed/")
        sys.exit(1)

    return np.array(images), np.array(labels), filenames


def train_model():
    """Train MobileNetV2 with transfer learning."""
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    print(f"TensorFlow version: {tf.__version__}")

    # ── Load all images ───────────────────────────────────────────────
    X, y, filenames = load_dataset()
    n_open = int(y.sum())
    n_closed = len(y) - n_open
    print(f"Dataset: {len(y)} images ({n_open} open, {n_closed} closed)")

    # ── Stratified split: ensure diverse val set ──────────────────────
    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)

    # Separate indices by class and shuffle
    closed_idx = [i for i, l in enumerate(y) if l == 0]
    open_idx = [i for i, l in enumerate(y) if l == 1]
    random.shuffle(closed_idx)
    random.shuffle(open_idx)

    # Take ~20% from each class for validation
    n_val_closed = max(2, len(closed_idx) // 5)
    n_val_open = max(4, len(open_idx) // 5)
    val_idx = closed_idx[:n_val_closed] + open_idx[:n_val_open]
    train_idx = closed_idx[n_val_closed:] + open_idx[n_val_open:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    val_filenames = [filenames[i] for i in val_idx]

    print(f"Train: {len(y_train)} ({int(y_train.sum())} open, {len(y_train) - int(y_train.sum())} closed)")
    print(f"Val: {len(y_val)} ({int(y_val.sum())} open, {len(y_val) - int(y_val.sum())} closed)")
    print(f"Val images: {val_filenames}")

    # ── Data augmentation via tf.data ─────────────────────────────────
    def augment(image, label):
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, 0.3)
        image = tf.image.random_contrast(image, 0.7, 1.3)
        image = tf.image.random_saturation(image, 0.7, 1.3)
        image = tf.clip_by_value(image, 0.0, 1.0)
        return image, label

    train_ds = (
        tf.data.Dataset.from_tensor_slices((X_train, y_train))
        .shuffle(len(X_train), seed=42)
        .map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    val_ds = (
        tf.data.Dataset.from_tensor_slices((X_val, y_val))
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    # ── Class weights ─────────────────────────────────────────────────
    n_total = len(y_train)
    n_closed_train = int((y_train == 0).sum())
    n_open_train = int((y_train == 1).sum())
    class_weight = {
        0: n_total / (2 * n_closed_train) if n_closed_train > 0 else 1.0,
        1: n_total / (2 * n_open_train) if n_open_train > 0 else 1.0,
    }
    print(f"Class weights: {class_weight}")

    # ── Base model: MobileNetV2 frozen ────────────────────────────────
    base_model = keras.applications.MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    # ── Classification head ───────────────────────────────────────────
    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()

    # ── Phase 1: Train head only ──────────────────────────────────────
    print("\n--- Phase 1: Training classification head ---")
    history1 = model.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=val_ds,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=8, restore_best_weights=True,
            ),
        ],
        class_weight=class_weight,
    )

    # ── Phase 2: Fine-tune last layers of MobileNetV2 ─────────────────
    print("\n--- Phase 2: Fine-tuning last 20 layers ---")
    base_model.trainable = True
    for layer in base_model.layers[:-20]:
        layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=FINE_TUNE_LR),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    history2 = model.fit(
        train_ds,
        epochs=FINE_TUNE_EPOCHS,
        validation_data=val_ds,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=8, restore_best_weights=True,
            ),
        ],
        class_weight=class_weight,
    )

    # ── Evaluate ──────────────────────────────────────────────────────
    print("\n--- Validation Results ---")
    val_loss, val_acc = model.evaluate(val_ds)
    print(f"Validation accuracy: {val_acc:.2%}")

    predictions = model.predict(val_ds)
    errors = []
    for i, (fname, pred) in enumerate(zip(val_filenames, predictions)):
        pred_label = "open" if pred[0] > 0.5 else "closed"
        true_label = "open" if y_val[i] == 1 else "closed"
        conf = pred[0] if pred_label == "open" else 1 - pred[0]
        status = "OK" if pred_label == true_label else "WRONG"
        if status == "WRONG":
            errors.append(fname)
        print(f"  {status:5s} {fname:25s} → {pred_label:6s} (conf={conf:.2f}, true={true_label})")

    if errors:
        print(f"\nErrors ({len(errors)}/{len(val_filenames)}):")
        for e in errors:
            print(f"  - {e}")
    else:
        print(f"\nAll {len(val_filenames)} validation images correct!")

    # ── Full dataset evaluation ───────────────────────────────────────
    print("\n--- Full Dataset Evaluation ---")
    all_preds = model.predict(X, batch_size=BATCH_SIZE)
    full_errors = []
    for i, (fname, pred) in enumerate(zip(filenames, all_preds)):
        pred_label = "open" if pred[0] > 0.5 else "closed"
        true_label = "open" if y[i] == 1 else "closed"
        if pred_label != true_label:
            conf = pred[0] if pred_label == "open" else 1 - pred[0]
            full_errors.append((fname, pred_label, true_label, conf))

    if full_errors:
        print(f"Errors on full dataset ({len(full_errors)}/{len(filenames)}):")
        for fname, pred_l, true_l, conf in full_errors:
            print(f"  {fname:25s} → {pred_l:6s} (conf={conf:.2f}, true={true_l})")
    else:
        print(f"All {len(filenames)} images classified correctly!")

    full_acc = 1 - len(full_errors) / len(filenames)
    print(f"Full dataset accuracy: {full_acc:.2%}")

    # ── Save ──────────────────────────────────────────────────────────
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    keras_path = MODEL_DIR / "garage_classifier.keras"
    model.save(str(keras_path))
    print(f"\nKeras model saved: {keras_path}")

    return model


def export_tflite(model=None):
    """Export trained model to TFLite format via SavedModel (Keras 3 compatible)."""
    import tempfile

    import tensorflow as tf

    keras_path = MODEL_DIR / "garage_classifier.keras"
    tflite_path = MODEL_DIR / "garage_classifier.tflite"

    if model is None:
        print(f"Loading model from {keras_path}")
        model = tf.keras.models.load_model(str(keras_path))

    # Keras 3 + TF 2.16: export to SavedModel first, then convert
    with tempfile.TemporaryDirectory() as tmpdir:
        saved_model_path = str(Path(tmpdir) / "saved_model")
        model.export(saved_model_path)
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        tflite_model = converter.convert()

    tflite_path.parent.mkdir(parents=True, exist_ok=True)
    tflite_path.write_bytes(tflite_model)

    size_mb = len(tflite_model) / (1024 * 1024)
    print(f"TFLite model saved: {tflite_path} ({size_mb:.1f} MB)")

    # Verify
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()
    out = interpreter.get_output_details()
    print(f"Input: {inp[0]['shape']}, Output: {out[0]['shape']}")


def predict_image(image_path: str):
    """Run inference on a single image using TFLite model."""
    import tensorflow as tf
    from PIL import Image

    tflite_path = MODEL_DIR / "garage_classifier.tflite"
    if not tflite_path.exists():
        print(f"Model not found: {tflite_path}. Run --train first.")
        sys.exit(1)

    img = Image.open(image_path).convert("RGB").resize(IMG_SIZE)
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]["index"], img_array)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]["index"])

    prob_open = float(output[0][0])
    label = "open" if prob_open > 0.5 else "closed"
    confidence = prob_open if label == "open" else 1 - prob_open

    print(f"Prediction: {label.upper()} (confidence={confidence:.2%})")
    print(f"Raw output: {prob_open:.4f} (>0.5 = open, <0.5 = closed)")


def main():
    parser = argparse.ArgumentParser(description="Train garage door classifier")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--export", action="store_true", help="Export to TFLite")
    parser.add_argument("--predict", type=str, help="Predict on a single image")
    args = parser.parse_args()

    if args.train:
        model = train_model()
        export_tflite(model)
    elif args.export:
        export_tflite()
    elif args.predict:
        predict_image(args.predict)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
