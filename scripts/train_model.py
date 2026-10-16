#!/usr/bin/env python3
"""Train the binary garage door classifier (open/closed) from the reviewed dataset.

Dataset: data/telegram/manifest.csv built by import_telegram_export.py and
labeled with review_dataset.py (column final_label). The task is binary: a door
open even a few cm is "open". Photos excluded during review are skipped.

Method (designed for a CPU-only machine):
  * MobileNetV2 (ImageNet) transfer learning. Inputs stay in [0, 1] like the
    production TFLiteAnalyzer; a Rescaling layer maps them to the [-1, 1]
    range MobileNetV2 was trained on.
  * The backbone up to SPLIT_LAYER is frozen, so its activations are computed
    once per photo (plus N_AUG augmented copies) and cached. Training the last
    block + head on cached activations takes minutes instead of hours.
  * BatchNorm layers stay frozen (inference mode) while fine-tuning.
  * Photos whose door is only slightly open (review label "partial", or a
    reviewer override) get a higher sample weight: they are the risky cases.

Usage:
    python scripts/train_model.py --evaluate   # temporal split, new vs current model on test
    python scripts/train_model.py --train      # final model on all data + TFLite export
    python scripts/train_model.py --export     # re-export .keras to .tflite
    python scripts/train_model.py --predict /path/to/image.jpg
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data/telegram"
MANIFEST = DATA_DIR / "manifest.csv"
CACHE_DIR = DATA_DIR / "cache"
MODEL_DIR = ROOT / "model"
KERAS_PATH = MODEL_DIR / "garage_classifier.keras"
TFLITE_PATH = MODEL_DIR / "garage_classifier.tflite"

IMG_SIZE = (224, 224)  # MobileNetV2 input size, same as TFLiteAnalyzer
SPLIT_LAYER = "block_15_add"  # frozen below, trainable above (block 16 + Conv_1)
N_AUG = 5  # augmented copies per training photo (plus the original)
BATCH_SIZE = 64
HEAD_EPOCHS, HEAD_LR = 40, 1e-3
FINE_TUNE_EPOCHS, FINE_TUNE_LR = 30, 5e-5
RISKY_WEIGHT = 2.0  # sample weight for slightly-open doors
PROD_CONFIDENCE = 0.7  # GM_CONFIDENCE_THRESHOLD: below it production keeps the old state
SEED = 42

# Default temporal split for --evaluate (ISO dates, by photo timestamp)
VAL_FROM = "2026-08-01"
TEST_FROM = "2026-09-01"
AUG_VERSION = "v1"  # bump when augment() changes to invalidate the cache


@dataclass
class Sample:
    file: str
    y: int  # 1 = open, 0 = closed
    date: str
    risky: bool


# ── Data ─────────────────────────────────────────────────────────────


def load_samples() -> list[Sample]:
    if not MANIFEST.exists():
        sys.exit(f"{MANIFEST} not found: run import_telegram_export.py first")
    samples, unlabeled = [], 0
    with open(MANIFEST, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if not row["file"]:
                continue
            label = row.get("final_label", "")
            if label not in ("open", "closed"):
                unlabeled += label == ""
                continue
            risky = label == "open" and (
                row["review_label"] == "partial" or row["review_label"] not in ("", "open")
            )
            samples.append(Sample(row["file"], int(label == "open"), row["date"], risky))
    if unlabeled:
        print(f"Warning: {unlabeled} photos without final_label skipped (finish the review first)")
    samples.sort(key=lambda s: s.date)
    return samples


def load_image(path: Path) -> np.ndarray:
    """Exactly the production preprocessing (TFLiteAnalyzer.analyze)."""
    from PIL import Image

    img = Image.open(io.BytesIO(path.read_bytes())).convert("RGB").resize(IMG_SIZE)
    return np.array(img, dtype=np.float32) / 255.0


def augment(path: Path, rng: random.Random) -> np.ndarray:
    """Random variation of a photo, mimicking real camera conditions.

    Small rotation/crop (camera nudged), brightness/contrast, color casts,
    grayscale (night infrared), blur and JPEG recompression. No horizontal
    flip: the camera is fixed and a mirrored garage never occurs.
    """
    from PIL import Image, ImageEnhance, ImageFilter

    img = Image.open(path).convert("RGB")
    w, h = img.size
    img = img.rotate(rng.uniform(-4, 4), resample=Image.BILINEAR)
    scale = rng.uniform(0.82, 1.0)
    cw, ch = int(w * scale), int(h * scale)
    x0, y0 = rng.randint(0, w - cw), rng.randint(0, h - ch)
    img = img.crop((x0, y0, x0 + cw, y0 + ch)).resize(IMG_SIZE)

    if rng.random() < 0.2:
        img = img.convert("L").convert("RGB")
    else:
        img = ImageEnhance.Color(img).enhance(rng.uniform(0.6, 1.4))
        if rng.random() < 0.3:  # color cast (blue/red tints seen in the data)
            gains = [rng.uniform(0.8, 1.2) for _ in range(3)]
            img = Image.merge("RGB", [c.point(lambda v, g=g: min(255, int(v * g))) for c, g in zip(img.split(), gains)])
    img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.65, 1.35))
    img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.7, 1.3))
    if rng.random() < 0.15:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.0)))
    if rng.random() < 0.5:
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=rng.randint(40, 90))
        img = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    return np.array(img, dtype=np.float32) / 255.0


# ── Model ────────────────────────────────────────────────────────────


def build_models():
    """Return (bottom, top, full) sharing the same layers.

    bottom: [0,1] image -> SPLIT_LAYER activations (frozen, used for caching)
    top:    SPLIT_LAYER activations -> P(open) (trained)
    full:   [0,1] image -> P(open) (exported to TFLite)
    """
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    tf.random.set_seed(SEED)
    base = keras.applications.MobileNetV2(input_shape=(*IMG_SIZE, 3), include_top=False, weights="imagenet")
    names = [layer.name for layer in base.layers]
    top_layers = base.layers[names.index(SPLIT_LAYER) + 1:]

    image_in = keras.Input((*IMG_SIZE, 3), name="image")
    rescale = layers.Rescaling(2.0, offset=-1.0, name="to_mobilenet_range")
    bottom_net = keras.Model(base.input, base.get_layer(SPLIT_LAYER).output, name="bottom")
    bottom = keras.Model(image_in, bottom_net(rescale(image_in)), name="bottom_with_rescale")
    for layer in bottom.layers:
        layer.trainable = False

    head = [
        layers.GlobalAveragePooling2D(name="gap"),
        layers.Dropout(0.3, name="drop1"),
        layers.Dense(64, activation="relu", name="fc"),
        layers.Dropout(0.2, name="drop2"),
        layers.Dense(1, activation="sigmoid", name="p_open"),
    ]

    # Blocks above SPLIT_LAYER are a plain chain (no residual), so they can be
    # re-applied to a cached-activation input; numerically checked below.
    feat_in = keras.Input(bottom.output.shape[1:], name="features")
    x = feat_in
    for layer in top_layers:
        x = layer(x)
    for layer in head:
        x = layer(x)
    top = keras.Model(feat_in, x, name="top")

    x = base(rescale(image_in))
    for layer in head:
        x = layer(x)
    full = keras.Model(image_in, x, name="garage_classifier")

    probe = np.random.default_rng(SEED).random((2, *IMG_SIZE, 3), dtype=np.float32)
    chained = top_layers_output(top_layers, bottom(probe))
    direct = base(rescale(probe))
    if not np.allclose(chained, direct, atol=1e-4):
        sys.exit(f"Layers after {SPLIT_LAYER} are not a plain chain: pick another SPLIT_LAYER")
    return bottom, top, full, top_layers


def top_layers_output(top_layers, x):
    for layer in top_layers:
        x = layer(x)
    return x


def set_fine_tune(top_layers, enabled: bool) -> None:
    """Train conv layers above SPLIT_LAYER; BatchNorm always stays frozen."""
    from tensorflow.keras import layers

    for layer in top_layers:
        layer.trainable = enabled and not isinstance(layer, layers.BatchNormalization)


# ── Feature cache ────────────────────────────────────────────────────


def compute_features(bottom, samples: list[Sample], n_aug: int, tag: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bottom activations for each sample (k=0 original, k>=1 augmented), cached on disk.

    Returns (features float16 [N*(1+n_aug), 7, 7, C], index into samples, is_original).
    """
    key = hashlib.sha256(
        f"{AUG_VERSION}|{SPLIT_LAYER}|{n_aug}|".encode() + "\n".join(s.file for s in samples).encode()
    ).hexdigest()[:16]
    path = CACHE_DIR / f"features_{tag}_{key}.npz"
    if path.exists():
        data = np.load(path)
        print(f"  {tag}: cached features loaded ({path.name})")
        return data["feats"], data["idx"], data["orig"]

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    feats, idx, orig = [], [], []
    batch, meta = [], []
    t0 = time.time()
    total = len(samples) * (1 + n_aug)

    def flush():
        out = bottom.predict_on_batch(np.stack(batch))
        feats.append(out.astype(np.float16))
        batch.clear()

    for i, s in enumerate(samples):
        src = DATA_DIR / s.file
        rng = random.Random(f"{SEED}:{s.file}")
        for k in range(1 + n_aug):
            batch.append(load_image(src) if k == 0 else augment(src, rng))
            meta.append((i, k == 0))
            if len(batch) == 32:
                flush()
        done = len(meta)
        if done % 640 < 1 + n_aug:
            rate = done / (time.time() - t0)
            print(f"  {tag}: {done}/{total} images ({rate:.1f}/s, ~{(total - done) / rate / 60:.0f} min left)")
    if batch:
        flush()
    result = (np.concatenate(feats), np.array([m[0] for m in meta]), np.array([m[1] for m in meta]))
    np.savez(path, feats=result[0], idx=result[1], orig=result[2])
    print(f"  {tag}: {total} images in {(time.time() - t0) / 60:.1f} min, cached")
    return result


# ── Training ─────────────────────────────────────────────────────────


def sample_weights(samples: list[Sample], idx: np.ndarray) -> np.ndarray:
    y = np.array([s.y for s in samples])
    n_open, n_closed = int(y.sum()), int(len(y) - y.sum())
    class_w = {1: len(y) / (2 * n_open), 0: len(y) / (2 * n_closed)}
    return np.array([class_w[samples[i].y] * (RISKY_WEIGHT if samples[i].risky else 1.0) for i in idx], dtype=np.float32)


def fit(top, top_layers, train, val):
    """Two phases on cached features: head only, then block 16 + Conv_1 + head."""
    import tensorflow as tf
    from tensorflow import keras

    (x_tr, y_tr, w_tr), (x_val, y_val) = train, val

    def dataset(x, y, w=None, shuffle=False):
        ds = tf.data.Dataset.from_tensor_slices((x, y) if w is None else (x, y, w))
        if shuffle:
            ds = ds.shuffle(len(x), seed=SEED, reshuffle_each_iteration=True)
        cast = (lambda a, b, *c: (tf.cast(a, tf.float32), b, *c))
        return ds.map(cast).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    train_ds = dataset(x_tr, y_tr, w_tr, shuffle=True)
    val_ds = dataset(x_val, y_val)
    callbacks = lambda: [keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True)]
    metrics = [keras.metrics.BinaryAccuracy(name="acc"), keras.metrics.AUC(name="auc")]

    print("\n--- Phase 1: classification head ---")
    set_fine_tune(top_layers, False)
    top.compile(optimizer=keras.optimizers.Adam(HEAD_LR), loss="binary_crossentropy", metrics=metrics)
    h1 = top.fit(train_ds, validation_data=val_ds, epochs=HEAD_EPOCHS, callbacks=callbacks(), verbose=2)

    print("\n--- Phase 2: fine-tune block 16 + Conv_1 (BatchNorm frozen) ---")
    set_fine_tune(top_layers, True)
    top.compile(optimizer=keras.optimizers.Adam(FINE_TUNE_LR), loss="binary_crossentropy", metrics=metrics)
    h2 = top.fit(train_ds, validation_data=val_ds, epochs=FINE_TUNE_EPOCHS, callbacks=callbacks(), verbose=2)
    set_fine_tune(top_layers, False)
    return len(h1.history["loss"]), len(h2.history["loss"])


def prepare(bottom, samples: list[Sample], n_aug: int, tag: str):
    feats, idx, orig = compute_features(bottom, samples, n_aug, tag)
    y = np.array([samples[i].y for i in idx], dtype=np.float32)
    return feats, y, idx, orig


# ── Evaluation ───────────────────────────────────────────────────────


def report(name: str, probs: np.ndarray, samples: list[Sample], show_errors: bool = True) -> dict:
    y = np.array([s.y for s in samples])
    risky = np.array([s.risky for s in samples])
    pred = (probs > 0.5).astype(int)
    conf = np.where(pred == 1, probs, 1 - probs)
    wrong = pred != y
    uncertain = conf < PROD_CONFIDENCE
    missed_open = wrong & (y == 1)
    res = {
        "accuracy": 1 - wrong.mean(),
        "errors": int(wrong.sum()),
        "missed_open": int(missed_open.sum()),
        "false_open": int((wrong & (y == 0)).sum()),
        "risky_recall": float((pred[risky] == 1).mean()) if risky.any() else float("nan"),
        "uncertain": int(uncertain.sum()),
        # Production behaviour: a wrong answer is only acted upon when confident.
        "confident_errors": int((wrong & ~uncertain).sum()),
    }
    print(
        f"  {name:<14} acc {res['accuracy']:.2%}  errors {res['errors']:>3} "
        f"(open missed {res['missed_open']}, false open {res['false_open']})  "
        f"slightly-open recall {res['risky_recall']:.0%} ({int(risky.sum())})  "
        f"uncertain<{PROD_CONFIDENCE:.0%} {res['uncertain']}  confident errors {res['confident_errors']}"
    )
    if show_errors:
        for i in np.flatnonzero(wrong):
            s = samples[i]
            print(f"      {s.file:<38} true={'open' if s.y else 'closed':<6} p_open={probs[i]:.2f}{'  (slightly open)' if s.risky else ''}")
    return res


def tflite_probs(model_path: Path, samples: list[Sample]) -> np.ndarray:
    """Run a .tflite file exactly like production does."""
    import tensorflow as tf

    interp = tf.lite.Interpreter(model_path=str(model_path))
    interp.allocate_tensors()
    inp, out = interp.get_input_details()[0], interp.get_output_details()[0]
    probs = []
    for s in samples:
        interp.set_tensor(inp["index"], load_image(DATA_DIR / s.file)[None])
        interp.invoke()
        probs.append(float(interp.get_tensor(out["index"])[0][0]))
    return np.array(probs)


def split_by_date(samples, val_from, test_from):
    train = [s for s in samples if s.date < val_from]
    val = [s for s in samples if val_from <= s.date < test_from]
    test = [s for s in samples if s.date >= test_from]
    return train, val, test


def holdout_days(samples, fraction=0.1):
    """Hold out whole random days (no near-duplicate leakage) for early stopping."""
    days = sorted({s.date[:10] for s in samples})
    rng = random.Random(SEED)
    held = set(rng.sample(days, max(1, int(len(days) * fraction))))
    return [s for s in samples if s.date[:10] not in held], [s for s in samples if s.date[:10] in held]


def describe(name, samples):
    n_open = sum(s.y for s in samples)
    period = f"{samples[0].date[:10]} -> {samples[-1].date[:10]}" if samples else "-"
    print(f"  {name:<6} {len(samples):>5} photos ({n_open} open, {len(samples) - n_open} closed)  {period}")


def run_evaluate(val_from: str, test_from: str) -> None:
    samples = load_samples()
    train, val, test = split_by_date(samples, val_from, test_from)
    print("Temporal split:")
    for name, part in (("train", train), ("val", val), ("test", test)):
        describe(name, part)

    bottom, top, full, top_layers = build_models()
    print("\nComputing backbone features:")
    x_tr, y_tr, idx_tr, _ = prepare(bottom, train, N_AUG, "train")
    x_val, y_val, _, _ = prepare(bottom, val, 0, "val")
    x_te, _, _, _ = prepare(bottom, test, 0, "test")
    w_tr = sample_weights(train, idx_tr)
    fit(top, top_layers, (x_tr, y_tr, w_tr), (x_val, y_val))

    print("\n--- Results on TEST (never seen in training) ---")
    new = top.predict(x_te.astype(np.float32), batch_size=BATCH_SIZE, verbose=0)[:, 0]
    report("new model", new, test)
    if TFLITE_PATH.exists():
        report("current model", tflite_probs(TFLITE_PATH, test), test)
    print("\n--- Validation (used for early stopping) ---")
    report("new model", top.predict(x_val.astype(np.float32), verbose=0)[:, 0], val, show_errors=False)
    if TFLITE_PATH.exists():
        report("current model", tflite_probs(TFLITE_PATH, val), val, show_errors=False)


def run_train() -> None:
    samples = load_samples()
    train, val = holdout_days(samples)
    print("Final training on all periods (random held-out days only for early stopping):")
    describe("train", train)
    describe("val", val)

    bottom, top, full, top_layers = build_models()
    print("\nComputing backbone features:")
    x_tr, y_tr, idx_tr, _ = prepare(bottom, train, N_AUG, "final_train")
    x_val, y_val, _, _ = prepare(bottom, val, 0, "final_val")
    fit(top, top_layers, (x_tr, y_tr, sample_weights(train, idx_tr)), (x_val, y_val))

    print("\n--- Held-out days ---")
    report("new model", top.predict(x_val.astype(np.float32), verbose=0)[:, 0], val)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    full.save(str(KERAS_PATH))
    print(f"\nKeras model saved: {KERAS_PATH}")
    export_tflite(full)

    # The exported file must behave like the trained model on real photos.
    tfl = tflite_probs(TFLITE_PATH, val)
    ref = top.predict(x_val.astype(np.float32), verbose=0)[:, 0]
    print(f"TFLite vs Keras on held-out days: max |diff| {np.abs(tfl - ref).max():.4f}, "
          f"same decision {np.mean((tfl > 0.5) == (ref > 0.5)):.2%}")


# ── Export / predict ─────────────────────────────────────────────────


def export_tflite(model=None):
    """Export to float16 TFLite via SavedModel (Keras 3 + TF 2.16)."""
    import tempfile

    import tensorflow as tf

    if model is None:
        print(f"Loading model from {KERAS_PATH}")
        model = tf.keras.models.load_model(str(KERAS_PATH))

    with tempfile.TemporaryDirectory() as tmpdir:
        saved_model_path = str(Path(tmpdir) / "saved_model")
        model.export(saved_model_path, verbose=False)
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        tflite_model = converter.convert()

    TFLITE_PATH.parent.mkdir(parents=True, exist_ok=True)
    TFLITE_PATH.write_bytes(tflite_model)
    print(f"TFLite model saved: {TFLITE_PATH} ({len(tflite_model) / (1024 * 1024):.1f} MB)")

    interpreter = tf.lite.Interpreter(model_path=str(TFLITE_PATH))
    interpreter.allocate_tensors()
    print(f"Input: {interpreter.get_input_details()[0]['shape']}, Output: {interpreter.get_output_details()[0]['shape']}")


def predict_image(image_path: str):
    """Run inference on a single image using the TFLite model."""
    if not TFLITE_PATH.exists():
        sys.exit(f"Model not found: {TFLITE_PATH}. Run --train first.")
    s = Sample(str(Path(image_path).resolve()), 0, "", False)
    prob_open = float(tflite_probs(TFLITE_PATH, [s])[0])
    label = "open" if prob_open > 0.5 else "closed"
    confidence = prob_open if label == "open" else 1 - prob_open
    print(f"Prediction: {label.upper()} (confidence={confidence:.2%})")
    print(f"Raw output: {prob_open:.4f} (>0.5 = open, <0.5 = closed)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--evaluate", action="store_true", help="Temporal split: train, then compare new vs current model on test")
    parser.add_argument("--train", action="store_true", help="Train the final model on all data and export TFLite")
    parser.add_argument("--export", action="store_true", help="Re-export the .keras model to TFLite")
    parser.add_argument("--predict", type=str, help="Predict on a single image")
    parser.add_argument("--val-from", default=VAL_FROM, help=f"Validation period start for --evaluate (default {VAL_FROM})")
    parser.add_argument("--test-from", default=TEST_FROM, help=f"Test period start for --evaluate (default {TEST_FROM})")
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    if args.evaluate:
        run_evaluate(args.val_from, args.test_from)
    elif args.train:
        run_train()
    elif args.export:
        export_tflite()
    elif args.predict:
        predict_image(args.predict)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
