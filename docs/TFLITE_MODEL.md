# TFLite Local Classifier

## Overview

The system supports two pluggable analyzer backends for classifying the garage door state from a camera snapshot. The `GM_ANALYZER` environment variable controls which one is used at runtime:

| | **Gemini** (default, `GM_ANALYZER=gemini`) | **TFLite** (local, `GM_ANALYZER=tflite`) |
|---|---|---|
| How it works | Sends JPEG to Gemini cloud API with a structured prompt; receives JSON `{status, confidence, reasoning}` | Loads a pre-trained MobileNetV2 binary classifier and runs inference locally on the Cloud Function CPU |
| Cost | ~$0.30/M input tokens, ~$2.50/M output tokens (paid) | Free — no API calls, no tokens |
| Latency | 1–3 seconds (network round-trip + API processing) | ~50–100 ms (CPU-only inference, no network) |
| Accuracy | High, but subject to occasional unpredictable false positives (LLM hallucinations) | 99.5% on the latest two months of production photos (Oct 2026 model); deterministic — same input always gives same output |
| Dependencies | `google-genai` | `ai-edge-litert` (Python ≥3.12) or `tflite-runtime` (Python <3.12) + `Pillow` |
| Model size | None (cloud API) | 4.4 MB file (`model/garage_classifier.tflite`) — deployed alongside the function code |
| Reasoning output | Italian text explanation from Gemini (e.g., "Pannelli visibili, porta chiusa") | Synthetic text with raw probability (e.g., "Porta chiusa (p=0.97)") |

Both analyzers implement the same `analyze(image_bytes) -> GeminiAnalysisResult` interface, returning `status` (open/closed), `confidence` (0.0–1.0), and `reasoning` (str). The rest of the pipeline — state change detection, confidence threshold, notifications, reminders, event logging — works identically regardless of which analyzer is active.

## How to Switch

Set `GM_ANALYZER` in `.env` and redeploy:

```bash
# Use local TFLite model (free, fast, deterministic)
GM_ANALYZER=tflite

# Use Gemini API (default — requires GM_GEMINI_API_KEY)
GM_ANALYZER=gemini
```

When using `tflite`, `GM_GEMINI_API_KEY` is not required (it can be empty or omitted). The deploy script (`deploy.sh`) validates this conditionally: it only checks for `GM_GEMINI_API_KEY` when `GM_ANALYZER=gemini`.

After changing, redeploy:
```bash
./deploy.sh
```

## What is MobileNetV2

[MobileNetV2](https://arxiv.org/abs/1801.04381) is a lightweight convolutional neural network designed by Google for mobile and edge devices. Key characteristics:

- **Inverted residual blocks** with depthwise separable convolutions — dramatically fewer parameters than standard CNNs (3.4M params vs. ResNet50's 25.6M)
- **Pre-trained on ImageNet** (1.4M images, 1000 classes) — the base model already "understands" visual features like edges, textures, shapes, and objects
- **Transfer learning friendly** — the pre-trained feature extraction layers can be reused as-is, and only a small classification head needs training on domain-specific data
- **224×224 input size** — images are resized to this standard resolution before inference

In this project, MobileNetV2 is used as a frozen feature extractor. A custom binary classification head is trained on top to distinguish "garage door open" vs. "garage door closed" from the Blink Mini camera's specific perspective.

## Model Architecture (detailed)

The full model is a Keras functional model (input: 224×224×3 float32 in [0, 1]):

```
Layer                           Output Shape       Parameters    Trainable
──────────────────────────────────────────────────────────────────────────
Rescaling(2, offset=-1)         (None, 224, 224, 3) 0            -
MobileNetV2 (base)              (None, 7, 7, 1280) ~2.2M         Partially*
GlobalAveragePooling2D          (None, 1280)        0             -
Dropout(0.3)                    (None, 1280)        0             -
Dense(64, activation="relu")    (None, 64)          81,984        Yes
Dropout(0.2)                    (None, 64)          0             -
Dense(1, activation="sigmoid")  (None, 1)           65            Yes
```

\* During Phase 1 training, MobileNetV2 is fully frozen. During Phase 2 fine-tuning, block 16 and `Conv_1` are trained (BatchNorm layers stay frozen).

**Why this architecture:**
- `GlobalAveragePooling2D` collapses the 7×7×1280 feature maps into a 1280-dimensional vector, reducing parameters and preventing overfitting
- Two `Dropout` layers (0.3 and 0.2) provide regularization — important with a single-camera dataset of ~2000 photos
- A single hidden `Dense(64)` layer provides enough capacity for binary classification without overfitting
- Final `Dense(1, sigmoid)` outputs a single probability: 0.0 = definitely closed, 1.0 = definitely open

**Decision boundary:**
- Output > 0.5 → **OPEN** (confidence = raw output)
- Output ≤ 0.5 → **CLOSED** (confidence = 1.0 − raw output)
- Values near 0 or 1 mean high confidence; values near 0.5 mean the model is uncertain

## TFLite Runtime and Quantization

### What is TFLite

TensorFlow Lite (TFLite) is a runtime for executing neural networks on edge devices without needing the full TensorFlow framework. The `.tflite` file is a FlatBuffer-serialized model optimized for fast loading and small binary size.

### Runtime fallback chain

The `TFLiteAnalyzer` constructor tries three runtimes in order:

```python
try:
    from ai_edge_litert import interpreter as tflite    # Python ≥3.12 (Google's new package)
except ImportError:
    try:
        import tflite_runtime.interpreter as tflite      # Python <3.12 (legacy package)
    except ImportError:
        import tensorflow.lite as tflite                  # Full TensorFlow (dev/training only)
```

In production (Cloud Function, Python 3.12), `ai-edge-litert` is used — it's a ~15 MB wheel with no native TensorFlow dependency. The `pyproject.toml` declares conditional dependencies:

```toml
"tflite-runtime>=2.14.0; python_version<'3.12'",
"ai-edge-litert>=1.0.0; python_version>='3.12'",
```

### Float16 quantization

The Keras model is converted to TFLite using float16 quantization:

```python
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
```

This reduces the model from ~9 MB (full float32) to **4.4 MB** with negligible accuracy loss. Float16 is a good balance: smaller than float32, but more accurate than int8 quantization (which would require a representative calibration dataset).

## Inference Pipeline (step by step)

When `GM_ANALYZER=tflite`, here's exactly what happens on each scheduled invocation:

1. **Model loading** — `TFLiteAnalyzer.__init__()` loads `model/garage_classifier.tflite` via the TFLite Interpreter, allocates tensors, and caches input/output tensor details. The model is loaded once per Cloud Function cold start and reused for the container's lifetime.

2. **Image preprocessing** — `analyze(image_bytes)` receives raw JPEG bytes from the Blink camera and:
   - Opens the JPEG via `PIL.Image.open(io.BytesIO(image_bytes))`
   - Converts to RGB (`.convert("RGB")`) — handles any color space differences
   - Resizes to 224×224 pixels (`.resize((224, 224))`) — MobileNetV2's expected input
   - Converts to a NumPy float32 array normalized to [0.0, 1.0] (`/ 255.0`)
   - Adds a batch dimension (`np.expand_dims(axis=0)`) → shape `(1, 224, 224, 3)`

3. **Inference** — Sets the input tensor, invokes the interpreter, reads the output tensor. This takes ~50–100 ms on a Cloud Function CPU.

4. **Result mapping** — The single output float is mapped to:
   - `GarageStatus.OPEN` or `GarageStatus.CLOSED`
   - A confidence score (distance from 0.5)
   - A synthetic Italian reasoning string (e.g., `"Porta aperta (p=0.92)"`)
   - `input_tokens=0, output_tokens=0` — so the usage tracking correctly skips Gemini cost calculations

5. **Return** — The `GeminiAnalysisResult` is returned to `main.py`, which processes it identically to a Gemini result.

## Training Pipeline (detailed)

Training is done locally, not on the Cloud Function. The script is `scripts/train_model.py`.

### Prerequisites

```bash
python3.12 -m venv .venv-train
.venv-train/bin/pip install "tensorflow==2.16.2" Pillow scipy
```

TensorFlow 2.16.2 is the last release with macOS x86_64 (Intel) wheels; it is only needed locally and is NOT deployed to the Cloud Function. Training is CPU-only and designed for it (see below): on an i7-6700HQ the full run takes ~20 minutes.

### Dataset

The dataset is built from the photos the bot sent on Telegram (see [Retraining the Model](#retraining-the-model)): `data/telegram/manifest.csv` lists every photo and its `final_label` set during review. Only `open` / `closed` photos are used (`exclude` and unlabeled ones are skipped). The task is binary: **a door open even a few cm is `open`**.

The October 2026 dataset has 2101 photos (1110 open, 991 closed) from March to October 2026: daytime color, night infrared, garage light on, color casts, people and other cars in frame, the door mid-travel, and two camera positions.

### Preprocessing

Inputs are exactly what `TFLiteAnalyzer` feeds in production: `PIL.Image.open(...).convert("RGB").resize((224, 224))`, float32 in [0, 1]. The model starts with a `Rescaling(2, offset=-1)` layer that maps them to the [-1, 1] range MobileNetV2 was pre-trained on (the model before October 2026 skipped this step and fed [0, 1] directly).

### Cached backbone features (why it is fast on CPU)

MobileNetV2 is frozen up to `block_15_add`. Its activations (7×7×160 per image) are computed once for every photo and for 5 augmented copies of each training photo, then cached in `data/telegram/cache/`. Only block 16, `Conv_1` and the classification head are trained, on the cached activations — each epoch takes ~20 s instead of minutes. The layers above `block_15_add` are a plain chain; `build_models()` checks numerically that `top(bottom(x)) == full(x)` before training.

### Data augmentation

Applied on the full-resolution photo before resizing, to mimic real camera conditions: rotation ±4° and random crop 82–100% (camera nudged), color saturation, random color cast (per-channel gains), 20% grayscale (night infrared), brightness ±35%, contrast ±30%, light blur, JPEG recompression. No horizontal flip: the camera is fixed and a mirrored garage never occurs.

### Weights

Class weights balance open/closed. Photos where the door is only **slightly open** (review label `partial`, or the reviewer overrode "closed" with "open") get weight ×2: they are the risky cases.

### Training process (2 phases)

1. **Head** (GAP → Dropout 0.3 → Dense 64 → Dropout 0.2 → Dense 1 sigmoid): Adam 1e-3, up to 40 epochs.
2. **Fine-tuning** of block 16 + `Conv_1` + head: Adam 5e-5, up to 30 epochs. BatchNorm layers stay frozen (inference mode).

Both phases use EarlyStopping on `val_loss` (patience 6, best weights restored). Seeds are fixed (42).

### Evaluation (`--evaluate`)

Photos are split **by date**, not randomly: consecutive photos of the same opening are near-identical, and a random split would leak them between train and test. Default: train until July, validation August (early stopping), test September–October. The new model and the current `model/garage_classifier.tflite` are scored on the same test photos, reporting errors, missed openings, slightly-open recall, photos below the production confidence threshold (0.7) and *confident errors* (wrong and above 0.7, i.e. a wrong notification in production).

October 2026 results on the September–October test set (185 photos, never seen in training):

| | Previous model | New model |
|---|---|---|
| Accuracy | 96.8% | 99.5% |
| Errors | 6 (5 missed openings) | 1 (a half-open door at night, p_open = 0.33) |
| Confident errors | 6 | 0 |

### Final model (`--train`)

Trained on all periods, holding out 10% of random whole days for early stopping only. It is saved as `model/garage_classifier.keras` and exported to `model/garage_classifier.tflite` (float16 via SavedModel, ~4.4 MB); the script then checks that the `.tflite` file gives the same decisions as the Keras model on the held-out days.

## Retraining the Model

### 1. Collect and label photos


Every photo the bot sent is a labeled sample. The Bot API cannot read chat history, so export it from **Telegram Desktop**: bot chat → ⋮ → *Export chat history* → only *Photos* (JSON or HTML format). Then:

```bash
python scripts/import_telegram_export.py <export_dir> --dry-run   # summary only (JSON or HTML export)
python scripts/import_telegram_export.py <export_dir>              # writes data/telegram/
```

Photos are labeled from their caption (status change, night alert, reminder, final warning → `open`; status change to closed → `closed`; `/foto` → `unlabeled`) and copied to `data/telegram/{open,closed,unlabeled}/` as `YYYYMMDD_HHMMSS_<msgid>.jpg`. Exact duplicates and near-identical reminders (dHash, `--near-dup-threshold`) are skipped; status-change photos are always kept.

Captions record what the model decided, not ground truth. `data/telegram/manifest.csv` flags photos worth reviewing before training:

| Flag | Meaning |
|---|---|
| `short_closed` | Closed, then reopened within `--flip-minutes` (default 10): one of the two is likely wrong. Short *open* spells are normal and not flagged |
| `low_confidence` | Caption confidence below `--low-confidence` (default 0.85) |
| `reminder_while_closed` | "Still open" reminder sent after a change to closed |
| `unlabeled` | `/foto` snapshot or unrecognized caption: needs a label |

Fill `review_label` / `review_note` during review. `data/` is git-ignored (private photos).

To decide the final labels, run the local review page (127.0.0.1 only, photos never leave the machine):

```bash
python scripts/review_dataset.py   # opens http://127.0.0.1:8765/
```

It shows a filterable grid (default view: photos to check — model/review disagreements, partial doors, suspicious flags, uncertain notes) and a large viewer with keys `A` open, `C` closed, `X` exclude, `Enter` accept the review label, `←/→` navigate. The task is binary: a door open even a few cm is **open** (review labels `partial` are proposed as open). Each decision is saved immediately to the `final_label` / `final_note` columns of `manifest.csv` (a `manifest.backup.csv` copy is made on first start).

### 2. Evaluate, then train and export

```bash
.venv-train/bin/python scripts/train_model.py --evaluate   # new vs current model on the latest months
.venv-train/bin/python scripts/train_model.py --train      # final model on all data -> model/garage_classifier.tflite
```

Deploy only if `--evaluate` shows the new model is better than the current one.

### 3. Test on a single image

```bash
python scripts/train_model.py --predict /path/to/image.jpg
```

Output example:
```
Prediction: CLOSED (confidence=97.23%)
Raw output: 0.0277 (>0.5 = open, <0.5 = closed)
```

### 4. Export only (without retraining)

If you have a `.keras` checkpoint and just want to re-export to TFLite:

```bash
python scripts/train_model.py --export
```

### 5. Deploy the updated model

```bash
./deploy.sh
```

The `.tflite` file is bundled with the function source and deployed to Cloud Functions.

## Confidence Threshold

The `GM_CONFIDENCE_THRESHOLD` setting (default 0.7) works identically with both analyzers. If the model's confidence is below this threshold, the detected status change is ignored and the previous state is kept. This prevents spurious notifications from ambiguous images.

Example: if the TFLite model outputs 0.55 (meaning "open" with only 55% confidence), the confidence (0.55) is below the 0.7 threshold, so the status change is discarded.

## File Structure

```
model/
  garage_classifier.tflite    # 4.4 MB — deployed with the Cloud Function
  garage_classifier.keras     # ~20 MB — Keras checkpoint for retraining (not deployed)
data/telegram/                # Reviewed dataset (not committed to git: private photos)
  open/ closed/ unlabeled/    # Photos imported from the Telegram export
  manifest.csv                # Labels, flags, review and final decisions
  cache/                      # Cached backbone features
src/garage_monitor/
  tflite_analyzer.py          # TFLite inference wrapper (TFLiteAnalyzer class)
  gemini_analyzer.py          # Gemini API wrapper (GeminiAnalyzer class)
  config.py                   # GM_ANALYZER setting (config.analyzer field)
  main.py                     # Analyzer switch: if settings.analyzer == "tflite" → TFLiteAnalyzer()
scripts/
  import_telegram_export.py   # Build the dataset from a Telegram Desktop export
  review_dataset.py           # Local review page (with review_ui.html)
  train_model.py              # Evaluation, training, export, and single-image prediction
```

## Known Limitations

- **Slightly open doors**: A door raised a few cm is the hardest case (and must be `open`). The training set weights these ×2; the confidence threshold keeps uncertain predictions from changing state.
- **Camera-specific**: The model is trained on one Blink Mini camera (two positions in 2026). If the camera is moved or replaced, retrain with photos from the new perspective — the previous model degraded after the camera moved in autumn 2026.
- **Dataset bias**: Photos only exist when the bot sent one (state changes, reminders, `/foto`). An open door the model kept calling "closed" produced no photo, so those errors are invisible in the dataset.
- **No semantic reasoning**: Unlike Gemini, the TFLite model cannot explain *why* it classified an image a certain way. It only provides a probability score. The `reasoning` field contains a synthetic string with the raw probability, not a human explanation.
- **Training environment**: Full TensorFlow (~2 GB) is required for training. Only the lightweight TFLite runtime (~15 MB) is needed for inference at runtime.
