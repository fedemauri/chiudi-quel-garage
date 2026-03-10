# TFLite Local Classifier

## Overview

The system supports two pluggable analyzer backends for classifying the garage door state from a camera snapshot. The `GM_ANALYZER` environment variable controls which one is used at runtime:

| | **Gemini** (default, `GM_ANALYZER=gemini`) | **TFLite** (local, `GM_ANALYZER=tflite`) |
|---|---|---|
| How it works | Sends JPEG to Gemini cloud API with a structured prompt; receives JSON `{status, confidence, reasoning}` | Loads a pre-trained MobileNetV2 binary classifier and runs inference locally on the Cloud Function CPU |
| Cost | ~$0.30/M input tokens, ~$2.50/M output tokens (paid) | Free — no API calls, no tokens |
| Latency | 1–3 seconds (network round-trip + API processing) | ~50–100 ms (CPU-only inference, no network) |
| Accuracy | High, but subject to occasional unpredictable false positives (LLM hallucinations) | 90%+ on real production images; deterministic — same input always gives same output |
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

The full model is a `keras.Sequential` stack:

```
Layer                           Output Shape       Parameters    Trainable
──────────────────────────────────────────────────────────────────────────
MobileNetV2 (base)              (None, 7, 7, 1280) ~2.2M         Partially*
GlobalAveragePooling2D          (None, 1280)        0             -
Dropout(0.3)                    (None, 1280)        0             -
Dense(64, activation="relu")    (None, 64)          81,984        Yes
Dropout(0.2)                    (None, 64)          0             -
Dense(1, activation="sigmoid")  (None, 1)           65            Yes
```

\* During Phase 1 training, MobileNetV2 is fully frozen. During Phase 2 fine-tuning, the last 20 layers are unfrozen.

**Why this architecture:**
- `GlobalAveragePooling2D` collapses the 7×7×1280 feature maps into a 1280-dimensional vector, reducing parameters and preventing overfitting
- Two `Dropout` layers (0.3 and 0.2) provide regularization — critical with a small dataset (~140 images)
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
pip install "tensorflow>=2.16" Pillow scipy
```

Note: Training requires full TensorFlow (not just tflite-runtime). TensorFlow is only needed locally for training — it is NOT deployed to the Cloud Function.

### Dataset structure

Images are organized in a directory-based layout:

```
data/images/
  closed/        # Images where the garage door is closed
    IMG_0001.jpg
    IMG_0002.jpg
    ...
  open/          # Images where the garage door is open
    IMG_0100.jpg
    IMG_0101.jpg
    ...
```

The training script automatically discovers all images (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`) in these directories and assigns labels based on the parent directory name. Currently the dataset contains ~140 labeled images from the Blink Mini camera, covering:

- Daytime (color images with natural light)
- Nighttime IR (black & white infrared images)
- Dusk / low-light transitions
- Night with garage light on (color, artificial light)
- With and without car in frame
- Various door positions (fully open, fully closed)

### Data augmentation

Because the dataset is small (~140 images), aggressive augmentation is applied during training:

```python
tf.image.random_flip_left_right(image)     # horizontal flip
tf.image.random_brightness(image, 0.3)     # brightness ±30%
tf.image.random_contrast(image, 0.7, 1.3)  # contrast variation
tf.image.random_saturation(image, 0.7, 1.3) # saturation variation
```

This effectively multiplies the training set by generating varied versions of each image on every epoch, helping the model generalize to different lighting conditions.

### Class weight balancing

The training script computes class weights inversely proportional to class frequency:

```python
class_weight = {
    0: n_total / (2 * n_closed),  # weight for "closed"
    1: n_total / (2 * n_open),    # weight for "open"
}
```

This ensures the model doesn't become biased toward the majority class if the dataset is imbalanced (e.g., more "closed" images than "open" ones).

### Training process (2 phases)

**Phase 1 — Head training (up to 30 epochs, LR=0.0005):**
- MobileNetV2 base is completely frozen (`trainable = False`)
- Only the classification head layers are trained (GlobalAveragePooling2D + Dense layers)
- Uses Adam optimizer with binary cross-entropy loss
- EarlyStopping monitors `val_loss` with patience=8 and restores best weights

**Phase 2 — Fine-tuning (up to 20 epochs, LR=0.00005):**
- The last 20 layers of MobileNetV2 are unfrozen (`base_model.layers[:-20]` stay frozen)
- Learning rate is reduced 10× to avoid destroying pre-trained features
- Same EarlyStopping settings
- This allows the model to adapt MobileNetV2's higher-level features to the specific camera perspective

All random seeds are fixed (`random.seed(42)`, `np.random.seed(42)`, `tf.random.set_seed(42)`) for reproducible results.

### Train/validation split

80/20 stratified split: 20% of each class (open, closed) is held out for validation. This ensures both classes are represented in the validation set.

### TFLite export

After training, the model is automatically exported to TFLite via an intermediate SavedModel (required for Keras 3 compatibility):

```python
model.export(saved_model_path)          # Keras 3 → SavedModel
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_model = converter.convert()      # SavedModel → TFLite (float16)
```

The exported model is saved to `model/garage_classifier.tflite` and a Keras checkpoint is saved to `model/garage_classifier.keras` for future fine-tuning.

## Retraining the Model

### 1. Add labeled images

Place new JPEG/PNG images in the appropriate directory:

```bash
# For closed door images:
cp new_image.jpg data/images/closed/

# For open door images:
cp new_image.jpg data/images/open/
```

### 2. Train and export

```bash
python scripts/train_model.py --train
```

This will:
- Load all images from `data/images/{open,closed}/`
- Split 80/20 into train/validation (stratified by class)
- Train in 2 phases (head only, then fine-tuning)
- Print per-image validation results and full-dataset accuracy
- Export to `model/garage_classifier.tflite`

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
data/images/                  # Training images (not committed to git)
  open/                       # Images of open garage door
  closed/                     # Images of closed garage door
src/garage_monitor/
  tflite_analyzer.py          # TFLite inference wrapper (TFLiteAnalyzer class)
  gemini_analyzer.py          # Gemini API wrapper (GeminiAnalyzer class)
  config.py                   # GM_ANALYZER setting (config.analyzer field)
  main.py                     # Analyzer switch: if settings.analyzer == "tflite" → TFLiteAnalyzer()
scripts/
  train_model.py              # Training, export, and single-image prediction
```

## Known Limitations

- **Transition frames**: The model may misclassify images where the door is mid-opening/closing (10–20% open). This is irrelevant in production since snapshots are taken every 5 minutes and the door takes ~15 seconds to open/close.
- **Partially open**: Images with the door ~50% open may be classified as closed with moderate confidence. The confidence threshold helps filter these.
- **Camera-specific**: The model is trained on images from one specific Blink Mini camera in one specific position. If the camera is moved or replaced, the model will likely need retraining with images from the new perspective.
- **No semantic reasoning**: Unlike Gemini, the TFLite model cannot explain *why* it classified an image a certain way. It only provides a probability score. The `reasoning` field contains a synthetic string with the raw probability, not a human explanation.
- **Training environment**: Full TensorFlow (~2 GB) is required for training. Only the lightweight TFLite runtime (~15 MB) is needed for inference at runtime.
