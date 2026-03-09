# TFLite Local Classifier

## Overview

The garage door status can be detected using two analyzers:

| | **Gemini** (default) | **TFLite** (local) |
|---|---|---|
| How it works | Sends image to Gemini API, gets JSON response | Runs MobileNetV2 classifier locally |
| Cost | ~$0.30/M input tokens | Free |
| Latency | 1-3 seconds (network + API) | ~50-100ms (CPU only) |
| Accuracy | Good, but occasional random false positives | 90%+ on real production images, 100% on validation set |
| Dependencies | `google-genai` | `ai-edge-litert` + `Pillow` (~35MB) |
| Model size | None (cloud API) | 4.4 MB (`model/garage_classifier.tflite`) |
| Reasoning | Returns Italian text explanation | Returns confidence score only |

Both analyzers return the same `GeminiAnalysisResult` object with `status`, `confidence`, and `reasoning`, so the rest of the pipeline (debounce, notifications, state management) works identically.

## How to Switch

Set the `GM_ANALYZER` environment variable in your `.env`:

```bash
# Use local TFLite model (free, fast)
GM_ANALYZER=tflite

# Use Gemini API (default, requires GM_GEMINI_API_KEY)
GM_ANALYZER=gemini
```

When using `tflite`, `GM_GEMINI_API_KEY` is not required.

Redeploy after changing:
```bash
./deploy.sh
```

## How the Model Works

### Architecture
- **Base model**: MobileNetV2 (pretrained on ImageNet)
- **Transfer learning**: Frozen base + custom classification head
- **Head**: GlobalAveragePooling2D → Dropout(0.3) → Dense(64, ReLU) → Dropout(0.2) → Dense(1, Sigmoid)
- **Output**: Single float 0.0-1.0 (0 = closed, 1 = open)
- **Confidence**: Distance from 0.5 decision boundary. Values close to 0 or 1 mean high confidence.

### Training Data
- ~137 labeled images from the Blink Mini camera
- Mix of conditions: daytime (color), IR nighttime (B&W), dusk, night with garage light
- With and without car in frame
- Images resized to 224x224 for model input
- Augmentation: random flip, brightness ±30%, contrast, saturation

### Training Process (2 phases)
1. **Head training** (30 epochs): Only the classification head is trained, base MobileNetV2 is frozen
2. **Fine-tuning** (20 epochs): Last 20 layers of MobileNetV2 are unfrozen and trained with a lower learning rate

### Quantization
The model is exported with float16 quantization, reducing size from ~9MB to 4.4MB with negligible accuracy loss.

## Confidence Threshold

The existing `GM_CONFIDENCE_THRESHOLD` (default 0.7) works with both analyzers. If the model's confidence is below this threshold, the status change is ignored — same behavior as with Gemini.

The debounce system (2 consecutive confirmations required for a status change) also applies, providing an additional safety net.

## Retraining the Model

If you need to retrain with new images:

### 1. Add labeled images

Edit `scripts/train_model.py` and add entries to the `LABELS` dictionary:

```python
LABELS["my_new_image.jpg"] = "open"   # or "closed"
```

Add the image files to one of the `SOURCE_DIRS` directories.

### 2. Run training

```bash
# Activate a Python environment with tensorflow installed
pip install "tensorflow<2.16" Pillow scipy

# Train and export
python scripts/train_model.py --train
```

This will:
- Load all labeled images
- Split 80/20 into train/validation (stratified by class)
- Train in 2 phases (head only, then fine-tuning)
- Evaluate on validation set and full dataset
- Export to `model/garage_classifier.tflite`

### 3. Test on a single image

```bash
python scripts/train_model.py --predict /path/to/image.jpg
```

### 4. Deploy

```bash
./deploy.sh
```

## File Structure

```
model/
  garage_classifier.tflite    # 4.4 MB - deployed with the function
src/garage_monitor/
  tflite_analyzer.py           # TFLite inference wrapper
  gemini_analyzer.py           # Gemini API wrapper (unchanged)
  config.py                    # GM_ANALYZER setting
  main.py                      # Analyzer switch logic
scripts/
  train_model.py               # Training pipeline
```

## Known Limitations

- **Transition frames**: The model may misclassify images where the door is mid-opening/closing (10-20% open). This is irrelevant in production since snapshots are taken every 5 minutes and the door takes ~15 seconds to open/close.
- **Partially open**: Images with the door ~50% open (like IMG_6532) may be classified as closed. The debounce handles this.
- **New camera angle**: If the camera is moved, the model may need retraining with images from the new position.
- **No reasoning text**: Unlike Gemini, the TFLite model only provides a confidence score, not a text explanation of its decision.
