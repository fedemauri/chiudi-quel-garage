# Chiudi Quel Garage - Development Guide

## Project Overview

Automated garage door monitoring system with interactive Telegram bot. A Google Cloud Function runs on a schedule, captures a snapshot from a Blink Mini camera, analyzes it with a pluggable AI backend (Gemini cloud API or local TFLite MobileNetV2 classifier) to determine if the garage door is open or closed, and sends Telegram notifications on state changes. Users can interact with the bot via commands to check status, take photos, mute notifications, and view event history.

## Tech Stack

- **Runtime:** Python 3.12, deployed as Google Cloud Function (gen2)
- **Camera:** Blink Mini via `blinkpy` (async, uses saved credentials from Firestore)
- **AI (pluggable):**
  - **Gemini** (default): Gemini 2.5 Flash via `google-genai` SDK (paid API — $0.30/M input, $2.50/M output tokens)
  - **TFLite** (alternative): Local MobileNetV2 classifier via `ai-edge-litert` + `Pillow` (free, ~50-100ms)
- **State:** Google Cloud Firestore (garage state, Blink credentials, usage stats, event history)
- **Notifications:** Telegram Bot API via `httpx` (push notifications + interactive bot commands)
- **Config:** `pydantic-settings` with `GM_` env prefix
- **Scheduling:** Cloud Scheduler (every 5 min daytime, every 15 min nighttime)

## Project Structure

```
src/garage_monitor/
  main.py              # Cloud Function entry point (check_garage) + webhook routing
  config.py            # Settings via pydantic-settings (GM_ env vars)
  blink_client.py      # Blink camera auth + snapshot capture
  gemini_analyzer.py   # Gemini image analysis with structured JSON prompt
  tflite_analyzer.py   # Local TFLite MobileNetV2 classifier (alternative to Gemini)
  firestore_store.py   # Firestore persistence (state, credentials, usage, events)
  telegram_notifier.py # Telegram notifications (status change, reminders, alerts, reports, bot responses)
  telegram_handler.py  # Interactive bot command handler (/stato, /foto, /muto, etc.)
  models.py            # Data models (GarageStatus, GarageState, UsageStats, GeminiAnalysisResult)
model/
  garage_classifier.tflite  # Pre-trained MobileNetV2 classifier (4.4 MB)
  garage_classifier.keras   # Keras checkpoint for retraining
scripts/
  setup_gcp.sh              # One-time GCP project setup (APIs, Firestore, TTL)
  setup_blink.py             # Interactive Blink 2FA auth, saves credentials to Firestore
  test_telegram.py           # Verify Telegram bot + chat ID
  setup_telegram_webhook.py  # Register/delete Telegram webhook for bot commands
  train_model.py             # TFLite model training pipeline (MobileNetV2 transfer learning)
  compare_models.py          # Compare multiple Gemini models on test images
  test_heuristic.py          # Heuristic-based garage door detection test
tests/                       # pytest tests
docs/
  SETUP_BOT.md         # Post-deploy guide for Telegram bot setup
  TFLITE_MODEL.md      # TFLite model architecture, training, and usage guide
deploy.sh              # Full deploy: Cloud Function + Scheduler jobs
```

## Key Environment Variables (GM_ prefix)

| Variable | Required | Default | Description |
|---|---|---|---|
| `GM_BLINK_USERNAME` | yes | - | Blink account email |
| `GM_BLINK_PASSWORD` | yes | - | Blink account password |
| `GM_BLINK_CAMERA_NAME` | no | `Garage` | Camera name in Blink app |
| `GM_ANALYZER` | no | `gemini` | Analyzer backend: `gemini` (cloud AI, paid) or `tflite` (local ML, free) |
| `GM_GEMINI_API_KEY` | conditional | - | Gemini API key (required only when `GM_ANALYZER=gemini`) |
| `GM_GEMINI_MODEL` | no | `gemini-2.5-flash` | Gemini model ID |
| `GM_CONFIDENCE_THRESHOLD` | no | `0.7` | Min confidence to accept status change |
| `GM_TELEGRAM_BOT_TOKEN` | yes | - | Telegram bot token |
| `GM_TELEGRAM_CHAT_ID` | yes | - | Telegram chat ID for notifications |
| `GM_GCP_PROJECT_ID` | yes | - | Google Cloud project ID |
| `GM_FIRESTORE_COLLECTION` | no | `garage_monitor` | Firestore collection name |
| `GM_TELEGRAM_WEBHOOK_SECRET` | no | `""` | Secret for Telegram webhook validation (required for bot commands) |
| `GM_GEMINI_COST_ALERT_THRESHOLD` | no | `3.0` | Monthly Gemini cost alert threshold ($) |

## Development Commands

```bash
# Install in dev mode
pip install -e ".[dev]"

# Run tests
pytest

# Deploy (reads .env, deploys function + scheduler)
./deploy.sh

# Setup Telegram webhook (after deploy)
source .env && python scripts/setup_telegram_webhook.py <FUNCTION_URL>

# View logs
gcloud functions logs read check-garage --gen2 --region=europe-west1

# Manual trigger
gcloud scheduler jobs run garage-monitor-day --location=europe-west1

# TFLite model: train and export (requires full TensorFlow locally)
pip install "tensorflow>=2.16" Pillow scipy
python scripts/train_model.py --train       # Train + export to model/garage_classifier.tflite
python scripts/train_model.py --predict /path/to/image.jpg  # Test on single image
python scripts/train_model.py --export      # Re-export .keras to .tflite without retraining
```

## Architecture Notes

- **Entry point:** `check_garage()` in `main.py` — HTTP Cloud Function that routes between: Telegram webhook (JSON with "message"), scheduled report (`?action=report`), and normal garage check
- **Webhook security:** Validates `X-Telegram-Bot-Api-Secret-Token` header + chat_id. Function is `--allow-unauthenticated` for Telegram webhook access
- **Bot commands:** `/stato`, `/foto`, `/report`, `/muto Nh`, `/smuto`, `/storico` — handled by `telegram_handler.py`
- **Analyzer switch:** `GM_ANALYZER` selects between `gemini` (cloud API, paid) and `tflite` (local MobileNetV2, free, ~50-100ms). Both implement `analyze(image_bytes) -> GeminiAnalysisResult` so the pipeline is identical. The switch is in `main.py:_check_garage_async()`. TFLite uses `ai-edge-litert` runtime on Python ≥3.12 with float16-quantized model (`model/garage_classifier.tflite`, 4.4 MB). See `docs/TFLITE_MODEL.md` for architecture, training pipeline, and retraining instructions
- **Reminders:** Sends "still open" alerts every 15 min for up to 60 min after door opens (skipped when muted)
- **Mute system:** `/muto` silences reminders for N hours, auto-expires when time is up
- **Night alerts:** Priority alert with distinct formatting when garage opens between 0:00-6:59 (Europe/Rome, matching scheduler night hours)
- **Final warning:** One-time escalation alert after garage has been open >60 min
- **Health check:** Staleness detection alerts if last check exceeds expected interval (12 min day, 20 min night)
- **Cost monitoring:** Projected monthly Gemini cost alert in daily report when exceeding threshold
- **Event history:** Status changes logged to Firestore with 30-day TTL, viewable via `/storico`. Requires composite index on `(type ASC, timestamp DESC)`
- **Error handling:** Tracks consecutive errors in Firestore, alerts via Telegram after 3 failures
- **Credentials refresh:** Blink tokens are refreshed on each call and saved back to Firestore
- **Usage tracking:** Firestore counters track invocations, Gemini tokens, Firestore ops, and garage openings per month
- **Scheduler:** Two jobs — daytime (7-23h, every 5 min) and nighttime (0-6h, every 15 min), timezone Europe/Rome
- **Daily report:** Scheduler job at 21:00 sends usage summary via Telegram (includes garage activity and cost warnings)

## Coding Conventions

- All user-facing messages (Telegram notifications, error alerts) are in Italian
- Code comments and docstrings are in English
- Documentation files (README, CLAUDE.md) are in English
- Config uses `pydantic-settings` with `GM_` prefix — never hardcode secrets
- Async only for Blink client (blinkpy requires it); rest of the code is synchronous
- `functions_framework.http` decorator for Cloud Function entry point
