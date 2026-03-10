# Chiudi Quel Garage!

An automated garage door monitoring system that uses a **Blink Mini** indoor camera and AI vision to detect whether a garage door is open or closed. Two analyzer backends are available: **Google Gemini AI** (cloud, paid) and a **local TFLite MobileNetV2 classifier** (free, fast). When the state changes, the system sends a **Telegram notification with a photo**. If the door stays open, periodic reminders are sent. Everything runs serverless on **Google Cloud Platform**.

## How It Works

The system is built around a single Google Cloud Function that executes the following pipeline on every invocation:

1. **Read state** — Fetches the current garage status and Blink credentials from Firestore.
2. **Capture snapshot** — Authenticates with the Blink API using stored credentials (no 2FA required after initial setup) and commands the camera to take a fresh JPEG snapshot.
3. **AI analysis** — The JPEG snapshot is sent to the configured analyzer backend (`GM_ANALYZER`):
   - **Gemini** (default): the image is sent to the Gemini cloud API with a detailed Italian prompt describing the camera's perspective. Gemini returns a structured JSON with `status` (open/closed), `confidence` (0.0–1.0), and `reasoning` (Italian text). Temperature is set to 0.0 for deterministic output. Cost: ~$0.30/M input tokens + $2.50/M output tokens.
   - **TFLite** (alternative): a pre-trained MobileNetV2 binary classifier (`model/garage_classifier.tflite`, 4.4 MB) runs locally on the Cloud Function CPU. The image is resized to 224×224, normalized to [0, 1], and fed through the model. The output is a single float (0=closed, 1=open) mapped to the same `GeminiAnalysisResult` format. Inference takes ~50–100 ms with zero API cost. See [TFLite Model](docs/TFLITE_MODEL.md) for full architecture and training details.
4. **State change detection** — If the AI classification differs from the stored state *and* the confidence exceeds the configured threshold (default 0.7), the system records a state change.
5. **Telegram notification** — On state change, sends a Telegram message with the snapshot photo, the old and new status, confidence score, and the analyzer's reasoning. On first run, sends a "monitor started" confirmation.
6. **Open-door reminders** — If the door has been open for more than 15 minutes, sends a reminder with photo every 15 minutes, up to a maximum of 60 minutes.
7. **Persist state** — Saves the updated garage state, refreshed Blink credentials, and increments usage counters in Firestore.
8. **Error handling** — Tracks consecutive errors. After 3 consecutive failures, sends a Telegram alert. Specific errors (expired Blink auth, camera not found, unparseable Gemini response) trigger immediate targeted alerts.

## Architecture

```
Cloud Scheduler
  |
  |-- garage-monitor-day:   */5  7-23 * * *  (every 5 min, 7 AM – 11 PM)
  |-- garage-monitor-night:  */15 0-6  * * *  (every 15 min, midnight – 6 AM)
  |-- garage-report-trigger: 0 21 * * *       (daily at 9 PM)
  |
  v
Google Cloud Function "check-garage" (Python 3.12, gen2, 512 MB, 120s timeout)
  |
  +---> Firestore (collection: garage_monitor)
  |       - doc "state": current_status, last_check_time, last_change_time,
  |                       consecutive_errors, last_reminder_time,
  |                       muted_until, last_final_warning_sent
  |       - doc "blink_credentials": login_attributes (auto-refreshed tokens)
  |       - doc "usage_stats_YYYY_MM": monthly invocation/token/cost counters
  |       - doc "event_*": status change events (30-day TTL via expire_at field)
  |
  +---> Blink API (via blinkpy library, async)
  |       - Auth with saved credentials (no interactive 2FA)
  |       - snap_picture() -> 3s wait -> refresh() -> read cached JPEG
  |       - Updated tokens saved back to Firestore after each call
  |
  +---> Analyzer (configurable via GM_ANALYZER)
  |       Option A — Gemini API (default, via google-genai SDK):
  |       - Model: gemini-2.5-flash (configurable)
  |       - Input: JPEG image + structured Italian prompt
  |       - Output: JSON {status, confidence, reasoning}
  |       - Temperature: 0.0 (deterministic)
  |       - Response MIME type forced to application/json
  |       Option B — TFLite (local, via ai-edge-litert):
  |       - MobileNetV2 transfer learning, 4.4 MB model
  |       - Input: JPEG resized to 224x224
  |       - Output: open probability 0.0-1.0
  |       - Inference: ~50-100ms, no network call, no cost
  |
  +---> Telegram Bot API (via httpx)
  |       - Status change notifications (with photo)
  |       - Night alerts (priority notification when garage opens 0:00–6:59)
  |       - "Still open" reminders (with photo, every 15 min up to 60 min)
  |       - Final warning (one-time escalation after 60 min open)
  |       - Error alerts (text only)
  |       - Daily usage reports (text only)
  |
  +<---- Telegram Webhook (inbound bot commands)
          - Validated via X-Telegram-Bot-Api-Secret-Token header + chat_id
          - Commands: /stato, /foto, /storico, /report, /muto, /smuto

Timezone: Europe/Rome (all scheduler cron expressions)
```

## Cost Breakdown

The only paid service is the **Gemini API** — and it is only used when `GM_ANALYZER=gemini` (the default). When using the **TFLite** local model (`GM_ANALYZER=tflite`), there are zero API costs. All GCP infrastructure components fit within the free tier, Blink cloud API has no per-call charges, and the Telegram Bot API is free.

### Scheduled invocations per month

| Period | Schedule | Calls/day | Calls/month |
|---|---|---|---|
| Daytime (7:00–23:59) | every 5 min | 204 | ~6,120 |
| Nighttime (0:00–6:59) | every 15 min | 28 | ~840 |
| Usage report (21:00) | once daily | 1 | ~30 |
| **Total** | | **233** | **~6,990** |

### Gemini API cost (only when GM_ANALYZER=gemini)

> **Tip:** If you switch to `GM_ANALYZER=tflite`, this entire section does not apply — the local model runs on-device at zero cost. See [TFLite Model](docs/TFLITE_MODEL.md) for details.

Gemini 2.5 Flash paid tier pricing (Standard, per million tokens):

| | Price per 1M tokens |
|---|---|
| Input tokens (text/image) | $0.30 |
| Output tokens (includes thinking tokens) | $2.50 |

Each call sends a JPEG image (~250–800 image tokens depending on resolution) plus the text prompt (~200 tokens). The output is a small JSON object (~30–50 tokens). Note: Gemini 2.5 Flash is a "thinking" model — output token count includes internal reasoning tokens, which can add overhead beyond the visible JSON response.

With ~7,000 scheduled invocations/month, all calling Gemini:
- Input: ~7,000 calls x ~500 tokens = ~3.5M tokens x $0.30/M = **~$1.05**
- Output: ~7,000 calls x ~40 visible tokens (+ thinking overhead) = ~0.28–1.5M tokens x $2.50/M = **~$0.70–3.75**
- **Total Gemini cost: ~$1.75–4.80/month**

The daily usage report tracks actual token consumption so you can monitor real costs.

### GCP infrastructure (free tier)

| Component | Free tier limit | Actual usage | Headroom |
|---|---|---|---|
| Cloud Functions | 2M invocations/month | ~7,000/month | <1% |
| Cloud Scheduler | 3 jobs free | 3 jobs | 100% |
| Firestore reads | 50,000/day | ~470/day | <1% |
| Firestore writes | 20,000/day | ~700/day | <4% |

### Other services (no cost)

| Service | Cost |
|---|---|
| Blink cloud API | Free (included with camera hardware) |
| Telegram Bot API | Free |

### Daily usage report

Every day at 21:00 (Europe/Rome), the system sends a Telegram message with a detailed breakdown of the current month's usage: function invocations, Gemini API calls, input/output token counts, estimated Gemini cost in dollars, Firestore read/write counts, and warnings if any resource exceeds 80% of its free tier quota.

## Prerequisites

- **Google Cloud account** with billing enabled (required even for free tier resources). A new account comes with $300 in credits.
- **Blink Mini camera** set up in the Blink app, pointed at the inside of the garage door. The camera name in the Blink app must match `GM_BLINK_CAMERA_NAME` (default: `Garage`).
- **Telegram** with a bot created via BotFather and the chat ID where notifications should be sent.
- **Python 3.12+** installed locally (for initial Blink setup and testing).
- **`gcloud` CLI** installed and authenticated (`gcloud auth login`).

## Setup

### Step 1 — GCP Project Setup

The `setup_gcp.sh` script enables all required Google Cloud APIs and creates the Firestore database in `eur3` (Europe multi-region):

```bash
./scripts/setup_gcp.sh
```

APIs enabled: Cloud Functions, Cloud Scheduler, Firestore, Cloud Run, Cloud Build, Artifact Registry.

### Step 2 — Gemini API Key (skip if using TFLite)

> If you plan to use `GM_ANALYZER=tflite` (the local MobileNetV2 classifier), you can skip this step entirely — no API key is needed.

1. Go to [Google AI Studio](https://aistudio.google.com/apikey)
2. Create a new API key
3. Note it for the `.env` file in step 4

### Step 3 — Telegram Bot

1. Open Telegram and search for `@BotFather`
2. Send `/newbot` and follow the prompts — note the **bot token**
3. Send any message to your new bot (this creates the chat)
4. Open `https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates` in a browser
5. Find `"chat":{"id":123456789}` in the JSON response — that number is your **chat ID**

### Step 4 — Environment Configuration

```bash
cp .env.example .env
```

Edit `.env` and fill in all values. Every variable uses the `GM_` prefix:

| Variable | Required | Default | Description |
|---|---|---|---|
| `GM_BLINK_USERNAME` | yes | — | Email address of your Blink account |
| `GM_BLINK_PASSWORD` | yes | — | Password of your Blink account |
| `GM_BLINK_CAMERA_NAME` | no | `Garage` | Exact camera name as displayed in the Blink app |
| `GM_ANALYZER` | no | `gemini` | Analyzer backend: `gemini` (cloud AI, paid) or `tflite` (local MobileNetV2, free). See [TFLite Model](docs/TFLITE_MODEL.md) |
| `GM_GEMINI_API_KEY` | conditional | — | API key from Google AI Studio. Required only when `GM_ANALYZER=gemini` |
| `GM_GEMINI_MODEL` | no | `gemini-2.5-flash` | Gemini model identifier. Can be changed to a cheaper/faster model |
| `GM_CONFIDENCE_THRESHOLD` | no | `0.7` | Minimum confidence (0.0–1.0) required to accept a state change. Lower = more sensitive, higher = fewer false positives |
| `GM_TELEGRAM_BOT_TOKEN` | yes | — | Bot token from BotFather |
| `GM_TELEGRAM_CHAT_ID` | yes | — | Numeric chat ID where notifications are sent |
| `GM_GCP_PROJECT_ID` | yes | — | Your Google Cloud project ID |
| `GM_FIRESTORE_COLLECTION` | no | `garage_monitor` | Firestore collection name. Change only if running multiple instances |
| `GM_TELEGRAM_WEBHOOK_SECRET` | no | `""` | Secret token for Telegram webhook validation. Required to enable interactive bot commands. Generate with `openssl rand -hex 32` |
| `GM_GEMINI_COST_ALERT_THRESHOLD` | no | `3.0` | Monthly projected Gemini cost threshold ($) — triggers a warning in the daily report if exceeded |

### Step 5 — Blink Authentication (one-time, interactive)

This script performs the interactive Blink 2FA flow (you'll receive a PIN via email), takes a test snapshot, and saves the authenticated credentials to Firestore so the Cloud Function can use them without 2FA:

```bash
pip install -e .
python scripts/setup_blink.py
```

The script will:
1. Prompt for Blink credentials (or read them from `GM_BLINK_USERNAME` / `GM_BLINK_PASSWORD`)
2. Trigger 2FA — check your email for the PIN code
3. List available cameras and verify the configured camera exists
4. Take a test snapshot and save it to `/tmp/garage_test.jpg`
5. Save authenticated credentials to Firestore (`blink_credentials` document)

If Blink authentication expires in the future, re-run this script. The system will send a Telegram alert when this happens.

### Step 6 — Verify Telegram

Sends a test message to confirm the bot token and chat ID are correct:

```bash
python scripts/test_telegram.py
```

### Step 7 — Deploy

The `deploy.sh` script performs a complete deployment:

```bash
./deploy.sh
```

What it does:
1. Loads variables from `.env`
2. Validates all required variables are set
3. Deploys the Cloud Function (`check-garage`, gen2, `europe-west1`, 512 MB RAM, 120s timeout, HTTP trigger, `--allow-unauthenticated` for Telegram webhook access)
4. Cleans up any legacy scheduler jobs
5. Creates three Cloud Scheduler jobs:
   - `garage-monitor-day`: every 5 minutes from 7:00 to 23:59 (Europe/Rome)
   - `garage-monitor-night`: every 15 minutes from 00:00 to 06:59 (Europe/Rome)
   - `garage-report-trigger`: daily at 21:00 (Europe/Rome), calls `?action=report`
6. Runs a manual test invocation to verify the deployment

All scheduler jobs use OIDC authentication with the default compute service account.

### Step 8 — Telegram Webhook (enables bot commands)

After deploy, register the Telegram webhook so the bot can receive interactive commands:

```bash
source .env
python scripts/setup_telegram_webhook.py <FUNCTION_URL>
```

This tells Telegram to forward all messages sent to the bot to your Cloud Function. The webhook is secured via the `GM_TELEGRAM_WEBHOOK_SECRET` header validation. This step is only needed once (unless the function URL changes).

### Step 9 — Firestore Composite Index (required for /storico)

The `/storico` command queries events with a `where` + `order_by`, which requires a composite index:

```bash
gcloud firestore indexes composite create --collection-group=garage_monitor --field-config field-path=type,order=ascending --field-config field-path=timestamp,order=descending --project=<YOUR_PROJECT_ID>
```

Index creation takes 2–3 minutes. Without this, `/storico` will return an error message (other commands work fine).

### Step 10 — Firestore TTL (optional, recommended)

Enable automatic cleanup of events older than 30 days:

```bash
gcloud firestore fields ttls update expire_at --collection-group=garage_monitor --enable-ttl --project=<YOUR_PROJECT_ID>
```

## Telegram Bot Commands

Once the webhook is set up, you can interact with the bot via Telegram:

| Command | Description |
|---|---|
| `/stato` | Current garage status (open/closed, time since last change, mute status) |
| `/foto` | Take a live snapshot from the camera |
| `/storico` | Last 10 status change events with timestamps and durations |
| `/report` | Monthly usage report (invocations, Gemini cost, Firestore ops) |
| `/muto` | Mute notifications for 2 hours (default) |
| `/muto Nh` | Mute notifications for N hours (max 24h). Example: `/muto 5h` |
| `/smuto` | Re-enable notifications immediately |

## Operations

### View Logs

```bash
gcloud functions logs read check-garage --gen2 --region=europe-west1
```

### Manual Trigger

```bash
# Trigger a garage check
gcloud scheduler jobs run garage-monitor-day --location=europe-west1

# Trigger the daily usage report
gcloud scheduler jobs run garage-report-trigger --location=europe-west1
```

### List Scheduler Jobs

```bash
gcloud scheduler jobs list --location=europe-west1
```

### Re-authenticate Blink

If you receive a Telegram alert about expired Blink authentication:

```bash
python scripts/setup_blink.py
```

## Development

### Install

```bash
pip install -e ".[dev]"
```

### Run Tests

```bash
pytest
```

### Project Structure

```
src/garage_monitor/
    __init__.py
    main.py              — Cloud Function entry point: check_garage(request)
                           Routes between: Telegram webhook, ?action=report, and normal check
                           Orchestrates the full pipeline (Blink -> Analyzer -> Telegram)
    config.py            — Settings class (pydantic-settings, GM_ env prefix)
    blink_client.py      — Async Blink camera client (auth, snapshot, credential refresh)
    gemini_analyzer.py   — Gemini image analysis (structured prompt, JSON response parsing)
    tflite_analyzer.py   — Local TFLite MobileNetV2 classifier (alternative to Gemini)
    firestore_store.py   — Firestore persistence (state, credentials, usage counters, events)
    telegram_notifier.py — Telegram Bot API client (notifications, reminders, alerts, reports, bot responses)
    telegram_handler.py  — Interactive bot command handler (/stato, /foto, /muto, /smuto, /storico, /report)
    models.py            — Data models: GarageStatus (enum), GarageState, GeminiAnalysisResult, UsageStats

model/
    garage_classifier.tflite  — Pre-trained MobileNetV2 classifier (4.4 MB, float16 quantized)
    garage_classifier.keras   — Keras checkpoint for retraining

scripts/
    setup_gcp.sh              — One-time GCP setup (enable APIs, create Firestore DB, TTL policy)
    setup_blink.py            — Interactive Blink 2FA authentication, saves credentials to Firestore
    test_telegram.py          — Verify Telegram bot token and chat ID
    setup_telegram_webhook.py — Register/delete Telegram webhook for bot commands
    train_model.py            — TFLite model training pipeline (MobileNetV2 transfer learning)
    compare_models.py         — Compare multiple Gemini models on test images
    test_heuristic.py         — Heuristic-based garage door detection test

tests/
    test_main.py              — Tests for main orchestration (staleness, night alerts, final warning, events)
    test_telegram_handler.py  — Tests for webhook routing and bot commands
    test_gemini_analyzer.py   — Tests for Gemini response parsing
    test_firestore_store.py   — Tests for Firestore persistence

docs/
    SETUP_BOT.md       — Post-deploy guide for Telegram bot setup and troubleshooting
    TFLITE_MODEL.md    — TFLite model architecture, training data, retraining guide

deploy.sh              — Full deployment script (Cloud Function + Scheduler jobs)
pyproject.toml         — Python project config (dependencies, pytest settings)
.env.example           — Template for environment variables
```

## Troubleshooting

| Symptom | Likely Cause | Solution |
|---|---|---|
| Telegram alert: "Autenticazione Blink scaduta" | Blink tokens have expired (happens periodically) | Re-run `python scripts/setup_blink.py` |
| Telegram alert: "Camera non trovata" | `GM_BLINK_CAMERA_NAME` doesn't match the Blink app | Check the exact camera name in the Blink app, update `.env`, re-deploy |
| Telegram alert: "Gemini ha risposto in modo inatteso" | Gemini returned malformed JSON | Check logs — may be a transient API issue. If persistent, the prompt may need adjustment |
| Telegram alert: "N errori consecutivi" | 3+ consecutive failures of any kind | Check logs with `gcloud functions logs read` for the root cause |
| No notifications at all | Bot token or chat ID incorrect | Run `python scripts/test_telegram.py` to verify |
| Confidence always below threshold | Poor lighting or camera angle | Ensure the camera is inside the garage, centered on the door. Check IR mode at night |
| Cloud Function timeout | Blink API slow to respond | The function has a 120s timeout. If persistent, check Blink service status |
| TFLite model misclassifies | Camera moved or new lighting condition | Retrain the model — see [TFLite Model](docs/TFLITE_MODEL.md) retraining instructions |
| Bot commands not working | Webhook not registered or secret mismatch | Run `setup_telegram_webhook.py` with correct `GM_TELEGRAM_WEBHOOK_SECRET` |
| `/storico` returns error | Missing Firestore composite index | Create the index — see Step 9 above |
| Night false positives | IR image misclassified as "open" | Gemini prompt is tuned for this camera; adjust confidence threshold if needed |
