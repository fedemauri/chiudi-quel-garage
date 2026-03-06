# Chiudi Quel Garage!

An automated garage door monitoring system that uses a **Blink Mini** indoor camera and **Google Gemini AI** vision to detect whether a garage door is open or closed. When the state changes, the system sends a **Telegram notification with a photo**. If the door stays open, periodic reminders are sent. Everything runs serverless on **Google Cloud Platform**.

## How It Works

The system is built around a single Google Cloud Function that executes the following pipeline on every invocation:

1. **Read state** — Fetches the current garage status and Blink credentials from Firestore.
2. **Capture snapshot** — Authenticates with the Blink API using stored credentials (no 2FA required after initial setup) and commands the camera to take a fresh JPEG snapshot.
3. **Image deduplication** — Computes an MD5 hash of the snapshot and compares it with the previously stored hash. If the image is byte-for-byte identical (common at night when nothing moves), the Gemini API call is skipped entirely to save cost.
4. **AI analysis** — Sends the JPEG image to Gemini along with a detailed prompt describing the specific camera's perspective (fixed indoor camera looking at a sectional garage door). Gemini returns a structured JSON response with `status` (open/closed), `confidence` (0.0–1.0), and `reasoning` (short text in Italian).
5. **State change detection** — If the AI classification differs from the stored state *and* the confidence exceeds the configured threshold (default 0.7), the system records a state change.
6. **Telegram notification** — On state change, sends a Telegram message with the snapshot photo, the old and new status, confidence score, and Gemini's reasoning. On first run, sends a "monitor started" confirmation.
7. **Open-door reminders** — If the door has been open for more than 15 minutes, sends a reminder with photo every 15 minutes, up to a maximum of 60 minutes.
8. **Persist state** — Saves the updated garage state, refreshed Blink credentials, and increments usage counters in Firestore.
9. **Error handling** — Tracks consecutive errors. After 3 consecutive failures, sends a Telegram alert. Specific errors (expired Blink auth, camera not found, unparseable Gemini response) trigger immediate targeted alerts.

## Architecture

```
Cloud Scheduler
  |
  |-- garage-monitor-day:   */5  7-23 * * *  (every 5 min, 7 AM – 11 PM)
  |-- garage-monitor-night:  */15 0-6  * * *  (every 15 min, midnight – 6 AM)
  |-- garage-report-trigger: 0 21 * * *       (daily at 9 PM)
  |
  v
Google Cloud Function "check-garage" (Python 3.12, gen2, 256 MB, 120s timeout)
  |
  +---> Firestore (collection: garage_monitor)
  |       - doc "state": current_status, last_check_time, last_change_time,
  |                       consecutive_errors, last_reminder_time, last_image_hash
  |       - doc "blink_credentials": login_attributes (auto-refreshed tokens)
  |       - doc "usage_stats_YYYY_MM": monthly invocation/token/cost counters
  |
  +---> Blink API (via blinkpy library, async)
  |       - Auth with saved credentials (no interactive 2FA)
  |       - snap_picture() -> 3s wait -> refresh() -> read cached JPEG
  |       - Updated tokens saved back to Firestore after each call
  |
  +---> Gemini API (via google-genai SDK)
  |       - Model: gemini-2.5-flash (configurable)
  |       - Input: JPEG image + structured Italian prompt
  |       - Output: JSON {status, confidence, reasoning}
  |       - Temperature: 0.1 (near-deterministic)
  |       - Response MIME type forced to application/json
  |       - Skipped when image hash matches previous (cost optimization)
  |
  +---> Telegram Bot API (via httpx)
          - Status change notifications (with photo)
          - "Still open" reminders (with photo)
          - Error alerts (text only)
          - Daily usage reports (text only)

Timezone: Europe/Rome (all scheduler cron expressions)
```

## Cost Breakdown

The only paid service is the **Gemini API**. All GCP infrastructure components fit within the free tier, Blink cloud API has no per-call charges, and the Telegram Bot API is free.

### Scheduled invocations per month

| Period | Schedule | Calls/day | Calls/month |
|---|---|---|---|
| Daytime (7:00–23:59) | every 5 min | 204 | ~6,120 |
| Nighttime (0:00–6:59) | every 15 min | 28 | ~840 |
| Usage report (21:00) | once daily | 1 | ~30 |
| **Total** | | **233** | **~6,990** |

### Gemini API cost (the only paid component)

Not every invocation calls Gemini. The system computes an MD5 hash of each snapshot and skips the API call when the image is identical to the previous one. This is common at night (static IR image of a closed door) and during long periods with no movement. In practice, the skip rate is typically 40–70%, reducing actual Gemini calls to roughly **2,000–4,000/month**.

Gemini 2.5 Flash paid tier pricing (Standard, per million tokens):

| | Price per 1M tokens |
|---|---|
| Input tokens (text/image) | $0.30 |
| Output tokens (includes thinking tokens) | $2.50 |

Each call sends a JPEG image (~250–800 image tokens depending on resolution) plus the text prompt (~200 tokens). The output is a small JSON object (~30–50 tokens). Note: Gemini 2.5 Flash is a "thinking" model — output token count includes internal reasoning tokens, which can add overhead beyond the visible JSON response.

Estimated monthly cost with ~3,000 actual Gemini calls:
- Input: ~3,000 calls x ~500 tokens = ~1.5M tokens x $0.30/M = **~$0.45**
- Output: ~3,000 calls x ~40 visible tokens (+ thinking overhead) = ~0.12–0.5M tokens x $2.50/M = **~$0.30–1.25**
- **Total Gemini cost: ~$0.75–1.70/month**

With aggressive image dedup (e.g., garage door rarely moves, fewer actual calls): can be lower. The daily usage report tracks actual token consumption so you can monitor real costs.

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

### Step 2 — Gemini API Key

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
| `GM_GEMINI_API_KEY` | yes | — | API key from Google AI Studio |
| `GM_GEMINI_MODEL` | no | `gemini-2.5-flash` | Gemini model identifier. Can be changed to a cheaper/faster model |
| `GM_CONFIDENCE_THRESHOLD` | no | `0.7` | Minimum confidence (0.0–1.0) required to accept a state change. Lower = more sensitive, higher = fewer false positives |
| `GM_TELEGRAM_BOT_TOKEN` | yes | — | Bot token from BotFather |
| `GM_TELEGRAM_CHAT_ID` | yes | — | Numeric chat ID where notifications are sent |
| `GM_GCP_PROJECT_ID` | yes | — | Your Google Cloud project ID |
| `GM_FIRESTORE_COLLECTION` | no | `garage_monitor` | Firestore collection name. Change only if running multiple instances |

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
3. Deploys the Cloud Function (`check-garage`, gen2, `europe-west1`, 256 MB RAM, 120s timeout, HTTP trigger, no unauthenticated access)
4. Cleans up any legacy scheduler jobs
5. Creates three Cloud Scheduler jobs:
   - `garage-monitor-day`: every 5 minutes from 7:00 to 23:59 (Europe/Rome)
   - `garage-monitor-night`: every 15 minutes from 00:00 to 06:59 (Europe/Rome)
   - `garage-report-trigger`: daily at 21:00 (Europe/Rome), calls `?action=report`
6. Runs a manual test invocation to verify the deployment

All scheduler jobs use OIDC authentication with the default compute service account.

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
                           Handles ?action=report for usage reports
                           Orchestrates the full pipeline (Blink -> Gemini -> Telegram)
    config.py            — Settings class (pydantic-settings, GM_ env prefix)
    blink_client.py      — Async Blink camera client (auth, snapshot, credential refresh)
    gemini_analyzer.py   — Gemini image analysis (structured prompt, JSON response parsing)
    firestore_store.py   — Firestore persistence (state, credentials, usage counters)
    telegram_notifier.py — Telegram Bot API client (notifications, reminders, error alerts, usage reports)
    models.py            — Data models: GarageStatus (enum), GarageState, GeminiAnalysisResult, UsageStats

scripts/
    setup_gcp.sh         — One-time GCP setup (enable APIs, create Firestore DB)
    setup_blink.py       — Interactive Blink 2FA authentication, saves credentials to Firestore
    test_telegram.py     — Verify Telegram bot token and chat ID

tests/
    test_main.py         — Tests for the main orchestration logic
    test_gemini_analyzer.py — Tests for Gemini response parsing
    test_firestore_store.py — Tests for Firestore persistence

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
| Gemini never called (logs show "skip Gemini") | Image dedup — camera returns identical bytes | Normal behavior when nothing changes. Will resume when the scene changes |
| Cloud Function timeout | Blink API slow to respond | The function has a 120s timeout. If persistent, check Blink service status |
