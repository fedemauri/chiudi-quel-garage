# Telegram Bot Setup Guide

This guide covers the post-deploy configuration needed to enable the interactive Telegram bot commands.

## Prerequisites

- The project is already deployed as a Cloud Function (see main README)
- You have a working `.env` file with all base variables configured
- The Telegram bot is already created via BotFather

## 1. New Environment Variables

Add the following to your `.env` file:

```bash
# Webhook secret (REQUIRED for bot commands)
# Generate a random secret:
#   python3 -c "import secrets; print(secrets.token_urlsafe(32))"
GM_TELEGRAM_WEBHOOK_SECRET=<your-random-secret-string>

# Gemini cost alert threshold in $/month (optional, default: 3.0)
GM_GEMINI_COST_ALERT_THRESHOLD=3.0
```

## 2. Deploy the Cloud Function

```bash
./deploy.sh
```

**Important change**: The function is now deployed with `--allow-unauthenticated` to receive Telegram webhook calls. Security is enforced by:

- **Webhook secret**: The `X-Telegram-Bot-Api-Secret-Token` header is validated against `GM_TELEGRAM_WEBHOOK_SECRET`
- **Chat ID validation**: Only messages from the configured `GM_TELEGRAM_CHAT_ID` are processed
- **Scheduler jobs**: Continue to work normally (they don't send JSON with a "message" field, so they bypass webhook routing)

## 3. Register the Telegram Webhook

After deploying, register the Cloud Function URL as the bot's webhook:

```bash
# Load environment variables
source .env

# Get the function URL
FUNCTION_URL=$(gcloud functions describe check-garage --gen2 --region=europe-west1 --format='value(serviceConfig.uri)')

# Register the webhook
python scripts/setup_telegram_webhook.py "$FUNCTION_URL"
```

To verify the webhook is active:

```bash
curl "https://api.telegram.org/bot${GM_TELEGRAM_BOT_TOKEN}/getWebhookInfo"
```

To remove the webhook (revert to push-only mode):

```bash
python scripts/setup_telegram_webhook.py --delete
```

## 4. Configure Firestore TTL for Events

Run once to enable automatic cleanup of event history after 30 days:

```bash
gcloud firestore fields ttls update expire_at \
  --collection-group=garage_monitor \
  --enable-ttl \
  --project=$GM_GCP_PROJECT_ID
```

Verify it's active:

```bash
gcloud firestore fields ttls list --project=$GM_GCP_PROJECT_ID
```

Firestore TTL deletes expired documents automatically at no cost (TTL deletes don't count as write operations).

## 5. Available Bot Commands

Send these commands to your Telegram bot:

| Command | Description |
|---|---|
| `/stato` | Current garage status + time since last change |
| `/foto` | Take a live photo from the camera |
| `/report` | Send the monthly usage/cost report |
| `/muto 2h` | Mute reminders for N hours (default 2h, max 24h) |
| `/smuto` | Unmute — re-enable notifications immediately |
| `/storico` | Show the last 10 open/close events |

## 6. Register Commands in Bot Menu (Optional)

For command autocompletion in the Telegram UI:

```bash
curl -X POST "https://api.telegram.org/bot${GM_TELEGRAM_BOT_TOKEN}/setMyCommands" \
  -H "Content-Type: application/json" \
  -d '{
    "commands": [
      {"command": "stato", "description": "Stato attuale del garage"},
      {"command": "foto", "description": "Scatta foto dal vivo"},
      {"command": "report", "description": "Report utilizzo risorse"},
      {"command": "muto", "description": "Silenzia notifiche (es. /muto 2h)"},
      {"command": "smuto", "description": "Riattiva notifiche"},
      {"command": "storico", "description": "Ultimi 10 eventi"}
    ]
  }'
```

## Troubleshooting

**Bot doesn't respond to commands:**
- Check webhook is registered: `getWebhookInfo` should show your function URL
- Verify `GM_TELEGRAM_WEBHOOK_SECRET` matches between `.env` and the registered webhook
- Check function logs: `gcloud functions logs read check-garage --gen2 --region=europe-west1`

**403 on webhook:**
- Empty `GM_TELEGRAM_WEBHOOK_SECRET` in config disables the webhook (returns 403)
- Mismatched secret between Telegram and your config returns 403

**Commands return "Nessuno stato disponibile":**
- The monitor hasn't run its first check yet. Trigger manually: `gcloud scheduler jobs run garage-monitor-day --location=europe-west1`
