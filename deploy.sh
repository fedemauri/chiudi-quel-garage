#!/usr/bin/env bash
set -euo pipefail

echo "=== Garage Monitor - Deploy ==="
echo ""

# Load .env if exists
if [ -f .env ]; then
    echo ">>> Carico variabili da .env..."
    set -a
    source .env
    set +a
fi

# Validate required vars
REQUIRED_VARS=(
    GM_BLINK_USERNAME GM_BLINK_PASSWORD GM_BLINK_CAMERA_NAME
    GM_GEMINI_API_KEY GM_TELEGRAM_BOT_TOKEN GM_TELEGRAM_CHAT_ID
    GM_GCP_PROJECT_ID
)

for var in "${REQUIRED_VARS[@]}"; do
    if [ -z "${!var:-}" ]; then
        echo "ERROR: Variabile $var non impostata. Compila il file .env"
        exit 1
    fi
done

PROJECT_ID="$GM_GCP_PROJECT_ID"
REGION="${GM_GCP_REGION:-europe-west1}"
FUNCTION_NAME="check-garage"
SCHEDULER_JOB_DAY="garage-monitor-day"
SCHEDULER_JOB_NIGHT="garage-monitor-night"

gcloud config set project "$PROJECT_ID"

echo ""
echo ">>> Deploy Cloud Function (gen2)..."
gcloud functions deploy "$FUNCTION_NAME" \
    --gen2 \
    --region="$REGION" \
    --runtime=python312 \
    --source=. \
    --entry-point=check_garage \
    --trigger-http \
    --allow-unauthenticated \
    --memory=256Mi \
    --timeout=120s \
    --set-env-vars="GM_BLINK_USERNAME=$GM_BLINK_USERNAME,GM_BLINK_PASSWORD=$GM_BLINK_PASSWORD,GM_BLINK_CAMERA_NAME=${GM_BLINK_CAMERA_NAME:-Garage},GM_GEMINI_API_KEY=$GM_GEMINI_API_KEY,GM_GEMINI_MODEL=${GM_GEMINI_MODEL:-gemini-2.5-flash},GM_CONFIDENCE_THRESHOLD=${GM_CONFIDENCE_THRESHOLD:-0.7},GM_TELEGRAM_BOT_TOKEN=$GM_TELEGRAM_BOT_TOKEN,GM_TELEGRAM_CHAT_ID=$GM_TELEGRAM_CHAT_ID,GM_GCP_PROJECT_ID=$PROJECT_ID,GM_FIRESTORE_COLLECTION=${GM_FIRESTORE_COLLECTION:-garage_monitor},GM_TELEGRAM_WEBHOOK_SECRET=${GM_TELEGRAM_WEBHOOK_SECRET:-},GM_GEMINI_COST_ALERT_THRESHOLD=${GM_GEMINI_COST_ALERT_THRESHOLD:-3.0}"

FUNCTION_URL=$(gcloud functions describe "$FUNCTION_NAME" --gen2 --region="$REGION" --format='value(serviceConfig.uri)')
echo "Function URL: $FUNCTION_URL"

echo ""
echo ">>> Creo/aggiorno Cloud Scheduler jobs (giorno ogni 5 min, notte ogni 15 min)..."
SA_EMAIL=$(gcloud iam service-accounts list --filter="displayName:Default compute service account" --format='value(email)' | head -1)
if [ -z "$SA_EMAIL" ]; then
    SA_EMAIL="$(gcloud config get-value account 2>/dev/null)"
fi

# Pulizia vecchio job singolo + nuovi job
gcloud scheduler jobs delete "garage-monitor-trigger" --location="$REGION" --quiet 2>/dev/null || true
gcloud scheduler jobs delete "$SCHEDULER_JOB_DAY" --location="$REGION" --quiet 2>/dev/null || true
gcloud scheduler jobs delete "$SCHEDULER_JOB_NIGHT" --location="$REGION" --quiet 2>/dev/null || true

gcloud scheduler jobs create http "$SCHEDULER_JOB_DAY" \
    --location="$REGION" \
    --schedule="*/5 7-23 * * *" \
    --time-zone="Europe/Rome" \
    --uri="$FUNCTION_URL" \
    --http-method=POST \
    --oidc-service-account-email="$SA_EMAIL" \
    --oidc-token-audience="$FUNCTION_URL"

gcloud scheduler jobs create http "$SCHEDULER_JOB_NIGHT" \
    --location="$REGION" \
    --schedule="*/15 0-6 * * *" \
    --time-zone="Europe/Rome" \
    --uri="$FUNCTION_URL" \
    --http-method=POST \
    --oidc-service-account-email="$SA_EMAIL" \
    --oidc-token-audience="$FUNCTION_URL"

REPORT_JOB="garage-report-trigger"
echo ""
echo ">>> Creo/aggiorno Cloud Scheduler job report (21:00)..."
gcloud scheduler jobs delete "$REPORT_JOB" --location="$REGION" --quiet 2>/dev/null || true

gcloud scheduler jobs create http "$REPORT_JOB" \
    --location="$REGION" \
    --schedule="0 21 * * *" \
    --time-zone="Europe/Rome" \
    --uri="${FUNCTION_URL}?action=report" \
    --http-method=POST \
    --oidc-service-account-email="$SA_EMAIL" \
    --oidc-token-audience="$FUNCTION_URL"

echo ""
echo ">>> Test invocazione manuale..."
gcloud functions call "$FUNCTION_NAME" --gen2 --region="$REGION" || echo "Test fallito, controlla i log."

echo ""
echo "=== Deploy completato! ==="
echo ""
echo "Comandi utili:"
echo "  Logs:    gcloud functions logs read $FUNCTION_NAME --gen2 --region=$REGION"
echo "  Trigger: gcloud scheduler jobs run $SCHEDULER_JOB_DAY --location=$REGION"
echo "  Jobs:    gcloud scheduler jobs list --location=$REGION"
echo "  Webhook: python scripts/setup_telegram_webhook.py $FUNCTION_URL"
