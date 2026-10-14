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

# GM_BLINK_PASSWORD is only used by scripts/setup_blink.py: it is not sent to the function.
# GM_SCHEDULER_TOKEN: without it anyone with the function URL can trigger checks (Blink snapshots).
if [ -z "${GM_SCHEDULER_TOKEN:-}" ]; then
    echo "WARNING: GM_SCHEDULER_TOKEN non impostata: chiunque conosca l'URL puo' far partire un controllo"
fi

# Validate required vars
REQUIRED_VARS=(
    GM_BLINK_USERNAME GM_BLINK_CAMERA_NAME
    GM_TELEGRAM_BOT_TOKEN GM_TELEGRAM_CHAT_ID
    GM_GCP_PROJECT_ID
)

# Gemini API key required only when using gemini analyzer (default)
if [ "${GM_ANALYZER:-gemini}" = "gemini" ] && [ -z "${GM_GEMINI_API_KEY:-}" ]; then
    echo "ERROR: GM_GEMINI_API_KEY richiesta con analyzer=gemini. Impostala in .env o usa GM_ANALYZER=tflite"
    exit 1
fi

for var in "${REQUIRED_VARS[@]}"; do
    if [ -z "${!var:-}" ]; then
        echo "ERROR: Variabile $var non impostata. Compila il file .env"
        exit 1
    fi
done

PROJECT_ID="$GM_GCP_PROJECT_ID"
REGION="${GM_GCP_REGION:-europe-west1}"
FUNCTION_NAME="check-garage"
SCHEDULER_JOB="garage-monitor"

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
    --memory=512Mi \
    --timeout=120s \
    --set-env-vars="GM_ANALYZER=${GM_ANALYZER:-gemini},GM_BLINK_USERNAME=$GM_BLINK_USERNAME,GM_BLINK_CAMERA_NAME=${GM_BLINK_CAMERA_NAME:-Garage},GM_GEMINI_API_KEY=${GM_GEMINI_API_KEY:-},GM_GEMINI_MODEL=${GM_GEMINI_MODEL:-gemini-2.5-flash},GM_CONFIDENCE_THRESHOLD=${GM_CONFIDENCE_THRESHOLD:-0.7},GM_TELEGRAM_BOT_TOKEN=$GM_TELEGRAM_BOT_TOKEN,GM_TELEGRAM_CHAT_ID=$GM_TELEGRAM_CHAT_ID,GM_GCP_PROJECT_ID=$PROJECT_ID,GM_FIRESTORE_COLLECTION=${GM_FIRESTORE_COLLECTION:-garage_monitor},GM_TELEGRAM_WEBHOOK_SECRET=${GM_TELEGRAM_WEBHOOK_SECRET:-},GM_GEMINI_COST_ALERT_THRESHOLD=${GM_GEMINI_COST_ALERT_THRESHOLD:-3.0},GM_STATUS_TOKEN=${GM_STATUS_TOKEN:-},GM_SCHEDULER_TOKEN=${GM_SCHEDULER_TOKEN:-}"

FUNCTION_URL=$(gcloud functions describe "$FUNCTION_NAME" --gen2 --region="$REGION" --format='value(serviceConfig.uri)')
echo "Function URL: $FUNCTION_URL"

echo ""
echo ">>> Creo/aggiorno Cloud Scheduler job (ogni 5 min, 24/7)..."
SA_EMAIL=$(gcloud iam service-accounts list --filter="displayName:Default compute service account" --format='value(email)' | head -1)
if [ -z "$SA_EMAIL" ]; then
    SA_EMAIL="$(gcloud config get-value account 2>/dev/null)"
fi

# Pulizia vecchi job (day, night, report, legacy)
gcloud scheduler jobs delete "garage-monitor-trigger" --location="$REGION" --quiet 2>/dev/null || true
gcloud scheduler jobs delete "garage-monitor-day" --location="$REGION" --quiet 2>/dev/null || true
gcloud scheduler jobs delete "garage-monitor-night" --location="$REGION" --quiet 2>/dev/null || true
gcloud scheduler jobs delete "garage-report-trigger" --location="$REGION" --quiet 2>/dev/null || true
gcloud scheduler jobs delete "$SCHEDULER_JOB" --location="$REGION" --quiet 2>/dev/null || true

gcloud scheduler jobs create http "$SCHEDULER_JOB" \
    --location="$REGION" \
    --schedule="*/5 * * * *" \
    --time-zone="Europe/Rome" \
    --uri="$FUNCTION_URL" \
    --http-method=POST \
    --oidc-service-account-email="$SA_EMAIL" \
    --oidc-token-audience="$FUNCTION_URL" \
    ${GM_SCHEDULER_TOKEN:+--headers="X-Scheduler-Token=$GM_SCHEDULER_TOKEN"}

echo ""
echo ">>> Test: esecuzione manuale del job Scheduler (stesso percorso, con token)..."
gcloud scheduler jobs run "$SCHEDULER_JOB" --location="$REGION" || echo "Test fallito, controlla i log."
if [ -n "${GM_SCHEDULER_TOKEN:-}" ]; then
    CODE=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$FUNCTION_URL")
    echo ">>> Chiamata senza token: HTTP $CODE (atteso 403)"
fi

echo ""
echo "=== Deploy completato! ==="
echo ""
echo "Comandi utili:"
echo "  Logs:    gcloud functions logs read $FUNCTION_NAME --gen2 --region=$REGION"
echo "  Trigger: gcloud scheduler jobs run $SCHEDULER_JOB --location=$REGION"
echo "  Jobs:    gcloud scheduler jobs list --location=$REGION"
echo "  Webhook: python scripts/setup_telegram_webhook.py $FUNCTION_URL"
