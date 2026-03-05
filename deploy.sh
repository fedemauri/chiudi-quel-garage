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
SCHEDULER_JOB="garage-monitor-trigger"

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
    --allow-unauthenticated=false \
    --memory=256Mi \
    --timeout=120s \
    --set-env-vars="GM_BLINK_USERNAME=$GM_BLINK_USERNAME,GM_BLINK_PASSWORD=$GM_BLINK_PASSWORD,GM_BLINK_CAMERA_NAME=${GM_BLINK_CAMERA_NAME:-Garage},GM_GEMINI_API_KEY=$GM_GEMINI_API_KEY,GM_GEMINI_MODEL=${GM_GEMINI_MODEL:-gemini-2.5-flash-lite},GM_CONFIDENCE_THRESHOLD=${GM_CONFIDENCE_THRESHOLD:-0.7},GM_TELEGRAM_BOT_TOKEN=$GM_TELEGRAM_BOT_TOKEN,GM_TELEGRAM_CHAT_ID=$GM_TELEGRAM_CHAT_ID,GM_GCP_PROJECT_ID=$PROJECT_ID,GM_FIRESTORE_COLLECTION=${GM_FIRESTORE_COLLECTION:-garage_monitor}"

FUNCTION_URL=$(gcloud functions describe "$FUNCTION_NAME" --gen2 --region="$REGION" --format='value(serviceConfig.uri)')
echo "Function URL: $FUNCTION_URL"

echo ""
echo ">>> Creo/aggiorno Cloud Scheduler job (ogni 5 minuti)..."
SA_EMAIL=$(gcloud iam service-accounts list --filter="displayName:Default compute service account" --format='value(email)' | head -1)
if [ -z "$SA_EMAIL" ]; then
    SA_EMAIL="$(gcloud config get-value account 2>/dev/null)"
fi

gcloud scheduler jobs delete "$SCHEDULER_JOB" --location="$REGION" --quiet 2>/dev/null || true

gcloud scheduler jobs create http "$SCHEDULER_JOB" \
    --location="$REGION" \
    --schedule="*/5 * * * *" \
    --uri="$FUNCTION_URL" \
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
echo "  Trigger: gcloud scheduler jobs run $SCHEDULER_JOB --location=$REGION"
