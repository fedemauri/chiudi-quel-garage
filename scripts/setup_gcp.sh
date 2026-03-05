#!/usr/bin/env bash
set -euo pipefail

echo "=== Garage Monitor - Setup GCP ==="
echo ""

# Check gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo "ERROR: gcloud CLI non trovato. Installalo da https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Project ID
read -rp "GCP Project ID (lascia vuoto per usare il progetto corrente): " PROJECT_ID
if [ -z "$PROJECT_ID" ]; then
    PROJECT_ID=$(gcloud config get-value project 2>/dev/null)
    echo "Uso progetto corrente: $PROJECT_ID"
fi

if [ -z "$PROJECT_ID" ]; then
    echo "ERROR: Nessun progetto GCP configurato. Specifica un Project ID."
    exit 1
fi

gcloud config set project "$PROJECT_ID"

echo ""
echo ">>> Abilito le API necessarie..."
gcloud services enable \
    cloudfunctions.googleapis.com \
    cloudscheduler.googleapis.com \
    firestore.googleapis.com \
    run.googleapis.com \
    cloudbuild.googleapis.com \
    artifactregistry.googleapis.com

echo ""
echo ">>> Creo database Firestore (Native mode)..."
gcloud firestore databases create --location=eur3 2>/dev/null || \
    echo "Database Firestore gia' esistente, continuo."

echo ""
echo "=== Setup GCP completato! ==="
echo ""
echo "Prossimi passi:"
echo "  1. Crea API key Gemini: https://aistudio.google.com/apikey"
echo "  2. Crea bot Telegram: cerca @BotFather su Telegram"
echo "  3. Copia .env.example in .env e compila i valori"
echo "  4. Esegui: python scripts/setup_blink.py"
echo "  5. Esegui: ./deploy.sh"
