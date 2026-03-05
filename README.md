# Chiudi Quel Garage!

Sistema automatico di monitoraggio stato garage con telecamera Blink Mini e AI Gemini.

Cattura periodicamente uno snapshot dalla telecamera, lo analizza con Gemini 2.5 Flash-Lite per determinare se il box e' aperto o chiuso, e invia una notifica Telegram quando lo stato cambia.

## Architettura

```
Cloud Scheduler (ogni 5 min)
        |
        v
Google Cloud Function (Python 3.12)
        |
        +---> Firestore: leggi stato + credenziali Blink
        +---> API Blink: autentica + cattura snapshot
        +---> Gemini 2.5 Flash-Lite: analizza immagine
        +---> Se stato cambiato: Telegram Bot -> notifica con foto
        +---> Firestore: salva nuovo stato
```

## Costi

| Componente | Costo |
|-----------|-------|
| Cloud Function | $0 (free tier) |
| Cloud Scheduler | $0 (3 job gratis) |
| Firestore | $0 (free tier) |
| Gemini API | ~$0.04/mese |
| **TOTALE** | **~$0.04/mese** |

## Setup

### Prerequisiti

- Account Google Cloud (con carta, free tier)
- Account Blink con telecamera configurata
- Telegram installato
- Python 3.12+
- `gcloud` CLI installato

### 1. API key Gemini

Vai su https://aistudio.google.com/apikey e crea una API key.

### 2. Bot Telegram

1. Apri Telegram, cerca `@BotFather`
2. Invia `/newbot`, segui le istruzioni, annota il **TOKEN**
3. Invia un messaggio qualsiasi al tuo nuovo bot
4. Visita `https://api.telegram.org/bot<TOKEN>/getUpdates` e annota il `chat_id`

### 3. Setup GCP

```bash
./scripts/setup_gcp.sh
```

### 4. Configurazione

```bash
cp .env.example .env
# Compila tutti i valori in .env
```

### 5. Setup Blink (una tantum)

```bash
pip install -e .
python scripts/setup_blink.py
```

Questo script gestisce l'autenticazione 2FA e salva le credenziali su Firestore.

### 6. Test Telegram

```bash
python scripts/test_telegram.py
```

### 7. Deploy

```bash
./deploy.sh
```

## Verifica

```bash
# Controlla i log
gcloud functions logs read check-garage --gen2 --region=europe-west1

# Trigger manuale
gcloud scheduler jobs run garage-monitor-trigger --location=europe-west1
```

## Test

```bash
pip install -e ".[dev]"
pytest
```

## Troubleshooting

| Problema | Soluzione |
|---------|----------|
| "Blink authentication expired" | Riesegui `python scripts/setup_blink.py` |
| Notifiche non arrivano | Verifica `GM_TELEGRAM_CHAT_ID` con `scripts/test_telegram.py` |
| Confidence sempre bassa | Verifica illuminazione garage, posizione telecamera |
| Errori consecutivi | Controlla i log con `gcloud functions logs read` |
