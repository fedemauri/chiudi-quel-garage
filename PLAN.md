# Garage Monitor - Rilevamento stato box con Blink Mini + Gemini AI

## Contesto

Telecamera Blink Mini installata nel garage con porta sezionale. Sistema automatico che:
- Cattura periodicamente un'immagine dalla telecamera
- Usa Gemini 2.5 Flash-Lite per classificare se il box e' aperto o chiuso
- Invia notifica Telegram SOLO quando lo stato cambia

Gira **gratis** su Google Cloud Functions (serverless).

---

## Architettura

```
Cloud Scheduler (ogni 5 min)
        |
        v
Google Cloud Function (Python 3.12)
        |
        +---> Firestore: leggi stato + credenziali Blink
        |
        +---> API Blink: autentica + cattura snapshot
        |
        +---> Gemini 2.5 Flash-Lite: analizza immagine (aperto/chiuso?)
        |
        +---> Confronta con stato precedente
        |
        +---> Se cambiato: Telegram Bot -> notifica con foto
        |
        +---> Firestore: salva nuovo stato + aggiorna credenziali Blink
```

### Perche' serverless (non un loop persistente)
- Cloud Functions: 2M invocazioni/mese gratis, 400K GB-sec gratis
- Cloud Scheduler: 3 job gratis
- Firestore: 50K reads/day, 20K writes/day, 1 GiB storage gratis
- 288 invocazioni/giorno (ogni 5 min) = ~8640/mese << limiti free tier
- Ogni invocazione dura ~10-30 sec con 256MB RAM = ~75 GB-sec/giorno << limiti

---

## Costi stimati

| Componente | Costo |
|-----------|-------|
| Cloud Function | $0 (free tier) |
| Cloud Scheduler | $0 (3 job gratis) |
| Firestore | $0 (free tier) |
| Gemini API | ~$0.04/mese (288 img/giorno * $0.00013) |
| Telegram | $0 |
| **TOTALE** | **~$0.04/mese** |

---

## Struttura progetto

```
garage-monitor/
├── README.md
├── pyproject.toml
├── .env.example
├── .gcloudignore
├── deploy.sh                      # Script deploy su GCP
│
├── src/
│   └── garage_monitor/
│       ├── __init__.py
│       ├── main.py                # Entry point Cloud Function
│       ├── config.py              # Settings da env vars
│       ├── blink_client.py        # Auth Blink + snapshot
│       ├── gemini_analyzer.py     # Analisi immagine con Gemini
│       ├── telegram_notifier.py   # Invio notifiche
│       ├── firestore_store.py     # Persistenza stato + credenziali Blink
│       └── models.py              # Dataclass condivise
│
├── scripts/
│   ├── setup_blink.py             # Setup interattivo 2FA (locale, una tantum)
│   ├── setup_gcp.sh               # Crea progetto GCP, abilita API, deploy
│   └── test_telegram.py           # Verifica bot Telegram
│
└── tests/
    ├── test_gemini_analyzer.py
    ├── test_firestore_store.py
    └── test_main.py
```

---

## Moduli dettagliati

### 1. `config.py` - Configurazione

Carica da variabili d'ambiente (settate come secrets in Cloud Functions):

```
GM_BLINK_USERNAME          # Email account Blink
GM_BLINK_PASSWORD          # Password Blink
GM_BLINK_CAMERA_NAME       # Nome telecamera (es. "Garage")
GM_GEMINI_API_KEY          # API key Google AI Studio
GM_GEMINI_MODEL            # Default: gemini-2.5-flash-lite
GM_CONFIDENCE_THRESHOLD    # Default: 0.7
GM_TELEGRAM_BOT_TOKEN      # Token da BotFather
GM_TELEGRAM_CHAT_ID        # Chat ID destinatario
GM_GCP_PROJECT_ID          # ID progetto GCP
GM_FIRESTORE_COLLECTION    # Default: garage_monitor
```

### 2. `models.py` - Tipi condivisi

- `GarageStatus` (enum): OPEN, CLOSED, UNKNOWN
- `GeminiAnalysisResult`: status, confidence (float 0-1), reasoning (str)
- `GarageState`: current_status, last_check_time, last_change_time, consecutive_errors

### 3. `blink_client.py` - Modulo Blink

- `connect(credentials_json)`: autentica usando credenziali salvate da Firestore
- `take_snapshot()`: cattura immagine, ritorna bytes JPEG
- `get_updated_credentials()`: ritorna credenziali aggiornate (token refreshato) da salvare su Firestore
- Gestione: se auth fallisce, solleva eccezione specifica `BlinkAuthExpiredError`

### 4. `gemini_analyzer.py` - Analisi immagine

- Usa `google-genai` SDK (NON google-generativeai, deprecata)
- Modello: `gemini-2.5-flash-lite`
- Forza output JSON con `response_mime_type: "application/json"` + JSON schema
- Temperature: 0.1 (quasi deterministico)

**Prompt ottimizzato per porta sezionale:**
```
Analyze this image of a garage with a sectional door.
Determine if the garage door is OPEN or CLOSED.

OPEN: door is raised (partially or fully), interior visible,
      gap visible between door bottom and floor.
CLOSED: door is fully lowered, flush with frame,
        continuous surface, no interior visible.

Respond as JSON: {"status": "open"|"closed", "confidence": 0.0-1.0, "reasoning": "..."}
If image is too dark/blurry, set confidence below 0.5.
```

### 5. `telegram_notifier.py` - Notifiche

- `send_status_change(old, new, confidence, reasoning, photo_bytes)`: invia foto + testo
- `send_error_alert(message)`: invia avviso errore (dopo 3 errori consecutivi)
- Usa chiamate HTTP dirette all'API Telegram (piu' leggero per serverless)

### 6. `firestore_store.py` - Persistenza

Documenti Firestore nella collection `garage_monitor`:

**Documento `state`:**
```json
{
  "current_status": "closed",
  "last_check_time": "2026-03-05T14:30:00Z",
  "last_change_time": "2026-03-05T08:15:00Z",
  "consecutive_errors": 0
}
```

**Documento `blink_credentials`:**
```json
{
  "token": "...",
  "account_id": "...",
  "client_id": "...",
  "region_id": "...",
  "updated_at": "2026-03-05T14:30:00Z"
}
```

### 7. `main.py` - Entry point Cloud Function

```python
import functions_framework

@functions_framework.http
def check_garage(request):
    # 1. Carica config da env vars
    # 2. Leggi stato e credenziali Blink da Firestore
    # 3. Connetti a Blink con credenziali salvate
    # 4. Cattura snapshot (bytes JPEG)
    # 5. Invia a Gemini per analisi
    # 6. Se confidence >= soglia E stato cambiato:
    #       -> Invia notifica Telegram con foto
    # 7. Se confidence < soglia:
    #       -> Log warning, stato invariato
    # 8. Salva stato aggiornato su Firestore
    # 9. Salva credenziali Blink aggiornate su Firestore
    # 10. Se errore: incrementa contatore, dopo 3 errori -> notifica Telegram
    return "OK", 200
```

---

## Gestione edge cases

| Scenario | Strategia |
|----------|-----------|
| Gemini confidence < 0.7 | Stato NON cambiato, log warning |
| Camera Blink non risponde | Errore catturato, contatore +1, dopo 3 -> notifica |
| Token Blink scaduto | `BlinkAuthExpiredError` -> notifica Telegram "Rieseguire setup_blink.py" |
| Gemini API down | Errore catturato, retry nella prossima invocazione (5 min) |
| Immagine notturna/buia | Prompt istruisce Gemini a dare bassa confidence -> stato invariato |
| Firestore non raggiungibile | Cloud Function fallisce, Cloud Scheduler riprova |
| Primo avvio (stato UNKNOWN) | Imposta stato iniziale, invia notifica "Monitor avviato" |

---

## Procedura di primo setup

### Prerequisiti
- Account Google Cloud (con carta, ma free tier non addebita)
- Account Blink (con telecamera configurata)
- Telegram installato

### Step 1: Creare API key Gemini
1. Vai su https://aistudio.google.com/apikey
2. Crea API key, annotala

### Step 2: Creare bot Telegram
1. Apri Telegram, cerca @BotFather
2. `/newbot` -> segui istruzioni -> annota TOKEN
3. Invia un messaggio al bot
4. Visita `https://api.telegram.org/bot<TOKEN>/getUpdates` -> annota `chat_id`

### Step 3: Setup progetto GCP
```bash
./scripts/setup_gcp.sh
```

### Step 4: Setup credenziali Blink (una tantum, locale)
```bash
python scripts/setup_blink.py
```

### Step 5: Deploy
```bash
./deploy.sh
```

### Step 6: Verifica
- Controlla i log: `gcloud functions logs read check-garage`
- Apri/chiudi fisicamente il garage
- Verifica arrivo notifica Telegram

---

## Dipendenze (`pyproject.toml`)

```
blinkpy>=0.23.0
google-genai>=1.0.0
google-cloud-firestore>=2.0
functions-framework>=3.0
httpx>=0.27.0
pydantic>=2.0
pydantic-settings>=2.0
```
