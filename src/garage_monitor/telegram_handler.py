"""Telegram bot command handler for interactive commands."""

import asyncio
import logging
import re
from datetime import datetime, timedelta, timezone

from garage_monitor.blink_client import BlinkClient
from garage_monitor.config import Settings
from garage_monitor.firestore_store import FirestoreStore
from garage_monitor.telegram_notifier import TelegramNotifier

logger = logging.getLogger(__name__)

_MUTE_PATTERN = re.compile(r"(\d+)\s*h?")


def handle_command(update: dict, settings: Settings) -> str:
    """Parse and dispatch a Telegram bot command."""
    message = update.get("message", {})
    text = message.get("text", "").strip()
    store = FirestoreStore(settings.gcp_project_id, settings.firestore_collection)
    notifier = TelegramNotifier(settings.telegram_bot_token, settings.telegram_chat_id)

    if text.startswith("/stato"):
        return _cmd_stato(store, notifier)
    elif text.startswith("/foto"):
        return asyncio.run(_cmd_foto(store, notifier, settings))
    elif text.startswith("/report"):
        return _cmd_report(store, notifier, settings)
    elif text.startswith("/smuto"):
        return _cmd_smuto(store, notifier)
    elif text.startswith("/muto"):
        return _cmd_muto(text, store, notifier)
    elif text.startswith("/storico"):
        return _cmd_storico(store, notifier)
    else:
        notifier.send_command_response(
            "Comando non riconosciuto. Comandi disponibili:\n"
            "/stato /foto /report /muto /smuto /storico"
        )
        return "UNKNOWN_COMMAND"


def _cmd_stato(store: FirestoreStore, notifier: TelegramNotifier) -> str:
    state = store.get_state()
    if state is None:
        notifier.send_command_response("Nessuno stato disponibile. Il monitor non ha ancora eseguito un check.")
        return "NO_STATE"
    notifier.send_current_status(state)
    return "STATO_SENT"


async def _cmd_foto(store: FirestoreStore, notifier: TelegramNotifier, settings: Settings) -> str:
    blink = BlinkClient()
    try:
        credentials = store.get_blink_credentials()
        if not credentials:
            notifier.send_command_response("Nessuna credenziale Blink trovata.")
            return "NO_CREDENTIALS"
        await blink.connect(credentials)
        snapshot = await blink.take_snapshot(settings.blink_camera_name)
        updated_creds = blink.get_updated_credentials()
        store.save_blink_credentials(updated_creds)
        notifier.send_photo(snapshot)
        return "FOTO_SENT"
    except Exception as e:
        logger.exception("Error taking photo: %s", e)
        notifier.send_command_response(f"Errore nello scatto foto: {e}")
        return "FOTO_ERROR"
    finally:
        await blink.close()


def _cmd_report(store: FirestoreStore, notifier: TelegramNotifier, settings: Settings) -> str:
    from garage_monitor.main import _send_usage_report
    _send_usage_report(settings)
    return "REPORT_SENT"


def _cmd_muto(text: str, store: FirestoreStore, notifier: TelegramNotifier) -> str:
    match = _MUTE_PATTERN.search(text)
    hours = int(match.group(1)) if match else 2
    hours = min(hours, 24)
    now = datetime.now(timezone.utc)
    state = store.get_state()
    if state is None:
        notifier.send_command_response("Nessuno stato disponibile.")
        return "NO_STATE"
    state.muted_until = now + timedelta(hours=hours)
    store.save_state(state)
    notifier.send_command_response(f"\U0001f507 Notifiche silenziate per {hours}h.")
    return "MUTED"


def _cmd_smuto(store: FirestoreStore, notifier: TelegramNotifier) -> str:
    state = store.get_state()
    if state is None:
        notifier.send_command_response("Nessuno stato disponibile.")
        return "NO_STATE"
    state.muted_until = None
    store.save_state(state)
    notifier.send_command_response("\U0001f514 Notifiche riattivate.")
    return "UNMUTED"


def _cmd_storico(store: FirestoreStore, notifier: TelegramNotifier) -> str:
    try:
        events = store.get_recent_events(10)
    except Exception as e:
        logger.exception("Error fetching events: %s", e)
        notifier.send_command_response("Errore nel recupero storico. Indice Firestore mancante?")
        return "STORICO_ERROR"
    notifier.send_history(events)
    return "STORICO_SENT"
