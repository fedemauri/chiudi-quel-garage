import asyncio
import logging
from datetime import datetime, timezone

import functions_framework

from garage_monitor.blink_client import BlinkAuthExpiredError, BlinkClient
from garage_monitor.config import Settings
from garage_monitor.firestore_store import FirestoreStore
from garage_monitor.gemini_analyzer import GeminiAnalyzer
from garage_monitor.models import GarageState, GarageStatus
from garage_monitor.telegram_notifier import TelegramNotifier

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

MAX_CONSECUTIVE_ERRORS = 3


async def _check_garage_async(settings: Settings) -> str:
    store = FirestoreStore(settings.gcp_project_id, settings.firestore_collection)
    notifier = TelegramNotifier(settings.telegram_bot_token, settings.telegram_chat_id)
    analyzer = GeminiAnalyzer(settings.gemini_api_key, settings.gemini_model)

    state = store.get_state()
    first_run = state is None
    if first_run:
        state = GarageState()

    now = datetime.now(timezone.utc)
    blink = BlinkClient()

    try:
        credentials = store.get_blink_credentials()
        if not credentials:
            notifier.send_error_alert(
                "Nessuna credenziale Blink trovata. Eseguire setup_blink.py."
            )
            return "NO_CREDENTIALS"

        await blink.connect(credentials)
        snapshot = await blink.take_snapshot(settings.blink_camera_name)
        updated_creds = blink.get_updated_credentials()

        result = analyzer.analyze(snapshot)
        logger.info(
            "Analysis: status=%s confidence=%.2f reasoning=%s",
            result.status.value,
            result.confidence,
            result.reasoning,
        )

        state.last_check_time = now
        state.consecutive_errors = 0

        if first_run:
            state.current_status = result.status
            state.last_change_time = now
            notifier.send_monitor_started()
            logger.info("First run. Initial status: %s", result.status.value)
        elif (
            result.confidence >= settings.confidence_threshold
            and result.status != state.current_status
        ):
            old = state.current_status
            state.current_status = result.status
            state.last_change_time = now
            notifier.send_status_change(
                old_status=old.value,
                new_status=result.status.value,
                confidence=result.confidence,
                reasoning=result.reasoning,
                photo_bytes=snapshot,
            )
            logger.info("Status changed: %s -> %s", old.value, result.status.value)
        elif result.confidence < settings.confidence_threshold:
            logger.warning(
                "Low confidence (%.2f < %.2f). Status unchanged.",
                result.confidence,
                settings.confidence_threshold,
            )

        store.save_state(state)
        store.save_blink_credentials(updated_creds)
        return "OK"

    except BlinkAuthExpiredError:
        state.consecutive_errors += 1
        state.last_check_time = now
        store.save_state(state)
        notifier.send_error_alert(
            "Autenticazione Blink scaduta. Eseguire setup_blink.py."
        )
        return "AUTH_EXPIRED"

    except Exception as e:
        logger.exception("Error during check: %s", e)
        state.consecutive_errors += 1
        state.last_check_time = now
        store.save_state(state)

        if state.consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
            notifier.send_error_alert(
                f"{state.consecutive_errors} errori consecutivi. "
                f"Ultimo errore: {e}"
            )
        return "ERROR"

    finally:
        await blink.close()


@functions_framework.http
def check_garage(request):
    settings = Settings()
    result = asyncio.run(_check_garage_async(settings))
    return result, 200
