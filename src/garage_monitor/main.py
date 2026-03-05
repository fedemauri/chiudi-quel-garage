import asyncio
import hashlib
import logging
from datetime import datetime, timezone

import functions_framework

from garage_monitor.blink_client import BlinkAuthExpiredError, BlinkClient
from garage_monitor.config import Settings
from garage_monitor.firestore_store import FirestoreStore
from garage_monitor.gemini_analyzer import GeminiAnalyzer, GeminiParseError
from garage_monitor.models import GarageState, GarageStatus
from garage_monitor.telegram_notifier import TelegramNotifier

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

MAX_CONSECUTIVE_ERRORS = 3
REMINDER_INTERVAL_MINUTES = 15
MAX_REMINDER_MINUTES = 60


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

        # Skip Gemini se immagine identica (byte per byte)
        current_hash = hashlib.md5(snapshot).hexdigest()
        skip_gemini = (
            not first_run
            and state.last_image_hash is not None
            and current_hash == state.last_image_hash
        )

        state.last_check_time = now
        gemini_called = False
        status_just_changed = False

        if skip_gemini:
            logger.info("Immagine identica (hash=%s), skip Gemini", current_hash)
            state.consecutive_errors = 0
        else:
            result = analyzer.analyze(snapshot)
            gemini_called = True
            state.consecutive_errors = 0
            state.last_image_hash = current_hash
            logger.info(
                "Analysis: status=%s confidence=%.2f reasoning=%s",
                result.status.value,
                result.confidence,
                result.reasoning,
            )

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
                state.last_reminder_time = None
                notifier.send_status_change(
                    old_status=old.value,
                    new_status=result.status.value,
                    confidence=result.confidence,
                    reasoning=result.reasoning,
                    photo_bytes=snapshot,
                )
                logger.info("Status changed: %s -> %s", old.value, result.status.value)
                status_just_changed = True
            elif result.confidence < settings.confidence_threshold:
                logger.warning(
                    "Low confidence (%.2f < %.2f). Status unchanged.",
                    result.confidence,
                    settings.confidence_threshold,
                )

        # Reminder: box ancora aperto
        if (
            not first_run
            and not status_just_changed
            and state.current_status == GarageStatus.OPEN
            and state.last_change_time is not None
        ):
            minutes_open = (now - state.last_change_time).total_seconds() / 60
            if REMINDER_INTERVAL_MINUTES <= minutes_open <= MAX_REMINDER_MINUTES:
                should_remind = (
                    state.last_reminder_time is None
                    or (now - state.last_reminder_time).total_seconds() / 60
                    >= REMINDER_INTERVAL_MINUTES
                )
                if should_remind:
                    notifier.send_still_open_reminder(
                        int(minutes_open), snapshot
                    )
                    state.last_reminder_time = now

        store.save_state(state)
        store.save_blink_credentials(updated_creds)

        # Track usage
        period = now.strftime("%Y_%m")
        usage = dict(
            function_invocations=1,
            firestore_reads=2,
            firestore_writes=3,
        )
        if gemini_called:
            usage.update(
                gemini_calls=1,
                gemini_input_tokens=result.input_tokens,
                gemini_output_tokens=result.output_tokens,
            )
        store.increment_usage(period, **usage)

        return "OK"

    except BlinkAuthExpiredError:
        state.consecutive_errors += 1
        state.last_check_time = now
        store.save_state(state)
        notifier.send_error_alert(
            "Autenticazione Blink scaduta. Eseguire setup_blink.py."
        )
        return "AUTH_EXPIRED"

    except ValueError as e:
        logger.exception("Config/parse error: %s", e)
        state.consecutive_errors += 1
        state.last_check_time = now
        store.save_state(state)
        if isinstance(e, GeminiParseError):
            notifier.send_error_alert(
                "Gemini ha risposto in modo inatteso. Controlla i log."
            )
        else:
            notifier.send_error_alert(
                f"Camera non trovata. Verifica GM_BLINK_CAMERA_NAME e rideploya.\n"
                f"Dettaglio: {e}"
            )
        return "CONFIG_ERROR"

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


def _send_usage_report(settings: Settings) -> None:
    store = FirestoreStore(settings.gcp_project_id, settings.firestore_collection)
    notifier = TelegramNotifier(settings.telegram_bot_token, settings.telegram_chat_id)

    now = datetime.now(timezone.utc)
    period = now.strftime("%Y_%m")
    days_in_period = now.day

    stats = store.get_usage_stats(period)

    store.increment_usage(
        period, firestore_reads=1, firestore_writes=1, function_invocations=1
    )

    notifier.send_usage_report(stats, days_in_period)


@functions_framework.http
def check_garage(request):
    settings = Settings()
    action = request.args.get("action")
    if action == "report":
        _send_usage_report(settings)
        return "REPORT_SENT", 200
    result = asyncio.run(_check_garage_async(settings))
    return result, 200
