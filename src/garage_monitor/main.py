import asyncio
import logging
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import functions_framework

from garage_monitor.blink_client import BlinkAuthExpiredError, BlinkClient
from garage_monitor.config import Settings
from garage_monitor.firestore_store import FirestoreStore
from garage_monitor.gemini_analyzer import GeminiAnalyzer, GeminiParseError
from garage_monitor.tflite_analyzer import TFLiteAnalyzer
from garage_monitor.models import GarageState, GarageStatus
from garage_monitor.telegram_notifier import TelegramNotifier

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

MAX_CONSECUTIVE_ERRORS = 3
REMINDER_INTERVAL_MINUTES = 15
MAX_REMINDER_MINUTES = 60
STALENESS_MARGIN = 12
ROME_TZ = ZoneInfo("Europe/Rome")


def _is_night_time(utc_now: datetime) -> bool:
    """Return True if the local time in Rome is between 0:00 and 6:59 (night alert hours)."""
    return 0 <= utc_now.astimezone(ROME_TZ).hour < 7


async def _check_garage_async(settings: Settings) -> str:
    store = FirestoreStore(settings.gcp_project_id, settings.firestore_collection)
    notifier = TelegramNotifier(settings.telegram_bot_token, settings.telegram_chat_id)
    if settings.analyzer == "tflite":
        analyzer = TFLiteAnalyzer()
    else:
        analyzer = GeminiAnalyzer(settings.gemini_api_key, settings.gemini_model)

    state = store.get_state()
    first_run = state is None
    if first_run:
        state = GarageState()

    now = datetime.now(timezone.utc)

    # Staleness detection: alert if last check is older than expected
    if not first_run and state.last_check_time is not None:
        elapsed_minutes = (now - state.last_check_time).total_seconds() / 60
        if elapsed_minutes > STALENESS_MARGIN:
            notifier.send_error_alert(
                f"Possibile interruzione: ultimo controllo {int(elapsed_minutes)} minuti fa "
                f"(atteso max {STALENESS_MARGIN} min)."
            )

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

        state.last_check_time = now
        status_just_changed = False

        result = analyzer.analyze(snapshot)
        state.consecutive_errors = 0
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
        elif result.confidence < settings.confidence_threshold:
            logger.warning(
                "Low confidence (%.2f < %.2f). Status unchanged.",
                result.confidence,
                settings.confidence_threshold,
            )
        elif result.status != state.current_status:
            old = state.current_status
            old_change_time = state.last_change_time
            state.current_status = result.status
            state.last_change_time = now
            state.last_reminder_time = None
            state.last_final_warning_sent = False
            if result.status == GarageStatus.OPEN and _is_night_time(now):
                notifier.send_night_alert(
                    confidence=result.confidence,
                    reasoning=result.reasoning,
                    photo_bytes=snapshot,
                )
            else:
                notifier.send_status_change(
                    old_status=old.value,
                    new_status=result.status.value,
                    confidence=result.confidence,
                    reasoning=result.reasoning,
                    photo_bytes=snapshot,
                )
            logger.info("Status changed: %s -> %s", old.value, result.status.value)
            status_just_changed = True

            # Log event to Firestore
            duration_seconds = None
            if result.status == GarageStatus.CLOSED and old_change_time:
                duration_seconds = int((now - old_change_time).total_seconds())
            store.save_event({
                "timestamp": now,
                "old_status": old.value,
                "new_status": result.status.value,
                "confidence": result.confidence,
                "reasoning": result.reasoning,
                "duration_seconds": duration_seconds,
                "expire_at": now + timedelta(days=30),
            })

        # Auto-expire mute
        is_muted = state.muted_until is not None and state.muted_until > now
        if state.muted_until is not None and state.muted_until <= now:
            state.muted_until = None

        # Reminder: box ancora aperto
        if (
            not first_run
            and not status_just_changed
            and not is_muted
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
            elif minutes_open > MAX_REMINDER_MINUTES and not state.last_final_warning_sent:
                notifier.send_final_warning(int(minutes_open), snapshot)
                state.last_final_warning_sent = True

        store.save_state(state)
        store.save_blink_credentials(updated_creds)

        # Track usage
        period = now.strftime("%Y_%m")
        usage = dict(
            function_invocations=1,
            firestore_reads=2,
            firestore_writes=3,
        )
        if result.input_tokens > 0:
            usage.update(
                gemini_calls=1,
                gemini_input_tokens=result.input_tokens,
                gemini_output_tokens=result.output_tokens,
            )
        if status_just_changed:
            usage["firestore_writes"] = usage.get("firestore_writes", 0) + 1
            if result.status == GarageStatus.OPEN:
                usage["garage_openings"] = 1
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
        # Only alert after repeated failures: a single camera-not-found or
        # parse glitch is usually a transient Blink/API hiccup that self-heals
        # on the next 5-minute run, so it shouldn't trigger a notification.
        if state.consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
            if isinstance(e, GeminiParseError):
                notifier.send_error_alert(
                    "Gemini ha risposto in modo inatteso. Controlla i log."
                )
            else:
                notifier.send_error_alert(
                    f"Camera non trovata dopo {state.consecutive_errors} tentativi. "
                    f"Verifica GM_BLINK_CAMERA_NAME e rideploya.\n"
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

    # Projected monthly Gemini cost alert
    cost_warning = None
    if days_in_period > 0:
        current_cost = (
            stats.gemini_input_tokens * 0.30
            + stats.gemini_output_tokens * 2.50
        ) / 1_000_000
        projected_cost = current_cost / days_in_period * 30
        if projected_cost > settings.gemini_cost_alert_threshold:
            cost_warning = (
                f"Costo Gemini proiettato: ~${projected_cost:.2f}/mese "
                f"(soglia: ${settings.gemini_cost_alert_threshold:.2f})"
            )

    notifier.send_usage_report(stats, days_in_period, cost_warning=cost_warning)


def _handle_telegram_webhook(request, body: dict, settings: Settings):
    """Validate and dispatch Telegram webhook requests."""
    # Webhook disabled if secret not configured
    if not settings.telegram_webhook_secret:
        return "WEBHOOK_DISABLED", 403

    # Validate secret token header
    secret = request.headers.get("X-Telegram-Bot-Api-Secret-Token", "")
    if secret != settings.telegram_webhook_secret:
        return "INVALID_SECRET", 403

    # Extract chat_id from message or callback_query
    message = body.get("message") or body.get("callback_query", {}).get("message") or {}
    chat_id = str(message.get("chat", {}).get("id", ""))
    if chat_id != settings.telegram_chat_id:
        return "OK", 200  # Return 200 to avoid Telegram retries

    from garage_monitor.telegram_handler import handle_command
    result = handle_command(body, settings)
    return result, 200


@functions_framework.http
def check_garage(request):
    settings = Settings()

    # Telegram webhook? (JSON body with "message" or "callback_query" field)
    if request.content_type and "application/json" in request.content_type:
        body = request.get_json(silent=True)
        if body and ("message" in body or "callback_query" in body):
            return _handle_telegram_webhook(request, body, settings)

    action = request.args.get("action")
    if action == "report":
        _send_usage_report(settings)
        return "REPORT_SENT", 200
    result = asyncio.run(_check_garage_async(settings))
    return result, 200
