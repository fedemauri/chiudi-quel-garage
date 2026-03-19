import asyncio
import hashlib
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from garage_monitor.gemini_analyzer import GeminiParseError
from garage_monitor.main import (
    STALENESS_MARGIN,
    _check_garage_async,
    _is_night_time,
    _send_usage_report,
    check_garage,
)
from garage_monitor.models import (
    GarageState,
    GarageStatus,
    GeminiAnalysisResult,
    UsageStats,
)


def _make_settings(**overrides):
    defaults = {
        "blink_username": "test@test.com",
        "blink_password": "pass",
        "blink_camera_name": "Garage",
        "analyzer": "gemini",
        "gemini_api_key": "fake-key",
        "gemini_model": "gemini-2.5-flash",
        "confidence_threshold": 0.7,
        "telegram_bot_token": "fake-token",
        "telegram_chat_id": "123",
        "gcp_project_id": "test-project",
        "firestore_collection": "garage_monitor",
        "gemini_cost_alert_threshold": 3.0,
        "telegram_webhook_secret": "test-secret",
    }
    defaults.update(overrides)
    settings = MagicMock()
    for k, v in defaults.items():
        setattr(settings, k, v)
    return settings


def _setup_mocks(
    mock_store_cls,
    mock_tg_cls,
    mock_gem_cls,
    mock_blink_cls,
    state=None,
    analysis_status=GarageStatus.CLOSED,
    analysis_confidence=0.95,
    analysis_reasoning="Door closed",
    analyze_side_effect=None,
    snapshot_side_effect=None,
):
    store = MagicMock()
    store.get_state.return_value = state
    store.get_blink_credentials.return_value = {"token": "abc"}
    mock_store_cls.return_value = store

    notifier = MagicMock()
    mock_tg_cls.return_value = notifier

    analyzer = MagicMock()
    if analyze_side_effect:
        analyzer.analyze.side_effect = analyze_side_effect
    else:
        analyzer.analyze.return_value = GeminiAnalysisResult(
            status=analysis_status,
            confidence=analysis_confidence,
            reasoning=analysis_reasoning,
        )
    mock_gem_cls.return_value = analyzer

    blink = MagicMock()
    blink.connect = AsyncMock()
    if snapshot_side_effect:
        blink.take_snapshot = AsyncMock(side_effect=snapshot_side_effect)
    else:
        blink.take_snapshot = AsyncMock(return_value=b"jpeg-data")
    blink.get_updated_credentials.return_value = {"token": "refreshed"}
    blink.close = AsyncMock()
    mock_blink_cls.return_value = blink

    return store, notifier, analyzer, blink


class TestCheckGarage:
    @patch("garage_monitor.main.BlinkClient")
    @patch("garage_monitor.main.GeminiAnalyzer")
    @patch("garage_monitor.main.TelegramNotifier")
    @patch("garage_monitor.main.FirestoreStore")
    def test_first_run(self, mock_store_cls, mock_tg_cls, mock_gem_cls, mock_blink_cls):
        store, notifier, _, _ = _setup_mocks(
            mock_store_cls, mock_tg_cls, mock_gem_cls, mock_blink_cls
        )

        result = asyncio.run(_check_garage_async(_make_settings()))

        assert result == "OK"
        notifier.send_monitor_started.assert_called_once()
        store.save_state.assert_called_once()
        store.save_blink_credentials.assert_called_once()

    @patch("garage_monitor.main._is_night_time", return_value=False)
    @patch("garage_monitor.main.BlinkClient")
    @patch("garage_monitor.main.GeminiAnalyzer")
    @patch("garage_monitor.main.TelegramNotifier")
    @patch("garage_monitor.main.FirestoreStore")
    def test_status_change(self, mock_store_cls, mock_tg_cls, mock_gem_cls, mock_blink_cls, _mock_night):
        """Status change requires debounce: pending_count=1 means this is the 2nd confirmation."""
        now = datetime.now(timezone.utc)
        store, notifier, _, _ = _setup_mocks(
            mock_store_cls, mock_tg_cls, mock_gem_cls, mock_blink_cls,
            state=GarageState(
                current_status=GarageStatus.CLOSED,
                last_check_time=now,
                last_change_time=now,
                pending_status=GarageStatus.OPEN,
                pending_count=1,
            ),
            analysis_status=GarageStatus.OPEN,
            analysis_confidence=0.9,
            analysis_reasoning="Door raised",
        )

        result = asyncio.run(_check_garage_async(_make_settings()))

        assert result == "OK"
        notifier.send_status_change.assert_called_once()

    @patch("garage_monitor.main.BlinkClient")
    @patch("garage_monitor.main.GeminiAnalyzer")
    @patch("garage_monitor.main.TelegramNotifier")
    @patch("garage_monitor.main.FirestoreStore")
    def test_low_confidence_no_change(self, mock_store_cls, mock_tg_cls, mock_gem_cls, mock_blink_cls):
        now = datetime.now(timezone.utc)
        _setup_mocks(
            mock_store_cls, mock_tg_cls, mock_gem_cls, mock_blink_cls,
            state=GarageState(
                current_status=GarageStatus.CLOSED,
                last_check_time=now,
                last_change_time=now,
            ),
            analysis_status=GarageStatus.OPEN,
            analysis_confidence=0.3,
            analysis_reasoning="Too dark",
        )
        notifier = mock_tg_cls.return_value

        result = asyncio.run(_check_garage_async(_make_settings()))

        assert result == "OK"
        notifier.send_status_change.assert_not_called()

    @patch("garage_monitor.main.BlinkClient")
    @patch("garage_monitor.main.TelegramNotifier")
    @patch("garage_monitor.main.FirestoreStore")
    def test_no_credentials(self, mock_store_cls, mock_tg_cls, mock_blink_cls):
        store = MagicMock()
        store.get_state.return_value = None
        store.get_blink_credentials.return_value = None
        mock_store_cls.return_value = store

        notifier = MagicMock()
        mock_tg_cls.return_value = notifier

        blink = MagicMock()
        blink.close = AsyncMock()
        mock_blink_cls.return_value = blink

        result = asyncio.run(_check_garage_async(_make_settings()))

        assert result == "NO_CREDENTIALS"
        notifier.send_error_alert.assert_called_once()

    @patch("garage_monitor.main.BlinkClient")
    @patch("garage_monitor.main.GeminiAnalyzer")
    @patch("garage_monitor.main.TelegramNotifier")
    @patch("garage_monitor.main.FirestoreStore")
    def test_consecutive_errors_trigger_alert(
        self, mock_store_cls, mock_tg_cls, mock_gem_cls, mock_blink_cls
    ):
        now = datetime.now(timezone.utc)
        _, notifier, _, _ = _setup_mocks(
            mock_store_cls, mock_tg_cls, mock_gem_cls, mock_blink_cls,
            state=GarageState(
                current_status=GarageStatus.CLOSED,
                last_check_time=now,
                last_change_time=now,
                consecutive_errors=2,
            ),
            analyze_side_effect=RuntimeError("Gemini down"),
        )

        result = asyncio.run(_check_garage_async(_make_settings()))

        assert result == "ERROR"
        notifier.send_error_alert.assert_called_once()


    @patch("garage_monitor.main.BlinkClient")
    @patch("garage_monitor.main.TFLiteAnalyzer")
    @patch("garage_monitor.main.TelegramNotifier")
    @patch("garage_monitor.main.FirestoreStore")
    def test_tflite_analyzer_used_when_configured(
        self, mock_store_cls, mock_tg_cls, mock_tflite_cls, mock_blink_cls
    ):
        """When analyzer=tflite, TFLiteAnalyzer is used instead of GeminiAnalyzer."""
        store, notifier, _, _ = _setup_mocks(
            mock_store_cls, mock_tg_cls, mock_tflite_cls, mock_blink_cls
        )

        result = asyncio.run(_check_garage_async(_make_settings(analyzer="tflite")))

        assert result == "OK"
        mock_tflite_cls.assert_called_once()
        notifier.send_monitor_started.assert_called_once()


class TestDebounce:
    @patch("garage_monitor.main.BlinkClient")
    @patch("garage_monitor.main.GeminiAnalyzer")
    @patch("garage_monitor.main.TelegramNotifier")
    @patch("garage_monitor.main.FirestoreStore")
    def test_single_detection_sets_pending(
        self, mock_store_cls, mock_tg_cls, mock_gem_cls, mock_blink_cls
    ):
        """First detection of a different status sets pending but doesn't change state."""
        now = datetime.now(timezone.utc)
        store, notifier, _, _ = _setup_mocks(
            mock_store_cls, mock_tg_cls, mock_gem_cls, mock_blink_cls,
            state=GarageState(
                current_status=GarageStatus.CLOSED,
                last_check_time=now,
                last_change_time=now,
            ),
            analysis_status=GarageStatus.OPEN,
            analysis_confidence=0.9,
            analysis_reasoning="Door raised",
        )

        result = asyncio.run(_check_garage_async(_make_settings()))

        assert result == "OK"
        notifier.send_status_change.assert_not_called()
        saved_state = store.save_state.call_args[0][0]
        assert saved_state.current_status == GarageStatus.CLOSED
        assert saved_state.pending_status == GarageStatus.OPEN
        assert saved_state.pending_count == 1

    @patch("garage_monitor.main.BlinkClient")
    @patch("garage_monitor.main.GeminiAnalyzer")
    @patch("garage_monitor.main.TelegramNotifier")
    @patch("garage_monitor.main.FirestoreStore")
    def test_oscillation_resets_pending(
        self, mock_store_cls, mock_tg_cls, mock_gem_cls, mock_blink_cls
    ):
        """If current reading matches current_status, pending is reset."""
        now = datetime.now(timezone.utc)
        store, notifier, _, _ = _setup_mocks(
            mock_store_cls, mock_tg_cls, mock_gem_cls, mock_blink_cls,
            state=GarageState(
                current_status=GarageStatus.CLOSED,
                last_check_time=now,
                last_change_time=now,
                pending_status=GarageStatus.OPEN,
                pending_count=1,
            ),
            analysis_status=GarageStatus.CLOSED,
            analysis_confidence=0.98,
            analysis_reasoning="Door closed",
        )

        result = asyncio.run(_check_garage_async(_make_settings()))

        assert result == "OK"
        notifier.send_status_change.assert_not_called()
        saved_state = store.save_state.call_args[0][0]
        assert saved_state.current_status == GarageStatus.CLOSED
        assert saved_state.pending_status is None
        assert saved_state.pending_count == 0


class TestImageHashSkip:
    @patch("garage_monitor.main.BlinkClient")
    @patch("garage_monitor.main.GeminiAnalyzer")
    @patch("garage_monitor.main.TelegramNotifier")
    @patch("garage_monitor.main.FirestoreStore")
    def test_same_image_skips_gemini(
        self, mock_store_cls, mock_tg_cls, mock_gem_cls, mock_blink_cls
    ):
        """Immagine identica alla precedente -> Gemini non viene chiamato."""
        snapshot_data = b"jpeg-data"
        image_hash = hashlib.md5(snapshot_data).hexdigest()
        now = datetime.now(timezone.utc)
        store, notifier, analyzer, _ = _setup_mocks(
            mock_store_cls, mock_tg_cls, mock_gem_cls, mock_blink_cls,
            state=GarageState(
                current_status=GarageStatus.CLOSED,
                last_check_time=now,
                last_change_time=now,
                last_image_hash=image_hash,
            ),
        )

        result = asyncio.run(_check_garage_async(_make_settings()))

        assert result == "OK"
        analyzer.analyze.assert_not_called()
        store.save_state.assert_called_once()
        # function_invocations tracked, but no gemini_calls
        call_kwargs = store.increment_usage.call_args[1]
        assert call_kwargs["function_invocations"] == 1
        assert "gemini_calls" not in call_kwargs

    @patch("garage_monitor.main.BlinkClient")
    @patch("garage_monitor.main.GeminiAnalyzer")
    @patch("garage_monitor.main.TelegramNotifier")
    @patch("garage_monitor.main.FirestoreStore")
    def test_different_image_calls_gemini(
        self, mock_store_cls, mock_tg_cls, mock_gem_cls, mock_blink_cls
    ):
        """Immagine diversa dalla precedente -> Gemini viene chiamato."""
        now = datetime.now(timezone.utc)
        store, notifier, analyzer, _ = _setup_mocks(
            mock_store_cls, mock_tg_cls, mock_gem_cls, mock_blink_cls,
            state=GarageState(
                current_status=GarageStatus.CLOSED,
                last_check_time=now,
                last_change_time=now,
                last_image_hash="different-hash",
            ),
        )

        result = asyncio.run(_check_garage_async(_make_settings()))

        assert result == "OK"
        analyzer.analyze.assert_called_once()

    @patch("garage_monitor.main.BlinkClient")
    @patch("garage_monitor.main.GeminiAnalyzer")
    @patch("garage_monitor.main.TelegramNotifier")
    @patch("garage_monitor.main.FirestoreStore")
    def test_first_run_never_skips(
        self, mock_store_cls, mock_tg_cls, mock_gem_cls, mock_blink_cls
    ):
        """First run -> Gemini viene sempre chiamato anche se hash presente."""
        store, notifier, analyzer, _ = _setup_mocks(
            mock_store_cls, mock_tg_cls, mock_gem_cls, mock_blink_cls,
            state=None,
        )

        result = asyncio.run(_check_garage_async(_make_settings()))

        assert result == "OK"
        analyzer.analyze.assert_called_once()


class TestStillOpenReminder:
    @patch("garage_monitor.main.BlinkClient")
    @patch("garage_monitor.main.GeminiAnalyzer")
    @patch("garage_monitor.main.TelegramNotifier")
    @patch("garage_monitor.main.FirestoreStore")
    def test_still_open_sends_reminder(
        self, mock_store_cls, mock_tg_cls, mock_gem_cls, mock_blink_cls
    ):
        """Garage aperto da 15+ min -> reminder inviato."""
        now = datetime.now(timezone.utc)
        _, notifier, _, _ = _setup_mocks(
            mock_store_cls, mock_tg_cls, mock_gem_cls, mock_blink_cls,
            state=GarageState(
                current_status=GarageStatus.OPEN,
                last_check_time=now - timedelta(minutes=5),
                last_change_time=now - timedelta(minutes=20),
                last_reminder_time=None,
            ),
            analysis_status=GarageStatus.OPEN,
            analysis_confidence=0.9,
            analysis_reasoning="Door open",
        )

        result = asyncio.run(_check_garage_async(_make_settings()))

        assert result == "OK"
        notifier.send_still_open_reminder.assert_called_once()
        call_args = notifier.send_still_open_reminder.call_args
        assert call_args[0][0] >= 20  # minutes_open

    @patch("garage_monitor.main.BlinkClient")
    @patch("garage_monitor.main.GeminiAnalyzer")
    @patch("garage_monitor.main.TelegramNotifier")
    @patch("garage_monitor.main.FirestoreStore")
    def test_still_open_no_reminder_too_soon_since_last(
        self, mock_store_cls, mock_tg_cls, mock_gem_cls, mock_blink_cls
    ):
        """Garage aperto da 20 min, ultimo reminder 10 min fa -> nessun reminder."""
        now = datetime.now(timezone.utc)
        _, notifier, _, _ = _setup_mocks(
            mock_store_cls, mock_tg_cls, mock_gem_cls, mock_blink_cls,
            state=GarageState(
                current_status=GarageStatus.OPEN,
                last_check_time=now - timedelta(minutes=5),
                last_change_time=now - timedelta(minutes=20),
                last_reminder_time=now - timedelta(minutes=10),
            ),
            analysis_status=GarageStatus.OPEN,
            analysis_confidence=0.9,
            analysis_reasoning="Door open",
        )

        result = asyncio.run(_check_garage_async(_make_settings()))

        assert result == "OK"
        notifier.send_still_open_reminder.assert_not_called()

    @patch("garage_monitor.main.BlinkClient")
    @patch("garage_monitor.main.GeminiAnalyzer")
    @patch("garage_monitor.main.TelegramNotifier")
    @patch("garage_monitor.main.FirestoreStore")
    def test_still_open_no_reminder_before_interval(
        self, mock_store_cls, mock_tg_cls, mock_gem_cls, mock_blink_cls
    ):
        """Garage aperto da 5 min -> nessun reminder (< 15 min dall'apertura)."""
        now = datetime.now(timezone.utc)
        _, notifier, _, _ = _setup_mocks(
            mock_store_cls, mock_tg_cls, mock_gem_cls, mock_blink_cls,
            state=GarageState(
                current_status=GarageStatus.OPEN,
                last_check_time=now - timedelta(minutes=5),
                last_change_time=now - timedelta(minutes=5),
                last_reminder_time=None,
            ),
            analysis_status=GarageStatus.OPEN,
            analysis_confidence=0.9,
            analysis_reasoning="Door open",
        )

        result = asyncio.run(_check_garage_async(_make_settings()))

        assert result == "OK"
        notifier.send_still_open_reminder.assert_not_called()

    @patch("garage_monitor.main.BlinkClient")
    @patch("garage_monitor.main.GeminiAnalyzer")
    @patch("garage_monitor.main.TelegramNotifier")
    @patch("garage_monitor.main.FirestoreStore")
    def test_still_open_no_reminder_after_max(
        self, mock_store_cls, mock_tg_cls, mock_gem_cls, mock_blink_cls
    ):
        """Garage aperto da 65 min -> nessun reminder (> MAX_REMINDER_MINUTES)."""
        now = datetime.now(timezone.utc)
        _, notifier, _, _ = _setup_mocks(
            mock_store_cls, mock_tg_cls, mock_gem_cls, mock_blink_cls,
            state=GarageState(
                current_status=GarageStatus.OPEN,
                last_check_time=now - timedelta(minutes=5),
                last_change_time=now - timedelta(minutes=65),
                last_reminder_time=now - timedelta(minutes=20),
            ),
            analysis_status=GarageStatus.OPEN,
            analysis_confidence=0.9,
            analysis_reasoning="Door open",
        )

        result = asyncio.run(_check_garage_async(_make_settings()))

        assert result == "OK"
        notifier.send_still_open_reminder.assert_not_called()

    @patch("garage_monitor.main.BlinkClient")
    @patch("garage_monitor.main.GeminiAnalyzer")
    @patch("garage_monitor.main.TelegramNotifier")
    @patch("garage_monitor.main.FirestoreStore")
    def test_reminder_resets_on_close(
        self, mock_store_cls, mock_tg_cls, mock_gem_cls, mock_blink_cls
    ):
        """Garage si chiude -> last_reminder_time torna a None."""
        now = datetime.now(timezone.utc)
        store, notifier, _, _ = _setup_mocks(
            mock_store_cls, mock_tg_cls, mock_gem_cls, mock_blink_cls,
            state=GarageState(
                current_status=GarageStatus.OPEN,
                last_check_time=now - timedelta(minutes=5),
                last_change_time=now - timedelta(minutes=20),
                last_reminder_time=now - timedelta(minutes=5),
                pending_status=GarageStatus.CLOSED,
                pending_count=1,
            ),
            analysis_status=GarageStatus.CLOSED,
            analysis_confidence=0.95,
            analysis_reasoning="Door closed",
        )

        result = asyncio.run(_check_garage_async(_make_settings()))

        assert result == "OK"
        notifier.send_status_change.assert_called_once()
        saved_state = store.save_state.call_args[0][0]
        assert saved_state.last_reminder_time is None


class TestImmediateErrors:
    @patch("garage_monitor.main.BlinkClient")
    @patch("garage_monitor.main.GeminiAnalyzer")
    @patch("garage_monitor.main.TelegramNotifier")
    @patch("garage_monitor.main.FirestoreStore")
    def test_camera_not_found_immediate_alert(
        self, mock_store_cls, mock_tg_cls, mock_gem_cls, mock_blink_cls
    ):
        now = datetime.now(timezone.utc)
        _, notifier, _, _ = _setup_mocks(
            mock_store_cls, mock_tg_cls, mock_gem_cls, mock_blink_cls,
            state=GarageState(
                current_status=GarageStatus.CLOSED,
                last_check_time=now,
                last_change_time=now,
            ),
            snapshot_side_effect=ValueError("Camera 'X' not found"),
        )

        result = asyncio.run(_check_garage_async(_make_settings()))

        assert result == "CONFIG_ERROR"
        notifier.send_error_alert.assert_called_once()
        msg = notifier.send_error_alert.call_args[0][0]
        assert "Camera non trovata" in msg

    @patch("garage_monitor.main.BlinkClient")
    @patch("garage_monitor.main.GeminiAnalyzer")
    @patch("garage_monitor.main.TelegramNotifier")
    @patch("garage_monitor.main.FirestoreStore")
    def test_gemini_parse_error_immediate_alert(
        self, mock_store_cls, mock_tg_cls, mock_gem_cls, mock_blink_cls
    ):
        now = datetime.now(timezone.utc)
        _, notifier, _, _ = _setup_mocks(
            mock_store_cls, mock_tg_cls, mock_gem_cls, mock_blink_cls,
            state=GarageState(
                current_status=GarageStatus.CLOSED,
                last_check_time=now,
                last_change_time=now,
            ),
            analyze_side_effect=GeminiParseError("Bad JSON"),
        )

        result = asyncio.run(_check_garage_async(_make_settings()))

        assert result == "CONFIG_ERROR"
        notifier.send_error_alert.assert_called_once()
        msg = notifier.send_error_alert.call_args[0][0]
        assert "Gemini" in msg


class TestUsageReport:
    @patch("garage_monitor.main.TelegramNotifier")
    @patch("garage_monitor.main.FirestoreStore")
    def test_report_action_sends_usage(self, mock_store_cls, mock_tg_cls):
        store = MagicMock()
        store.get_usage_stats.return_value = UsageStats(
            period="2026_03",
            function_invocations=1440,
            gemini_calls=1440,
            gemini_input_tokens=72000,
            gemini_output_tokens=14400,
            firestore_reads=4320,
            firestore_writes=2880,
        )
        mock_store_cls.return_value = store

        notifier = MagicMock()
        mock_tg_cls.return_value = notifier

        _send_usage_report(_make_settings())

        notifier.send_usage_report.assert_called_once()
        store.increment_usage.assert_called_once()

    @patch("garage_monitor.main.BlinkClient")
    @patch("garage_monitor.main.GeminiAnalyzer")
    @patch("garage_monitor.main.TelegramNotifier")
    @patch("garage_monitor.main.FirestoreStore")
    def test_usage_tracking_incremented(
        self, mock_store_cls, mock_tg_cls, mock_gem_cls, mock_blink_cls
    ):
        now = datetime.now(timezone.utc)
        store, _, _, _ = _setup_mocks(
            mock_store_cls, mock_tg_cls, mock_gem_cls, mock_blink_cls,
            state=GarageState(
                current_status=GarageStatus.CLOSED,
                last_check_time=now,
                last_change_time=now,
            ),
        )
        # Set non-zero tokens so Gemini usage is tracked
        mock_gem_cls.return_value.analyze.return_value = GeminiAnalysisResult(
            status=GarageStatus.CLOSED,
            confidence=0.95,
            reasoning="Door closed",
            input_tokens=100,
            output_tokens=20,
        )

        result = asyncio.run(_check_garage_async(_make_settings()))

        assert result == "OK"
        store.increment_usage.assert_called_once()
        call_kwargs = store.increment_usage.call_args
        assert call_kwargs[1]["function_invocations"] == 1
        assert call_kwargs[1]["gemini_calls"] == 1

    @patch("garage_monitor.main.Settings")
    @patch("garage_monitor.main.FirestoreStore")
    @patch("garage_monitor.main.TelegramNotifier")
    def test_check_garage_routes_report(
        self, mock_tg_cls, mock_store_cls, mock_settings_cls
    ):
        mock_settings_cls.return_value = _make_settings()
        store = MagicMock()
        store.get_usage_stats.return_value = UsageStats(period="2026_03")
        mock_store_cls.return_value = store
        mock_tg_cls.return_value = MagicMock()

        request = MagicMock()
        request.args = {"action": "report"}

        result_text, status_code = check_garage(request)

        assert result_text == "REPORT_SENT"
        assert status_code == 200


class TestStalenessDetection:
    @patch("garage_monitor.main.BlinkClient")
    @patch("garage_monitor.main.GeminiAnalyzer")
    @patch("garage_monitor.main.TelegramNotifier")
    @patch("garage_monitor.main.FirestoreStore")
    def test_stale_check_sends_alert(
        self, mock_store_cls, mock_tg_cls, mock_gem_cls, mock_blink_cls
    ):
        """last_check 15+ min ago -> staleness alert sent."""
        now = datetime.now(timezone.utc)
        _, notifier, _, _ = _setup_mocks(
            mock_store_cls, mock_tg_cls, mock_gem_cls, mock_blink_cls,
            state=GarageState(
                current_status=GarageStatus.CLOSED,
                last_check_time=now - timedelta(minutes=15),
                last_change_time=now - timedelta(hours=1),
            ),
        )

        result = asyncio.run(_check_garage_async(_make_settings()))

        assert result == "OK"
        # Staleness alert was sent via send_error_alert
        alert_calls = [
            call for call in notifier.send_error_alert.call_args_list
            if "Possibile interruzione" in call[0][0]
        ]
        assert len(alert_calls) == 1
        assert "15 minuti fa" in alert_calls[0][0][0]

    @patch("garage_monitor.main.BlinkClient")
    @patch("garage_monitor.main.GeminiAnalyzer")
    @patch("garage_monitor.main.TelegramNotifier")
    @patch("garage_monitor.main.FirestoreStore")
    def test_fresh_check_no_alert(
        self, mock_store_cls, mock_tg_cls, mock_gem_cls, mock_blink_cls
    ):
        """last_check 4 min ago -> no staleness alert."""
        now = datetime.now(timezone.utc)
        _, notifier, _, _ = _setup_mocks(
            mock_store_cls, mock_tg_cls, mock_gem_cls, mock_blink_cls,
            state=GarageState(
                current_status=GarageStatus.CLOSED,
                last_check_time=now - timedelta(minutes=4),
                last_change_time=now - timedelta(hours=1),
            ),
        )

        result = asyncio.run(_check_garage_async(_make_settings()))

        assert result == "OK"
        # No staleness alert
        for call in notifier.send_error_alert.call_args_list:
            assert "Possibile interruzione" not in call[0][0]

    @patch("garage_monitor.main.BlinkClient")
    @patch("garage_monitor.main.GeminiAnalyzer")
    @patch("garage_monitor.main.TelegramNotifier")
    @patch("garage_monitor.main.FirestoreStore")
    def test_first_run_no_staleness_check(
        self, mock_store_cls, mock_tg_cls, mock_gem_cls, mock_blink_cls
    ):
        """state is None (first run) -> no staleness check."""
        _, notifier, _, _ = _setup_mocks(
            mock_store_cls, mock_tg_cls, mock_gem_cls, mock_blink_cls,
            state=None,
        )

        result = asyncio.run(_check_garage_async(_make_settings()))

        assert result == "OK"
        notifier.send_error_alert.assert_not_called()

class TestNightAlert:
    @patch("garage_monitor.main._is_night_time", return_value=True)
    @patch("garage_monitor.main.BlinkClient")
    @patch("garage_monitor.main.GeminiAnalyzer")
    @patch("garage_monitor.main.TelegramNotifier")
    @patch("garage_monitor.main.FirestoreStore")
    def test_open_at_night_sends_night_alert(
        self, mock_store_cls, mock_tg_cls, mock_gem_cls, mock_blink_cls, _mock_night
    ):
        """Garage opens at night -> send_night_alert called instead of send_status_change."""
        now = datetime.now(timezone.utc)
        _, notifier, _, _ = _setup_mocks(
            mock_store_cls, mock_tg_cls, mock_gem_cls, mock_blink_cls,
            state=GarageState(
                current_status=GarageStatus.CLOSED,
                last_check_time=now,
                last_change_time=now,
                pending_status=GarageStatus.OPEN,
                pending_count=1,
            ),
            analysis_status=GarageStatus.OPEN,
            analysis_confidence=0.9,
            analysis_reasoning="Door raised",
        )

        result = asyncio.run(_check_garage_async(_make_settings()))

        assert result == "OK"
        notifier.send_night_alert.assert_called_once()
        call_kwargs = notifier.send_night_alert.call_args[1]
        assert call_kwargs["confidence"] == 0.9
        assert call_kwargs["reasoning"] == "Door raised"
        notifier.send_status_change.assert_not_called()

    @patch("garage_monitor.main._is_night_time", return_value=False)
    @patch("garage_monitor.main.BlinkClient")
    @patch("garage_monitor.main.GeminiAnalyzer")
    @patch("garage_monitor.main.TelegramNotifier")
    @patch("garage_monitor.main.FirestoreStore")
    def test_open_during_day_sends_status_change(
        self, mock_store_cls, mock_tg_cls, mock_gem_cls, mock_blink_cls, _mock_night
    ):
        """Garage opens during day -> send_status_change called (not night_alert)."""
        now = datetime.now(timezone.utc)
        _, notifier, _, _ = _setup_mocks(
            mock_store_cls, mock_tg_cls, mock_gem_cls, mock_blink_cls,
            state=GarageState(
                current_status=GarageStatus.CLOSED,
                last_check_time=now,
                last_change_time=now,
                pending_status=GarageStatus.OPEN,
                pending_count=1,
            ),
            analysis_status=GarageStatus.OPEN,
            analysis_confidence=0.9,
            analysis_reasoning="Door raised",
        )

        result = asyncio.run(_check_garage_async(_make_settings()))

        assert result == "OK"
        notifier.send_status_change.assert_called_once()
        notifier.send_night_alert.assert_not_called()


class TestFinalWarning:
    @patch("garage_monitor.main.BlinkClient")
    @patch("garage_monitor.main.GeminiAnalyzer")
    @patch("garage_monitor.main.TelegramNotifier")
    @patch("garage_monitor.main.FirestoreStore")
    def test_final_warning_sent_after_60_min(
        self, mock_store_cls, mock_tg_cls, mock_gem_cls, mock_blink_cls
    ):
        """Garage open >60 min, final warning not yet sent -> send_final_warning called."""
        now = datetime.now(timezone.utc)
        store, notifier, _, _ = _setup_mocks(
            mock_store_cls, mock_tg_cls, mock_gem_cls, mock_blink_cls,
            state=GarageState(
                current_status=GarageStatus.OPEN,
                last_check_time=now - timedelta(minutes=5),
                last_change_time=now - timedelta(minutes=65),
                last_reminder_time=now - timedelta(minutes=20),
                last_final_warning_sent=False,
            ),
            analysis_status=GarageStatus.OPEN,
            analysis_confidence=0.9,
            analysis_reasoning="Door open",
        )

        result = asyncio.run(_check_garage_async(_make_settings()))

        assert result == "OK"
        notifier.send_final_warning.assert_called_once()
        call_args = notifier.send_final_warning.call_args[0]
        assert call_args[0] >= 65  # minutes_open
        # Flag is set to True
        saved_state = store.save_state.call_args[0][0]
        assert saved_state.last_final_warning_sent is True

    @patch("garage_monitor.main.BlinkClient")
    @patch("garage_monitor.main.GeminiAnalyzer")
    @patch("garage_monitor.main.TelegramNotifier")
    @patch("garage_monitor.main.FirestoreStore")
    def test_final_warning_not_duplicated(
        self, mock_store_cls, mock_tg_cls, mock_gem_cls, mock_blink_cls
    ):
        """Garage open >60 min, final warning already sent -> no duplicate."""
        now = datetime.now(timezone.utc)
        _, notifier, _, _ = _setup_mocks(
            mock_store_cls, mock_tg_cls, mock_gem_cls, mock_blink_cls,
            state=GarageState(
                current_status=GarageStatus.OPEN,
                last_check_time=now - timedelta(minutes=5),
                last_change_time=now - timedelta(minutes=65),
                last_reminder_time=now - timedelta(minutes=20),
                last_final_warning_sent=True,
            ),
            analysis_status=GarageStatus.OPEN,
            analysis_confidence=0.9,
            analysis_reasoning="Door open",
        )

        result = asyncio.run(_check_garage_async(_make_settings()))

        assert result == "OK"
        notifier.send_final_warning.assert_not_called()

    @patch("garage_monitor.main.BlinkClient")
    @patch("garage_monitor.main.GeminiAnalyzer")
    @patch("garage_monitor.main.TelegramNotifier")
    @patch("garage_monitor.main.FirestoreStore")
    def test_final_warning_resets_on_close(
        self, mock_store_cls, mock_tg_cls, mock_gem_cls, mock_blink_cls
    ):
        """Garage closes -> last_final_warning_sent resets to False."""
        now = datetime.now(timezone.utc)
        store, notifier, _, _ = _setup_mocks(
            mock_store_cls, mock_tg_cls, mock_gem_cls, mock_blink_cls,
            state=GarageState(
                current_status=GarageStatus.OPEN,
                last_check_time=now - timedelta(minutes=5),
                last_change_time=now - timedelta(minutes=70),
                last_reminder_time=now - timedelta(minutes=20),
                last_final_warning_sent=True,
                pending_status=GarageStatus.CLOSED,
                pending_count=1,
            ),
            analysis_status=GarageStatus.CLOSED,
            analysis_confidence=0.95,
            analysis_reasoning="Door closed",
        )

        result = asyncio.run(_check_garage_async(_make_settings()))

        assert result == "OK"
        saved_state = store.save_state.call_args[0][0]
        assert saved_state.last_final_warning_sent is False


class TestEventHistory:
    @patch("garage_monitor.main.BlinkClient")
    @patch("garage_monitor.main.GeminiAnalyzer")
    @patch("garage_monitor.main.TelegramNotifier")
    @patch("garage_monitor.main.FirestoreStore")
    def test_status_change_logs_event(
        self, mock_store_cls, mock_tg_cls, mock_gem_cls, mock_blink_cls
    ):
        """Status change -> store.save_event called with correct data."""
        now = datetime.now(timezone.utc)
        store, _, _, _ = _setup_mocks(
            mock_store_cls, mock_tg_cls, mock_gem_cls, mock_blink_cls,
            state=GarageState(
                current_status=GarageStatus.CLOSED,
                last_check_time=now,
                last_change_time=now - timedelta(minutes=30),
                pending_status=GarageStatus.OPEN,
                pending_count=1,
            ),
            analysis_status=GarageStatus.OPEN,
            analysis_confidence=0.9,
            analysis_reasoning="Door raised",
        )

        result = asyncio.run(_check_garage_async(_make_settings()))

        assert result == "OK"
        store.save_event.assert_called_once()
        event = store.save_event.call_args[0][0]
        assert event["old_status"] == "closed"
        assert event["new_status"] == "open"
        assert event["confidence"] == 0.9
        assert event["reasoning"] == "Door raised"
        assert "timestamp" in event
        assert "expire_at" in event

    @patch("garage_monitor.main.BlinkClient")
    @patch("garage_monitor.main.GeminiAnalyzer")
    @patch("garage_monitor.main.TelegramNotifier")
    @patch("garage_monitor.main.FirestoreStore")
    def test_close_event_has_duration(
        self, mock_store_cls, mock_tg_cls, mock_gem_cls, mock_blink_cls
    ):
        """Close event -> duration_seconds is calculated."""
        now = datetime.now(timezone.utc)
        store, _, _, _ = _setup_mocks(
            mock_store_cls, mock_tg_cls, mock_gem_cls, mock_blink_cls,
            state=GarageState(
                current_status=GarageStatus.OPEN,
                last_check_time=now - timedelta(minutes=5),
                last_change_time=now - timedelta(minutes=30),
                pending_status=GarageStatus.CLOSED,
                pending_count=1,
            ),
            analysis_status=GarageStatus.CLOSED,
            analysis_confidence=0.95,
            analysis_reasoning="Door closed",
        )

        result = asyncio.run(_check_garage_async(_make_settings()))

        assert result == "OK"
        store.save_event.assert_called_once()
        event = store.save_event.call_args[0][0]
        assert event["new_status"] == "closed"
        assert event["duration_seconds"] is not None
        assert event["duration_seconds"] >= 30 * 60  # at least 30 min in seconds

    @patch("garage_monitor.main.BlinkClient")
    @patch("garage_monitor.main.GeminiAnalyzer")
    @patch("garage_monitor.main.TelegramNotifier")
    @patch("garage_monitor.main.FirestoreStore")
    def test_open_event_no_duration(
        self, mock_store_cls, mock_tg_cls, mock_gem_cls, mock_blink_cls
    ):
        """Open event -> duration_seconds is None."""
        now = datetime.now(timezone.utc)
        store, _, _, _ = _setup_mocks(
            mock_store_cls, mock_tg_cls, mock_gem_cls, mock_blink_cls,
            state=GarageState(
                current_status=GarageStatus.CLOSED,
                last_check_time=now,
                last_change_time=now,
                pending_status=GarageStatus.OPEN,
                pending_count=1,
            ),
            analysis_status=GarageStatus.OPEN,
            analysis_confidence=0.9,
            analysis_reasoning="Door raised",
        )

        result = asyncio.run(_check_garage_async(_make_settings()))

        assert result == "OK"
        store.save_event.assert_called_once()
        event = store.save_event.call_args[0][0]
        assert event["new_status"] == "open"
        assert event["duration_seconds"] is None

    @patch("garage_monitor.main.BlinkClient")
    @patch("garage_monitor.main.GeminiAnalyzer")
    @patch("garage_monitor.main.TelegramNotifier")
    @patch("garage_monitor.main.FirestoreStore")
    def test_no_event_on_same_status(
        self, mock_store_cls, mock_tg_cls, mock_gem_cls, mock_blink_cls
    ):
        """No status change -> no event saved."""
        now = datetime.now(timezone.utc)
        store, _, _, _ = _setup_mocks(
            mock_store_cls, mock_tg_cls, mock_gem_cls, mock_blink_cls,
            state=GarageState(
                current_status=GarageStatus.CLOSED,
                last_check_time=now,
                last_change_time=now,
            ),
            analysis_status=GarageStatus.CLOSED,
            analysis_confidence=0.95,
            analysis_reasoning="Door closed",
        )

        result = asyncio.run(_check_garage_async(_make_settings()))

        assert result == "OK"
        store.save_event.assert_not_called()
