import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from garage_monitor.gemini_analyzer import GeminiParseError
from garage_monitor.main import (
    _check_garage_async,
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
        "gemini_api_key": "fake-key",
        "gemini_model": "gemini-2.5-flash-lite",
        "confidence_threshold": 0.7,
        "telegram_bot_token": "fake-token",
        "telegram_chat_id": "123",
        "gcp_project_id": "test-project",
        "firestore_collection": "garage_monitor",
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

    @patch("garage_monitor.main.BlinkClient")
    @patch("garage_monitor.main.GeminiAnalyzer")
    @patch("garage_monitor.main.TelegramNotifier")
    @patch("garage_monitor.main.FirestoreStore")
    def test_status_change(self, mock_store_cls, mock_tg_cls, mock_gem_cls, mock_blink_cls):
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
    def test_still_open_no_reminder_too_soon(
        self, mock_store_cls, mock_tg_cls, mock_gem_cls, mock_blink_cls
    ):
        """Garage aperto da 10 min -> nessun reminder (< 15 min da ultimo reminder)."""
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
