import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from garage_monitor.main import check_garage, _handle_telegram_webhook
from garage_monitor.models import GarageState, GarageStatus, UsageStats
from garage_monitor.telegram_handler import handle_command


def _make_settings(**overrides):
    defaults = {
        "blink_username": "test@test.com",
        "blink_password": "pass",
        "blink_camera_name": "Garage",
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


def _make_request(content_type=None, json_body=None, args=None, headers=None):
    request = MagicMock()
    request.content_type = content_type
    request.get_json.return_value = json_body
    request.args = args or {}
    request.headers = headers or {}
    return request


class TestWebhookRouting:
    @patch("garage_monitor.telegram_handler.TelegramNotifier")
    @patch("garage_monitor.telegram_handler.FirestoreStore")
    @patch("garage_monitor.main.Settings")
    def test_webhook_recognized(self, mock_settings_cls, mock_store_cls, mock_tg_cls):
        """JSON body with 'message' field routes to webhook handler."""
        settings = _make_settings()
        mock_settings_cls.return_value = settings

        request = _make_request(
            content_type="application/json",
            json_body={"message": {"chat": {"id": 123}, "text": "/stato"}},
            headers={"X-Telegram-Bot-Api-Secret-Token": "test-secret"},
        )

        store = MagicMock()
        store.get_state.return_value = GarageState(
            current_status=GarageStatus.CLOSED,
            last_check_time=datetime.now(timezone.utc),
            last_change_time=datetime.now(timezone.utc),
        )
        mock_store_cls.return_value = store
        mock_tg_cls.return_value = MagicMock()

        result_text, status_code = check_garage(request)
        assert status_code == 200
        assert result_text == "STATO_SENT"

    @patch("garage_monitor.main.Settings")
    def test_invalid_secret_returns_403(self, mock_settings_cls):
        """Invalid webhook secret returns 403."""
        settings = _make_settings()
        mock_settings_cls.return_value = settings

        request = _make_request(
            content_type="application/json",
            json_body={"message": {"chat": {"id": 123}, "text": "/stato"}},
            headers={"X-Telegram-Bot-Api-Secret-Token": "wrong-secret"},
        )

        result_text, status_code = check_garage(request)
        assert status_code == 403
        assert result_text == "INVALID_SECRET"

    @patch("garage_monitor.main.Settings")
    def test_empty_secret_disables_webhook(self, mock_settings_cls):
        """Empty webhook secret in config disables webhook."""
        settings = _make_settings(telegram_webhook_secret="")
        mock_settings_cls.return_value = settings

        request = _make_request(
            content_type="application/json",
            json_body={"message": {"chat": {"id": 123}, "text": "/stato"}},
            headers={},
        )

        result_text, status_code = check_garage(request)
        assert status_code == 403
        assert result_text == "WEBHOOK_DISABLED"

    @patch("garage_monitor.main.Settings")
    def test_wrong_chat_id_returns_200(self, mock_settings_cls):
        """Wrong chat_id returns 200 (to avoid Telegram retries)."""
        settings = _make_settings()
        mock_settings_cls.return_value = settings

        request = _make_request(
            content_type="application/json",
            json_body={"message": {"chat": {"id": 999}, "text": "/stato"}},
            headers={"X-Telegram-Bot-Api-Secret-Token": "test-secret"},
        )

        result_text, status_code = check_garage(request)
        assert status_code == 200
        assert result_text == "OK"

    @patch("garage_monitor.main._check_garage_async", return_value="OK")
    @patch("garage_monitor.main.Settings")
    def test_normal_scheduler_still_works(self, mock_settings_cls, mock_async):
        """Non-JSON request routes to normal scheduler check."""
        settings = _make_settings()
        mock_settings_cls.return_value = settings

        request = _make_request(content_type=None, args={})

        result_text, status_code = check_garage(request)
        assert result_text == "OK"
        assert status_code == 200

    @patch("garage_monitor.main._send_usage_report")
    @patch("garage_monitor.main.Settings")
    def test_report_action_still_works(self, mock_settings_cls, mock_report):
        """?action=report still works with webhook routing."""
        settings = _make_settings()
        mock_settings_cls.return_value = settings

        request = _make_request(content_type=None, args={"action": "report"})

        result_text, status_code = check_garage(request)
        assert result_text == "REPORT_SENT"
        assert status_code == 200
        mock_report.assert_called_once()


class TestCommands:
    @patch("garage_monitor.telegram_handler.TelegramNotifier")
    @patch("garage_monitor.telegram_handler.FirestoreStore")
    def test_stato_command(self, mock_store_cls, mock_tg_cls):
        store = MagicMock()
        now = datetime.now(timezone.utc)
        store.get_state.return_value = GarageState(
            current_status=GarageStatus.CLOSED,
            last_check_time=now,
            last_change_time=now - timedelta(hours=2),
        )
        mock_store_cls.return_value = store
        notifier = MagicMock()
        mock_tg_cls.return_value = notifier

        result = handle_command({"message": {"text": "/stato"}}, _make_settings())

        assert result == "STATO_SENT"
        notifier.send_current_status.assert_called_once()

    @patch("garage_monitor.telegram_handler.BlinkClient")
    @patch("garage_monitor.telegram_handler.TelegramNotifier")
    @patch("garage_monitor.telegram_handler.FirestoreStore")
    def test_foto_command(self, mock_store_cls, mock_tg_cls, mock_blink_cls):
        store = MagicMock()
        store.get_blink_credentials.return_value = {"token": "abc"}
        mock_store_cls.return_value = store

        blink = MagicMock()
        blink.connect = AsyncMock()
        blink.take_snapshot = AsyncMock(return_value=b"jpeg-data")
        blink.get_updated_credentials.return_value = {"token": "refreshed"}
        blink.close = AsyncMock()
        mock_blink_cls.return_value = blink

        notifier = MagicMock()
        mock_tg_cls.return_value = notifier

        result = handle_command({"message": {"text": "/foto"}}, _make_settings())

        assert result == "FOTO_SENT"
        notifier.send_photo.assert_called_once()

    @patch("garage_monitor.main.TelegramNotifier")
    @patch("garage_monitor.main.FirestoreStore")
    def test_report_command(self, mock_store_cls, mock_tg_cls):
        """_cmd_report delegates to main._send_usage_report, so mock at main module level."""
        store = MagicMock()
        store.get_usage_stats.return_value = UsageStats(period="2026_03")
        mock_store_cls.return_value = store
        notifier = MagicMock()
        mock_tg_cls.return_value = notifier

        result = handle_command({"message": {"text": "/report"}}, _make_settings())

        assert result == "REPORT_SENT"
        notifier.send_usage_report.assert_called_once()

    @patch("garage_monitor.telegram_handler.TelegramNotifier")
    @patch("garage_monitor.telegram_handler.FirestoreStore")
    def test_muto_default_2h(self, mock_store_cls, mock_tg_cls):
        store = MagicMock()
        now = datetime.now(timezone.utc)
        store.get_state.return_value = GarageState(
            current_status=GarageStatus.CLOSED,
            last_check_time=now,
            last_change_time=now,
        )
        mock_store_cls.return_value = store
        notifier = MagicMock()
        mock_tg_cls.return_value = notifier

        result = handle_command({"message": {"text": "/muto"}}, _make_settings())

        assert result == "MUTED"
        saved_state = store.save_state.call_args[0][0]
        assert saved_state.muted_until is not None
        # Default 2h
        delta = saved_state.muted_until - now
        assert 7100 < delta.total_seconds() < 7300

    @patch("garage_monitor.telegram_handler.TelegramNotifier")
    @patch("garage_monitor.telegram_handler.FirestoreStore")
    def test_muto_custom_hours(self, mock_store_cls, mock_tg_cls):
        store = MagicMock()
        now = datetime.now(timezone.utc)
        store.get_state.return_value = GarageState(
            current_status=GarageStatus.CLOSED,
            last_check_time=now,
            last_change_time=now,
        )
        mock_store_cls.return_value = store
        notifier = MagicMock()
        mock_tg_cls.return_value = notifier

        result = handle_command({"message": {"text": "/muto 5h"}}, _make_settings())

        assert result == "MUTED"
        saved_state = store.save_state.call_args[0][0]
        delta = saved_state.muted_until - now
        assert 17900 < delta.total_seconds() < 18100  # ~5h

    @patch("garage_monitor.telegram_handler.TelegramNotifier")
    @patch("garage_monitor.telegram_handler.FirestoreStore")
    def test_muto_capped_at_24h(self, mock_store_cls, mock_tg_cls):
        store = MagicMock()
        now = datetime.now(timezone.utc)
        store.get_state.return_value = GarageState(
            current_status=GarageStatus.CLOSED,
            last_check_time=now,
            last_change_time=now,
        )
        mock_store_cls.return_value = store
        notifier = MagicMock()
        mock_tg_cls.return_value = notifier

        result = handle_command({"message": {"text": "/muto 48h"}}, _make_settings())

        assert result == "MUTED"
        saved_state = store.save_state.call_args[0][0]
        delta = saved_state.muted_until - now
        assert delta.total_seconds() <= 24 * 3600 + 10

    @patch("garage_monitor.telegram_handler.TelegramNotifier")
    @patch("garage_monitor.telegram_handler.FirestoreStore")
    def test_smuto_command(self, mock_store_cls, mock_tg_cls):
        store = MagicMock()
        now = datetime.now(timezone.utc)
        store.get_state.return_value = GarageState(
            current_status=GarageStatus.CLOSED,
            last_check_time=now,
            last_change_time=now,
            muted_until=now + timedelta(hours=2),
        )
        mock_store_cls.return_value = store
        notifier = MagicMock()
        mock_tg_cls.return_value = notifier

        result = handle_command({"message": {"text": "/smuto"}}, _make_settings())

        assert result == "UNMUTED"
        saved_state = store.save_state.call_args[0][0]
        assert saved_state.muted_until is None

    @patch("garage_monitor.telegram_handler.TelegramNotifier")
    @patch("garage_monitor.telegram_handler.FirestoreStore")
    def test_storico_command(self, mock_store_cls, mock_tg_cls):
        store = MagicMock()
        store.get_recent_events.return_value = []
        mock_store_cls.return_value = store
        notifier = MagicMock()
        mock_tg_cls.return_value = notifier

        result = handle_command({"message": {"text": "/storico"}}, _make_settings())

        assert result == "STORICO_SENT"
        store.get_recent_events.assert_called_once_with(10)
        notifier.send_history.assert_called_once()

    @patch("garage_monitor.telegram_handler.TelegramNotifier")
    @patch("garage_monitor.telegram_handler.FirestoreStore")
    def test_storico_error_handled(self, mock_store_cls, mock_tg_cls):
        store = MagicMock()
        store.get_recent_events.side_effect = Exception("Missing index")
        mock_store_cls.return_value = store
        notifier = MagicMock()
        mock_tg_cls.return_value = notifier

        result = handle_command({"message": {"text": "/storico"}}, _make_settings())

        assert result == "STORICO_ERROR"
        notifier.send_command_response.assert_called_once()

    @patch("garage_monitor.telegram_handler.TelegramNotifier")
    @patch("garage_monitor.telegram_handler.FirestoreStore")
    def test_unknown_command(self, mock_store_cls, mock_tg_cls):
        store = MagicMock()
        mock_store_cls.return_value = store
        notifier = MagicMock()
        mock_tg_cls.return_value = notifier

        result = handle_command({"message": {"text": "/invalid"}}, _make_settings())

        assert result == "UNKNOWN_COMMAND"
        notifier.send_command_response.assert_called_once()


class TestMuteReminders:
    @patch("garage_monitor.main.BlinkClient")
    @patch("garage_monitor.main.GeminiAnalyzer")
    @patch("garage_monitor.main.TelegramNotifier")
    @patch("garage_monitor.main.FirestoreStore")
    def test_muted_skips_reminder(
        self, mock_store_cls, mock_tg_cls, mock_gem_cls, mock_blink_cls
    ):
        """Mute active -> reminder not sent."""
        from garage_monitor.main import _check_garage_async
        from garage_monitor.models import GeminiAnalysisResult

        now = datetime.now(timezone.utc)
        store = MagicMock()
        store.get_state.return_value = GarageState(
            current_status=GarageStatus.OPEN,
            last_check_time=now - timedelta(minutes=5),
            last_change_time=now - timedelta(minutes=20),
            last_reminder_time=None,
            muted_until=now + timedelta(hours=1),
        )
        store.get_blink_credentials.return_value = {"token": "abc"}
        mock_store_cls.return_value = store

        notifier = MagicMock()
        mock_tg_cls.return_value = notifier

        analyzer = MagicMock()
        analyzer.analyze.return_value = GeminiAnalysisResult(
            status=GarageStatus.OPEN, confidence=0.9, reasoning="Door open"
        )
        mock_gem_cls.return_value = analyzer

        blink = MagicMock()
        blink.connect = AsyncMock()
        blink.take_snapshot = AsyncMock(return_value=b"jpeg-data")
        blink.get_updated_credentials.return_value = {"token": "refreshed"}
        blink.close = AsyncMock()
        mock_blink_cls.return_value = blink

        result = asyncio.run(_check_garage_async(_make_settings()))

        assert result == "OK"
        notifier.send_still_open_reminder.assert_not_called()

    @patch("garage_monitor.main.BlinkClient")
    @patch("garage_monitor.main.GeminiAnalyzer")
    @patch("garage_monitor.main.TelegramNotifier")
    @patch("garage_monitor.main.FirestoreStore")
    def test_expired_mute_auto_clears_and_reminds(
        self, mock_store_cls, mock_tg_cls, mock_gem_cls, mock_blink_cls
    ):
        """Expired mute -> auto-clear + reminder sent."""
        from garage_monitor.main import _check_garage_async
        from garage_monitor.models import GeminiAnalysisResult

        now = datetime.now(timezone.utc)
        store = MagicMock()
        store.get_state.return_value = GarageState(
            current_status=GarageStatus.OPEN,
            last_check_time=now - timedelta(minutes=5),
            last_change_time=now - timedelta(minutes=20),
            last_reminder_time=None,
            muted_until=now - timedelta(minutes=5),  # expired
        )
        store.get_blink_credentials.return_value = {"token": "abc"}
        mock_store_cls.return_value = store

        notifier = MagicMock()
        mock_tg_cls.return_value = notifier

        analyzer = MagicMock()
        analyzer.analyze.return_value = GeminiAnalysisResult(
            status=GarageStatus.OPEN, confidence=0.9, reasoning="Door open"
        )
        mock_gem_cls.return_value = analyzer

        blink = MagicMock()
        blink.connect = AsyncMock()
        blink.take_snapshot = AsyncMock(return_value=b"jpeg-data")
        blink.get_updated_credentials.return_value = {"token": "refreshed"}
        blink.close = AsyncMock()
        mock_blink_cls.return_value = blink

        result = asyncio.run(_check_garage_async(_make_settings()))

        assert result == "OK"
        notifier.send_still_open_reminder.assert_called_once()
        # muted_until should be cleared
        saved_state = store.save_state.call_args[0][0]
        assert saved_state.muted_until is None
