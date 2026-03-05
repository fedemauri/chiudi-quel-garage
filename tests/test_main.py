import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from garage_monitor.main import _check_garage_async
from garage_monitor.models import GarageState, GarageStatus, GeminiAnalysisResult


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


class TestCheckGarage:
    @patch("garage_monitor.main.BlinkClient")
    @patch("garage_monitor.main.GeminiAnalyzer")
    @patch("garage_monitor.main.TelegramNotifier")
    @patch("garage_monitor.main.FirestoreStore")
    def test_first_run(self, mock_store_cls, mock_tg_cls, mock_gem_cls, mock_blink_cls):
        store = MagicMock()
        store.get_state.return_value = None
        store.get_blink_credentials.return_value = {"token": "abc"}
        mock_store_cls.return_value = store

        notifier = MagicMock()
        mock_tg_cls.return_value = notifier

        analyzer = MagicMock()
        analyzer.analyze.return_value = GeminiAnalysisResult(
            status=GarageStatus.CLOSED, confidence=0.95, reasoning="Door closed"
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
        notifier.send_monitor_started.assert_called_once()
        store.save_state.assert_called_once()
        store.save_blink_credentials.assert_called_once()

    @patch("garage_monitor.main.BlinkClient")
    @patch("garage_monitor.main.GeminiAnalyzer")
    @patch("garage_monitor.main.TelegramNotifier")
    @patch("garage_monitor.main.FirestoreStore")
    def test_status_change(self, mock_store_cls, mock_tg_cls, mock_gem_cls, mock_blink_cls):
        now = datetime.now(timezone.utc)
        store = MagicMock()
        store.get_state.return_value = GarageState(
            current_status=GarageStatus.CLOSED,
            last_check_time=now,
            last_change_time=now,
            consecutive_errors=0,
        )
        store.get_blink_credentials.return_value = {"token": "abc"}
        mock_store_cls.return_value = store

        notifier = MagicMock()
        mock_tg_cls.return_value = notifier

        analyzer = MagicMock()
        analyzer.analyze.return_value = GeminiAnalysisResult(
            status=GarageStatus.OPEN, confidence=0.9, reasoning="Door raised"
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
        notifier.send_status_change.assert_called_once()

    @patch("garage_monitor.main.BlinkClient")
    @patch("garage_monitor.main.GeminiAnalyzer")
    @patch("garage_monitor.main.TelegramNotifier")
    @patch("garage_monitor.main.FirestoreStore")
    def test_low_confidence_no_change(self, mock_store_cls, mock_tg_cls, mock_gem_cls, mock_blink_cls):
        now = datetime.now(timezone.utc)
        store = MagicMock()
        store.get_state.return_value = GarageState(
            current_status=GarageStatus.CLOSED,
            last_check_time=now,
            last_change_time=now,
        )
        store.get_blink_credentials.return_value = {"token": "abc"}
        mock_store_cls.return_value = store

        notifier = MagicMock()
        mock_tg_cls.return_value = notifier

        analyzer = MagicMock()
        analyzer.analyze.return_value = GeminiAnalysisResult(
            status=GarageStatus.OPEN, confidence=0.3, reasoning="Too dark"
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
        store = MagicMock()
        store.get_state.return_value = GarageState(
            current_status=GarageStatus.CLOSED,
            last_check_time=now,
            last_change_time=now,
            consecutive_errors=2,
        )
        store.get_blink_credentials.return_value = {"token": "abc"}
        mock_store_cls.return_value = store

        notifier = MagicMock()
        mock_tg_cls.return_value = notifier

        analyzer = MagicMock()
        analyzer.analyze.side_effect = RuntimeError("Gemini down")
        mock_gem_cls.return_value = analyzer

        blink = MagicMock()
        blink.connect = AsyncMock()
        blink.take_snapshot = AsyncMock(return_value=b"jpeg-data")
        blink.close = AsyncMock()
        mock_blink_cls.return_value = blink

        result = asyncio.run(_check_garage_async(_make_settings()))

        assert result == "ERROR"
        notifier.send_error_alert.assert_called_once()
