from unittest.mock import MagicMock, patch

import httpx
import pytest

from garage_monitor.telegram_notifier import TelegramNotifier


def _make_response(status_code):
    resp = MagicMock()
    resp.status_code = status_code
    if status_code >= 400:
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "error", request=MagicMock(), response=resp
        )
    else:
        resp.raise_for_status.return_value = None
    return resp


class TestSendMessageMarkdownFallback:
    @patch("garage_monitor.telegram_notifier.httpx.Client")
    def test_falls_back_to_plain_text_when_markdown_rejected(self, mock_client_cls):
        """A 400 (Markdown parse failure) retries as plain text so the alert is delivered."""
        client = MagicMock()
        mock_client_cls.return_value.__enter__.return_value = client
        client.post.side_effect = [_make_response(400), _make_response(200)]

        notifier = TelegramNotifier("token", "chat")
        notifier.send_error_alert("Camera 'Garage' not found. Available: []")

        assert client.post.call_count == 2
        first_payload = client.post.call_args_list[0].kwargs["json"]
        second_payload = client.post.call_args_list[1].kwargs["json"]
        assert first_payload["parse_mode"] == "Markdown"
        assert "parse_mode" not in second_payload
        # The raw error text is preserved in the delivered (plain-text) message.
        assert "Camera 'Garage' not found" in second_payload["text"]

    @patch("garage_monitor.telegram_notifier.httpx.Client")
    def test_uses_markdown_when_accepted(self, mock_client_cls):
        """When Markdown is valid, the message is sent once with parse_mode set."""
        client = MagicMock()
        mock_client_cls.return_value.__enter__.return_value = client
        client.post.side_effect = [_make_response(200)]

        notifier = TelegramNotifier("token", "chat")
        notifier.send_command_response("*ciao*")

        assert client.post.call_count == 1
        assert client.post.call_args.kwargs["json"]["parse_mode"] == "Markdown"

    @patch("garage_monitor.telegram_notifier.httpx.Client")
    def test_plain_text_failure_still_raises(self, mock_client_cls):
        """If even the plain-text retry fails, the error propagates (no silent drop)."""
        client = MagicMock()
        mock_client_cls.return_value.__enter__.return_value = client
        client.post.side_effect = [_make_response(400), _make_response(400)]

        notifier = TelegramNotifier("token", "chat")
        with pytest.raises(httpx.HTTPStatusError):
            notifier.send_error_alert("boom")

    @patch("garage_monitor.telegram_notifier.httpx.Client")
    def test_non_400_error_raises_without_retry(self, mock_client_cls):
        """A non-400 error (e.g. 500) is a real failure and must not trigger a plain-text retry."""
        client = MagicMock()
        mock_client_cls.return_value.__enter__.return_value = client
        client.post.side_effect = [_make_response(500)]

        notifier = TelegramNotifier("token", "chat")
        with pytest.raises(httpx.HTTPStatusError):
            notifier.send_error_alert("boom")
        assert client.post.call_count == 1
