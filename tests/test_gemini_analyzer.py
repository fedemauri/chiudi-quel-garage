import json
from unittest.mock import MagicMock, patch

import pytest

from garage_monitor.gemini_analyzer import GeminiAnalyzer, GeminiParseError
from garage_monitor.models import GarageStatus


def _mock_response(
    status: str,
    confidence: float,
    reasoning: str = "test",
    prompt_tokens: int = 50,
    candidates_tokens: int = 10,
):
    resp = MagicMock()
    resp.text = json.dumps(
        {"status": status, "confidence": confidence, "reasoning": reasoning}
    )
    usage = MagicMock()
    usage.prompt_token_count = prompt_tokens
    usage.candidates_token_count = candidates_tokens
    resp.usage_metadata = usage
    return resp


class TestGeminiAnalyzer:
    @patch("garage_monitor.gemini_analyzer.genai")
    def test_analyze_closed(self, mock_genai):
        client = MagicMock()
        mock_genai.Client.return_value = client
        client.models.generate_content.return_value = _mock_response(
            "closed", 0.95, "Door flush with frame"
        )

        analyzer = GeminiAnalyzer(api_key="fake-key")
        result = analyzer.analyze(b"fake-jpeg")

        assert result.status == GarageStatus.CLOSED
        assert result.confidence == 0.95
        assert result.reasoning == "Door flush with frame"

    @patch("garage_monitor.gemini_analyzer.genai")
    def test_analyze_open(self, mock_genai):
        client = MagicMock()
        mock_genai.Client.return_value = client
        client.models.generate_content.return_value = _mock_response(
            "open", 0.88, "Interior visible"
        )

        analyzer = GeminiAnalyzer(api_key="fake-key")
        result = analyzer.analyze(b"fake-jpeg")

        assert result.status == GarageStatus.OPEN
        assert result.confidence == 0.88

    @patch("garage_monitor.gemini_analyzer.genai")
    def test_analyze_low_confidence(self, mock_genai):
        client = MagicMock()
        mock_genai.Client.return_value = client
        client.models.generate_content.return_value = _mock_response(
            "closed", 0.3, "Image too dark"
        )

        analyzer = GeminiAnalyzer(api_key="fake-key")
        result = analyzer.analyze(b"fake-jpeg")

        assert result.confidence == 0.3

    @patch("garage_monitor.gemini_analyzer.genai")
    def test_analyze_invalid_json_raises_parse_error(self, mock_genai):
        client = MagicMock()
        mock_genai.Client.return_value = client
        resp = MagicMock()
        resp.text = "not json"
        client.models.generate_content.return_value = resp

        analyzer = GeminiAnalyzer(api_key="fake-key")
        with pytest.raises(GeminiParseError):
            analyzer.analyze(b"fake-jpeg")

    @patch("garage_monitor.gemini_analyzer.genai")
    def test_analyze_missing_key_raises_parse_error(self, mock_genai):
        client = MagicMock()
        mock_genai.Client.return_value = client
        resp = MagicMock()
        resp.text = json.dumps({"status": "open"})  # missing confidence
        client.models.generate_content.return_value = resp

        analyzer = GeminiAnalyzer(api_key="fake-key")
        with pytest.raises(GeminiParseError):
            analyzer.analyze(b"fake-jpeg")

    @patch("garage_monitor.gemini_analyzer.genai")
    def test_analyze_extracts_token_usage(self, mock_genai):
        client = MagicMock()
        mock_genai.Client.return_value = client
        client.models.generate_content.return_value = _mock_response(
            "closed", 0.95, "test", prompt_tokens=100, candidates_tokens=25
        )

        analyzer = GeminiAnalyzer(api_key="fake-key")
        result = analyzer.analyze(b"fake-jpeg")

        assert result.input_tokens == 100
        assert result.output_tokens == 25
