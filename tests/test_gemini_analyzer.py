import json
from unittest.mock import MagicMock, patch

import pytest

from garage_monitor.gemini_analyzer import GeminiAnalyzer
from garage_monitor.models import GarageStatus


def _mock_response(status: str, confidence: float, reasoning: str = "test"):
    resp = MagicMock()
    resp.text = json.dumps(
        {"status": status, "confidence": confidence, "reasoning": reasoning}
    )
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
    def test_analyze_invalid_json(self, mock_genai):
        client = MagicMock()
        mock_genai.Client.return_value = client
        resp = MagicMock()
        resp.text = "not json"
        client.models.generate_content.return_value = resp

        analyzer = GeminiAnalyzer(api_key="fake-key")
        with pytest.raises(json.JSONDecodeError):
            analyzer.analyze(b"fake-jpeg")
