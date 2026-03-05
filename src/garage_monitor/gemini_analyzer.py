"""Analyze a garage JPEG image using Gemini 2.5 Flash-Lite.

Uses the google-genai SDK (package: google-genai) to classify whether
the garage door is open or closed, returning a structured result with
confidence score and reasoning.
"""

import json
import logging

from google import genai
from google.genai import types

from garage_monitor.models import GarageStatus, GeminiAnalysisResult

logger = logging.getLogger(__name__)

PROMPT = """Analyze this image of a garage with a sectional door.
Determine if the garage door is OPEN or CLOSED.

OPEN: door is raised (partially or fully), interior visible,
      gap visible between door bottom and floor.
CLOSED: door is fully lowered, flush with frame,
        continuous surface, no interior visible.

Respond as JSON: {"status": "open"|"closed", "confidence": 0.0-1.0, "reasoning": "..."}
If image is too dark/blurry, set confidence below 0.5."""


class GeminiAnalyzer:
    """Classifies garage door state from a JPEG image via Gemini."""

    def __init__(self, api_key: str, model: str = "gemini-2.5-flash-lite") -> None:
        self._client = genai.Client(api_key=api_key)
        self._model = model

    def analyze(self, image_bytes: bytes) -> GeminiAnalysisResult:
        """Analyze garage image and return status classification.

        Args:
            image_bytes: Raw JPEG bytes of the garage image.

        Returns:
            GeminiAnalysisResult with status, confidence, and reasoning.

        Raises:
            ValueError: If the model response cannot be parsed as valid JSON
                or contains unexpected field values.
            google.genai.errors.ClientError: On API-level failures.
        """
        logger.debug(
            "Sending %d bytes to %s for analysis",
            len(image_bytes),
            self._model,
        )

        response = self._client.models.generate_content(
            model=self._model,
            contents=[
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type="image/jpeg",
                ),
                PROMPT,
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.1,
            ),
        )

        data = json.loads(response.text)

        result = GeminiAnalysisResult(
            status=GarageStatus(data["status"]),
            confidence=float(data["confidence"]),
            reasoning=data.get("reasoning", ""),
        )

        logger.info(
            "Garage door classified as %s (confidence=%.2f)",
            result.status.value,
            result.confidence,
        )

        return result
