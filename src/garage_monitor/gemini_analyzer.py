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


class GeminiParseError(ValueError):
    """Raised when Gemini response cannot be parsed."""

PROMPT = """Analizza questa immagine di un garage con porta sezionale.
Determina se la porta del garage è APERTA o CHIUSA.

OPEN: porta sollevata (parzialmente o totalmente), interno visibile.
CLOSED: porta completamente abbassata, superficie continua, interno non visibile.

Rispondi in JSON: {"status": "open"|"closed", "confidence": 0.0-1.0, "reasoning": "..."}
Il campo reasoning deve essere in italiano, massimo 10 parole.
Se l'immagine è troppo scura o sfocata, imposta confidence sotto 0.5."""


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

        try:
            data = json.loads(response.text)
            result = GeminiAnalysisResult(
                status=GarageStatus(data["status"]),
                confidence=float(data["confidence"]),
                reasoning=data.get("reasoning", ""),
            )
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            raise GeminiParseError(
                f"Gemini ha risposto in modo inatteso: {e}"
            ) from e

        usage = getattr(response, "usage_metadata", None)
        result.input_tokens = getattr(usage, "prompt_token_count", 0) or 0
        result.output_tokens = getattr(usage, "candidates_token_count", 0) or 0

        logger.info(
            "Garage door classified as %s (confidence=%.2f)",
            result.status.value,
            result.confidence,
        )

        return result
