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

PROMPT = """Telecamera fissa DENTRO un garage. L'immagine può essere a colori (giorno)
o IR bianco/nero (notte). La porta sezionale è al centro. Determina: APERTA o CHIUSA.

CLOSED — i pannelli coprono il varco:
  - Pannelli orizzontali chiari formano una superficie VERTICALE continua
  - Giunture/linee scure tra i pannelli (è normale, NON è un'apertura)
  - Binario curvo visibile in alto ai lati
  - Il varco è completamente sigillato, non si vede nulla oltre la porta

OPEN — il varco è libero, la porta è retratta:
  - I pannelli NON sono una superficie verticale: sono ripiegati/retratti
    lungo il soffitto (visibili come struttura orizzontale in alto)
  - Il varco mostra un'apertura ampia: si vede rampa, esterno, pergola, cielo
  - Anche parzialmente aperta (pannelli a metà, varco parziale in basso) = OPEN

FALSI POSITIVI DA EVITARE (IMPORTANTE):
  - Le GUIDE LATERALI (binari curvi metallici) sui lati destro e sinistro della porta
    creano un gap/striscia scura tra il bordo dei pannelli e il muro. Questo è NORMALE
    a porta chiusa e NON indica apertura.
  - Una striscia scura verticale lungo il BORDO laterale della porta = guida, NON apertura.
  - "Non completamente sigillata sul lato" = è la guida laterale = CLOSED.
  - OPEN significa che il VARCO CENTRALE è libero e si vede l'ESTERNO (rampa, cielo, strada).

COME DISTINGUERE (la regola fondamentale):
  Guarda SOLO il VARCO CENTRALE della porta (ignora i bordi laterali).
  → Se è coperto da pannelli piatti verticali con linee orizzontali → CLOSED
  → Se mostra un'apertura AMPIA verso l'esterno (non una fessura laterale) → OPEN
  Nel dubbio → CLOSED.

Rispondi in JSON. Il campo reasoning deve essere in italiano, massimo 10 parole.
Se l'immagine è troppo ambigua, imposta confidence sotto 0.5."""

RESPONSE_SCHEMA = types.Schema(
    type="OBJECT",
    required=["status", "confidence", "reasoning"],
    properties={
        "status": types.Schema(type="STRING", enum=["open", "closed"]),
        "confidence": types.Schema(type="NUMBER"),
        "reasoning": types.Schema(type="STRING"),
    },
)


class GeminiAnalyzer:
    """Classifies garage door state from a JPEG image via Gemini."""

    def __init__(self, api_key: str, model: str = "gemini-2.5-flash") -> None:
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
                response_schema=RESPONSE_SCHEMA,
                temperature=0.0,
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
