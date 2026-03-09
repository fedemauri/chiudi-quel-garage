"""Analyze a garage JPEG image using a local TFLite MobileNetV2 classifier.

Drop-in replacement for GeminiAnalyzer. Returns the same
GeminiAnalysisResult so the rest of the pipeline is unchanged.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PIL import Image

from garage_monitor.models import GarageStatus, GeminiAnalysisResult

logger = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "model" / "garage_classifier.tflite"
IMG_SIZE = (224, 224)


class TFLiteAnalyzer:
    """Classifies garage door state from a JPEG image via local TFLite model."""

    def __init__(self, model_path: str | None = None) -> None:
        try:
            from ai_edge_litert import interpreter as tflite
        except ImportError:
            try:
                import tflite_runtime.interpreter as tflite
            except ImportError:
                import tensorflow.lite as tflite

        path = model_path or str(MODEL_PATH)
        self._interpreter = tflite.Interpreter(model_path=path)
        self._interpreter.allocate_tensors()
        self._input_details = self._interpreter.get_input_details()
        self._output_details = self._interpreter.get_output_details()
        logger.info("TFLite model loaded from %s", path)

    def analyze(self, image_bytes: bytes) -> GeminiAnalysisResult:
        """Analyze garage image and return status classification.

        Args:
            image_bytes: Raw JPEG bytes of the garage image.

        Returns:
            GeminiAnalysisResult with status, confidence, and reasoning.
        """
        import io

        img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize(IMG_SIZE)
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        self._interpreter.set_tensor(self._input_details[0]["index"], img_array)
        self._interpreter.invoke()
        output = self._interpreter.get_tensor(self._output_details[0]["index"])

        prob_open = float(output[0][0])
        if prob_open > 0.5:
            status = GarageStatus.OPEN
            confidence = prob_open
            reasoning = f"Porta aperta (p={prob_open:.2f})"
        else:
            status = GarageStatus.CLOSED
            confidence = 1.0 - prob_open
            reasoning = f"Porta chiusa (p={1.0 - prob_open:.2f})"

        logger.info(
            "TFLite: %s (confidence=%.2f, raw=%.4f)",
            status.value, confidence, prob_open,
        )

        return GeminiAnalysisResult(
            status=status,
            confidence=confidence,
            reasoning=reasoning,
            input_tokens=0,
            output_tokens=0,
        )
