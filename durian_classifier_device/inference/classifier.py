"""Keras durian ripeness classifier loaded once per application."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from .exceptions import ModelConfigurationError, ModelInferenceError
from .preprocessing import preprocess_image
from .result import ClassificationResult


LOGGER = logging.getLogger(__name__)
CLASS_NAMES: list[str] = [
    "unripe",
    "ripe",
    "Overripe",
]


class KerasImageClassifier:
    """Validate, preprocess and run a three-class Keras model."""

    def __init__(
        self,
        model_path: Path,
        classes_path: Path,
        fallback_width: int,
        fallback_height: int,
        normalization: str,
        output_type: str,
        confidence_threshold: float,
    ) -> None:
        self.model_path = model_path
        self.classes_path = classes_path
        self.fallback_width = fallback_width
        self.fallback_height = fallback_height
        self.normalization = normalization
        self.output_type = output_type
        self.confidence_threshold = confidence_threshold
        self.model: Any | None = None
        self.input_width = fallback_width
        self.input_height = fallback_height
        self.channels = 3
        self.tensorflow_version = "unknown"
        self.input_shape: Any = None
        self.output_shape: Any = None
        # Metadata chỉ có khi checkpoint weights đi kèm experiment_config.json.
        self.metadata: dict[str, Any] = {}

    def initialize(self) -> None:
        """Load the model once and validate its input, output and class mapping."""
        if self.model is not None:
            return
        if not self.model_path.is_file():
            raise ModelConfigurationError(f"Model file does not exist: {self.model_path}")
        self._validate_class_mapping()

        try:
            import tensorflow as tf

            self.tensorflow_version = tf.__version__
            if self.model_path.name.endswith(".weights.h5"):
                # Checkpoint ELA chỉ chứa weights nên phải dựng kiến trúc trước.
                from .keras_ela_model import load_mobilenetv1_ela_weights

                model, self.metadata = load_mobilenetv1_ela_weights(
                    self.model_path,
                    CLASS_NAMES,
                )
            else:
                model = tf.keras.models.load_model(self.model_path)
        except Exception as exc:
            raise ModelConfigurationError(f"Cannot load Keras model: {exc}") from exc

        self.input_shape = model.input_shape
        self.output_shape = model.output_shape
        self._validate_input_shape(self.input_shape)
        self._validate_output_shape(self.output_shape)
        self.model = model
        LOGGER.info("TensorFlow version: %s", self.tensorflow_version)
        LOGGER.info("Model path: %s", self.model_path)
        LOGGER.info("Model input shape: %s", self.input_shape)
        LOGGER.info("Model output shape: %s", self.output_shape)
        if self.metadata:
            LOGGER.info(
                "ELA checkpoint: best_epoch=%s val_macro_f1=%.2f%%",
                self.metadata.get("best_epoch"),
                float(self.metadata.get("best_val_macro_f1", 0.0)) * 100.0,
            )
        LOGGER.info(
            "Model input width=%s height=%s channels=%s classes=%s",
            self.input_width,
            self.input_height,
            self.channels,
            len(CLASS_NAMES),
        )

    def _validate_class_mapping(self) -> None:
        try:
            configured = [line.strip() for line in self.classes_path.read_text().splitlines() if line.strip()]
        except OSError as exc:
            raise ModelConfigurationError(f"Cannot read class file: {self.classes_path}") from exc
        if configured != CLASS_NAMES:
            raise ModelConfigurationError(
                f"Configured classes {configured!r} do not exactly match CLASS_NAMES {CLASS_NAMES!r}"
            )

    def _validate_input_shape(self, shape: Any) -> None:
        if isinstance(shape, list):
            if len(shape) != 1:
                raise ModelConfigurationError("Only single-input image models are supported")
            shape = shape[0]
        if not isinstance(shape, (tuple, list)) or len(shape) != 4:
            raise ModelConfigurationError(f"Expected model input shape (None, H, W, C), got {shape}")
        height, width, channels = shape[1], shape[2], shape[3]
        if isinstance(height, int) and height > 0:
            self.input_height = height
        if isinstance(width, int) and width > 0:
            self.input_width = width
        if channels is not None and int(channels) != 3:
            raise ModelConfigurationError(f"Expected 3 input channels, got {channels}")
        self.channels = 3

    def _validate_output_shape(self, shape: Any) -> None:
        if isinstance(shape, list):
            if len(shape) != 1:
                raise ModelConfigurationError("Only single-output models are supported")
            shape = shape[0]
        try:
            output_size = int(shape[-1])
        except (TypeError, ValueError, IndexError) as exc:
            raise ModelConfigurationError(f"Cannot determine model output size from {shape}") from exc
        if output_size != len(CLASS_NAMES):
            LOGGER.error("Model output classes do not match configured CLASS_NAMES")
            raise ModelConfigurationError(
                f"Model output size is {output_size}, "
                f"but {len(CLASS_NAMES)} class names were configured"
            )

    @staticmethod
    def _softmax(values: np.ndarray) -> np.ndarray:
        shifted = values - np.max(values)
        exponentials = np.exp(shifted)
        return exponentials / np.sum(exponentials)

    def _probabilities(self, output: np.ndarray) -> np.ndarray:
        values = np.asarray(output, dtype=np.float64).reshape(-1)
        if values.size != len(CLASS_NAMES):
            raise ModelInferenceError(
                f"Prediction returned {values.size} values; expected {len(CLASS_NAMES)}"
            )
        if not np.all(np.isfinite(values)):
            raise ModelInferenceError("Prediction contains NaN or infinite values")

        output_type = self.output_type
        if output_type == "auto":
            looks_like_probabilities = (
                np.all(values >= 0.0)
                and np.all(values <= 1.0)
                and np.isclose(np.sum(values), 1.0, atol=1e-3)
            )
            output_type = "softmax" if looks_like_probabilities else "logits"

        if output_type == "logits":
            probabilities = self._softmax(values)
        elif output_type == "sigmoid":
            probabilities = 1.0 / (1.0 + np.exp(-np.clip(values, -80, 80)))
        elif output_type == "softmax":
            probabilities = values
        else:
            raise ModelInferenceError(f"Unsupported model output type: {output_type}")

        probabilities = np.clip(probabilities, 0.0, 1.0)
        return probabilities

    def predict(self, image_path: str) -> ClassificationResult:
        """Run one inference and return class, confidence and raw probabilities."""
        if self.model is None:
            raise ModelInferenceError("Classifier is not initialized")
        tensor = preprocess_image(
            image_path,
            self.input_width,
            self.input_height,
            self.normalization,
        )
        try:
            started = time.perf_counter()
            raw_output = self.model.predict(tensor, verbose=0)
            inference_time_ms = (time.perf_counter() - started) * 1000.0
            probabilities = self._probabilities(raw_output)
        except ModelInferenceError:
            raise
        except Exception as exc:
            raise ModelInferenceError(f"Model prediction failed: {exc}") from exc

        class_index = int(np.argmax(probabilities))
        confidence = float(probabilities[class_index])
        class_name = CLASS_NAMES[class_index] if confidence >= self.confidence_threshold else "Unknown"
        result = ClassificationResult(
            class_index=class_index,
            class_name=class_name,
            confidence=confidence,
            inference_time_ms=inference_time_ms,
            probabilities=[float(value) for value in probabilities],
        )
        LOGGER.info(
            "Inference result index=%s class=%s confidence=%.6f probabilities=%s time_ms=%.2f",
            result.class_index,
            result.class_name,
            result.confidence,
            result.probabilities,
            result.inference_time_ms,
        )
        return result

    def close(self) -> None:
        """Release the model reference and clear the Keras session."""
        self.model = None
        try:
            import tensorflow as tf

            tf.keras.backend.clear_session()
        except Exception:
            LOGGER.debug("TensorFlow session cleanup unavailable", exc_info=True)
        LOGGER.info("Classifier closed")
