"""Chọn image classifier theo định dạng file model."""

from __future__ import annotations

from typing import Any

from .classifier import KerasImageClassifier
from .onnx_image_classifier import OnnxImageClassifier
from .torch_image_classifier import TorchImageClassifier


def create_image_classifier(settings: Any):
    """Create the correct classifier for ONNX, PyTorch or Keras models."""

    suffix = settings.model_path.suffix.lower()
    if suffix == ".onnx":
        return OnnxImageClassifier(
            model_path=settings.model_path,
            python_executable=settings.image_model_python,
            confidence_threshold=settings.model_confidence_threshold,
        )
    if suffix in {".pt", ".pth"}:
        return TorchImageClassifier(
            model_path=settings.model_path,
            python_executable=settings.image_model_python,
            confidence_threshold=settings.model_confidence_threshold,
        )
    return KerasImageClassifier(
        model_path=settings.model_path,
        classes_path=settings.model_classes_path,
        fallback_width=settings.model_input_width,
        fallback_height=settings.model_input_height,
        normalization=settings.model_normalization,
        output_type=settings.model_output_type,
        confidence_threshold=settings.model_confidence_threshold,
    )
