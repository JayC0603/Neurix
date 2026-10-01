#!/usr/bin/env python3
"""Kiểm tra model audio PyTorch bằng record.wav hiện có."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from config import Settings  # noqa: E402
from inference.audio_classifier import TorchAudioClassifier  # noqa: E402


def main() -> None:
    """Nạp model thật và xác nhận đầu ra đủ ba xác suất hợp lệ."""
    settings = Settings.load()
    record_path = PROJECT_DIR / "record.wav"
    classifier = TorchAudioClassifier(
        model_path=settings.audio_model_path,
        python_executable=settings.audio_model_python,
        confidence_threshold=settings.audio_model_confidence_threshold,
    )
    try:
        classifier.initialize()
        result = classifier.predict(record_path)
        assert len(result.probabilities) == 3
        assert abs(sum(result.probabilities) - 1.0) < 1e-5
        print(
            "Audio classifier test: PASS | "
            f"class={result.class_name} confidence={result.confidence:.4f}"
        )
    finally:
        classifier.close()


if __name__ == "__main__":
    main()
