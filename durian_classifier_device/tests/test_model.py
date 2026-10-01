#!/usr/bin/env python3
"""Load the real Keras model and classify one provided image."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from config import Settings  # noqa: E402
from inference import CLASS_NAMES, create_image_classifier  # noqa: E402
from utils.logging_config import configure_logging  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    args = parser.parse_args()
    settings = Settings.load()
    configure_logging(PROJECT_DIR / "logs", settings.log_level)
    logger = logging.getLogger(__name__)
    classifier = create_image_classifier(settings)
    try:
        classifier.initialize()
        result = classifier.predict(args.image)
        if hasattr(classifier, "tensorflow_version"):
            logger.info("TensorFlow version: %s", classifier.tensorflow_version)
            logger.info("Model input shape: %s", classifier.input_shape)
            logger.info("Model output shape: %s", classifier.output_shape)
        else:
            logger.info("PyTorch version: %s", classifier.framework_version)
            logger.info("Model metadata: %s", classifier.metadata)
            logger.info(
                "Primary validation metric Macro-F1: %.2f%%",
                float(classifier.metadata.get("val_macro_f1", 0.0)) * 100.0,
            )
        logger.info("Class order: %s", CLASS_NAMES)
        logger.info("Class index: %s", result.class_index)
        logger.info("Class name: %s", result.class_name)
        logger.info(
            "Prediction confidence (not accuracy): %.6f",
            result.confidence,
        )
        logger.info("Probabilities: %s", result.probabilities)
        logger.info("Inference time: %.2f ms", result.inference_time_ms)
    finally:
        classifier.close()


if __name__ == "__main__":
    main()
