"""Image loading and configurable model preprocessing."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .exceptions import ModelInferenceError


def preprocess_image(
    image_path: str | Path,
    width: int,
    height: int,
    normalization: str,
) -> np.ndarray:
    """Read a BGR image, convert to RGB, resize and normalize it."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise ModelInferenceError(f"Cannot read image: {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    tensor = image.astype(np.float32)

    if normalization == "zero_one":
        tensor /= 255.0
    elif normalization == "minus_one_one":
        tensor = tensor / 127.5 - 1.0
    elif normalization != "none":
        raise ModelInferenceError(f"Unsupported normalization: {normalization}")

    return np.expand_dims(tensor, axis=0)
