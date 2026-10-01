"""Detect whether the fixed camera scene differs from the calibrated empty tray."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from .exceptions import ModelConfigurationError, ModelInferenceError


LOGGER = logging.getLogger(__name__)


class EmptySceneDetector:
    """Use a fixed empty-scene reference as an open-set gate for the classifier."""

    def __init__(self, reference_path: Path, difference_threshold: float) -> None:
        self.reference_path = Path(reference_path)
        self.difference_threshold = difference_threshold
        reference = cv2.imread(str(self.reference_path), cv2.IMREAD_COLOR)
        if reference is None:
            raise ModelConfigurationError(
                f"Empty-scene reference cannot be read: {self.reference_path}"
            )
        self._reference = self._scene_signature(reference)
        LOGGER.info(
            "Durian presence gate ready: reference=%s threshold=%.2f",
            self.reference_path,
            self.difference_threshold,
        )

    @staticmethod
    def _scene_signature(image: np.ndarray) -> np.ndarray:
        # A small LAB image preserves the tray layout while suppressing camera
        # noise. Centering each channel makes the comparison tolerant of a
        # global exposure/white-balance shift.
        reduced = cv2.resize(image, (8, 8), interpolation=cv2.INTER_AREA)
        lab = cv2.cvtColor(reduced, cv2.COLOR_BGR2LAB).astype(np.float32)
        return lab - lab.mean(axis=(0, 1), keepdims=True)

    def detect(self, image_path: str | Path) -> tuple[bool, float]:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ModelInferenceError(f"Presence-check image cannot be read: {image_path}")
        score = float(np.abs(self._scene_signature(image) - self._reference).mean())
        present = score >= self.difference_threshold
        LOGGER.info(
            "Durian presence check: present=%s difference=%.2f threshold=%.2f image=%s",
            present,
            score,
            self.difference_threshold,
            image_path,
        )
        return present, score
