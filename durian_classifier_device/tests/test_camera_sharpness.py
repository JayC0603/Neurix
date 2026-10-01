#!/usr/bin/env python3
"""Verify burst capture selects sharp frames and rejects a blurry burst."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from hardware.camera_service import CameraService  # noqa: E402
from inference.exceptions import CameraBlurError  # noqa: E402


def make_camera(output_dir: Path, threshold: float) -> CameraService:
    return CameraService(
        device_index=0,
        width=200,
        height=100,
        warmup_frames=0,
        retry_count=0,
        output_dir=output_dir,
        focus_sample_frames=3,
        blur_threshold=threshold,
    )


def main() -> None:
    with tempfile.TemporaryDirectory() as temporary_dir:
        output_dir = Path(temporary_dir)
        sharp = np.zeros((100, 200, 3), dtype=np.uint8)
        sharp[:, ::4] = 255
        blurry = cv2.GaussianBlur(sharp, (31, 31), 0)

        camera = make_camera(output_dir, threshold=1.0)
        frames = iter([blurry, sharp, blurry])
        camera._read_frame_locked = lambda *_args, **_kwargs: next(frames)
        selected = camera._capture_sharpest_roi_frame()
        assert camera._sharpness_score(selected) == camera._sharpness_score(sharp)

        rejecting_camera = make_camera(output_dir, threshold=1_000_000.0)
        frames = iter([blurry, blurry, blurry])
        rejecting_camera._read_frame_locked = lambda *_args, **_kwargs: next(frames)
        try:
            rejecting_camera._capture_sharpest_roi_frame()
        except CameraBlurError:
            pass
        else:
            raise AssertionError("A fully blurry burst should be rejected")
    print("Camera sharpness test: PASS")


if __name__ == "__main__":
    main()
