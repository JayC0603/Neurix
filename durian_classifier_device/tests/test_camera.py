#!/usr/bin/env python3
"""Standalone camera initialization and capture test."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from config import Settings  # noqa: E402
from hardware.camera_service import CameraService  # noqa: E402
from utils.logging_config import configure_logging  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock-image", type=Path)
    args = parser.parse_args()
    settings = Settings.load()
    configure_logging(PROJECT_DIR / "logs", settings.log_level)
    mock_image = args.mock_image or settings.mock_image_path
    camera = CameraService(
        device_index=(
            settings.camera_device_path
            or settings.camera_device
            or settings.camera_device_index
        ),
        width=settings.camera_width,
        height=settings.camera_height,
        warmup_frames=settings.camera_warmup_frames,
        retry_count=settings.camera_retry_count,
        output_dir=settings.capture_output_dir,
        capture_jpeg_quality=settings.camera_capture_jpeg_quality,
        mock=mock_image is not None or settings.hardware_mock,
        mock_image_path=mock_image,
        fourcc=settings.camera_fourcc,
        fps=settings.camera_fps,
        # Dùng cùng cấu hình camera với app chính để kết quả test nhất quán.
        brightness=settings.camera_brightness,
        contrast=settings.camera_contrast,
        saturation=settings.camera_saturation,
        gamma=settings.camera_gamma,
        sharpness=settings.camera_sharpness,
        backlight_compensation=settings.camera_backlight_compensation,
        power_line_frequency=settings.camera_power_line_frequency,
        auto_white_balance=settings.camera_auto_white_balance,
        white_balance_temperature=settings.camera_white_balance_temperature,
        auto_focus=settings.camera_auto_focus,
        red_gain=settings.camera_red_gain,
        green_gain=settings.camera_green_gain,
        blue_gain=settings.camera_blue_gain,
        color_saturation_scale=settings.camera_color_saturation_scale,
        roi_path=settings.camera_roi_path,
        histogram_equalization_enabled=(
            settings.camera_histogram_equalization_enabled
        ),
        clahe_clip_limit=settings.camera_clahe_clip_limit,
        clahe_grid_size=settings.camera_clahe_grid_size,
        camera_device=settings.camera_device or "/dev/video0",
        warmup_seconds=settings.camera_warmup_seconds,
        lock_white_balance=settings.camera_lock_white_balance,
        lock_exposure=settings.camera_lock_exposure,
    )
    try:
        camera.initialize()
        # Đọc đúng đường preview để tạo cặp ảnh debug raw/preview cần so sánh.
        preview_jpeg = camera.read_jpeg(settings.camera_preview_jpeg_quality)
        if not preview_jpeg:
            raise RuntimeError("Camera preview did not return JPEG data")
        image_path = Path(camera.capture_image())
        if not image_path.is_file():
            raise RuntimeError(f"Capture file was not created: {image_path}")
        for debug_name in ("debug_raw.jpg", "debug_preview.jpg"):
            debug_path = settings.capture_output_dir.parent / debug_name
            if not debug_path.is_file():
                raise RuntimeError(f"Debug image was not created: {debug_path}")
        logging.getLogger(__name__).info("Camera test image: %s", image_path)
    finally:
        camera.close()


if __name__ == "__main__":
    main()
