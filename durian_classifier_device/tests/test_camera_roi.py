#!/usr/bin/env python3
"""Kiểm tra lưu, nạp và crop ROI mà không truy cập camera thật."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from hardware.camera_service import CameraService  # noqa: E402


def make_camera(roi_path: Path) -> CameraService:
    """Tạo service tối giản chỉ để kiểm tra phép cắt ROI."""
    return CameraService(
        device_index=0,
        width=1280,
        height=720,
        warmup_frames=0,
        retry_count=0,
        output_dir=roi_path.parent,
        roi_path=roi_path,
    )


def main() -> None:
    """Xác nhận ROI lưu bền vững và đổi đúng sang tọa độ pixel."""
    with tempfile.TemporaryDirectory() as temporary_dir:
        roi_path = Path(temporary_dir) / "camera_roi.json"
        camera = make_camera(roi_path)
        selected = camera.set_roi(
            {"x": 0.25, "y": 0.2, "width": 0.5, "height": 0.6}
        )
        assert selected == {"x": 0.25, "y": 0.2, "width": 0.5, "height": 0.6}

        # Frame 1280x720 phải được cắt thành 640x432 theo ROI trên.
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        cropped = camera._crop_to_roi(frame)
        assert cropped.shape == (432, 640, 3)

        # Tạo service mới để chứng minh ROI vẫn còn sau khi khởi động lại.
        reloaded_camera = make_camera(roi_path)
        assert reloaded_camera.get_roi() == selected
    print("Camera ROI test: PASS")


if __name__ == "__main__":
    main()
