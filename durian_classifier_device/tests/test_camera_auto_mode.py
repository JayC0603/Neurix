#!/usr/bin/env python3
"""Kiểm tra camera luôn giữ auto WB/exposure và không đặt giá trị thủ công."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from hardware.camera_service import CameraService  # noqa: E402


CONTROL_MENUS = """
                     brightness 0x00980900 (int)    : min=0 max=255 step=1 default=128 value=128
                       contrast 0x00980901 (int)    : min=0 max=255 step=1 default=128 value=128
                     saturation 0x00980902 (int)    : min=0 max=255 step=1 default=128 value=128
                          gamma 0x00980910 (int)    : min=72 max=500 step=1 default=100 value=100
                      sharpness 0x0098091b (int)    : min=0 max=255 step=1 default=128 value=128
         backlight_compensation 0x0098091c (int)    : min=0 max=1 step=1 default=1 value=1
           power_line_frequency 0x00980918 (menu)   : min=0 max=2 default=2 value=2 (60 Hz)
                                0: Disabled
                                1: 50 Hz
                                2: 60 Hz
        white_balance_automatic 0x0098090c (bool)   : default=1 value=1
   focus_automatic_continuous 0x009a090c (bool)   : default=1 value=1
      white_balance_temperature 0x0098091a (int)    : min=2800 max=7500 step=1 default=4000 value=3300 flags=inactive
                  auto_exposure 0x009a0901 (menu)   : min=0 max=3 default=3 value=3 (Aperture Priority Mode)
                                1: Manual Mode
                                3: Aperture Priority Mode
         exposure_time_absolute 0x009a0902 (int)    : min=5 max=2500 step=1 default=156 value=333 flags=inactive
"""


class FakeCapture:
    """Camera giả chỉ đếm các frame bị bỏ trong warm-up."""

    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.read_count = 0

    def read(self):
        self.read_count += 1
        self.events.append("frame")
        return True, np.zeros((8, 8, 3), dtype=np.uint8)


def main() -> None:
    """Xác nhận cấu hình auto trước warm-up và không hề chuyển sang manual."""
    events: list[str] = []
    camera = CameraService(
        device_index=0,
        width=1280,
        height=720,
        warmup_frames=0,
        retry_count=0,
        output_dir=Path("/tmp/captures"),
        brightness=124,
        contrast=128,
        saturation=110,
        gamma=100,
        sharpness=128,
        backlight_compensation=0,
        power_line_frequency=1,
        warmup_seconds=0.01,
    )

    def fake_run(arguments: list[str], device: str | None = None):
        command = arguments[0]
        events.append(command)
        if command == "--list-ctrls-menus":
            stdout = CONTROL_MENUS
        elif command == "--all":
            stdout = "Driver Info:\n\tDriver name: uvcvideo\n"
        else:
            stdout = ""
        return subprocess.CompletedProcess(arguments, 0, stdout, "")

    camera._run_v4l2 = fake_run  # type: ignore[method-assign]
    capture = FakeCapture(events)
    camera.configure_automatic_controls(capture, 0)

    assert capture.read_count > 0
    assert events[0] == "--list-ctrls-menus"
    assert events.count("--all") == 2
    first_frame_index = events.index("frame")

    # Các giá trị màu cơ bản và hai control auto phải được đặt trước warm-up.
    expected_before_warmup = {
        "--set-ctrl=brightness=124",
        "--set-ctrl=contrast=128",
        "--set-ctrl=saturation=110",
        "--set-ctrl=gamma=100",
        "--set-ctrl=sharpness=128",
        "--set-ctrl=backlight_compensation=0",
        "--set-ctrl=power_line_frequency=1",
        "--set-ctrl=white_balance_automatic=1",
        "--set-ctrl=auto_exposure=3",
        "--set-ctrl=focus_automatic_continuous=1",
    }
    for command in expected_before_warmup:
        assert command in events
        assert events.index(command) < first_frame_index

    # Giá trị 3 phải được lấy từ nhãn menu Aperture Priority của driver.
    assert camera._camera_controls["auto_exposure"]["menus"] == {
        1: "Manual Mode",
        3: "Aperture Priority Mode",
    }

    # Phiên auto-only tuyệt đối không đọc/đặt hai control thủ công hoặc tắt auto.
    forbidden_fragments = (
        "--get-ctrl=white_balance_temperature",
        "--get-ctrl=exposure_time_absolute",
        "--set-ctrl=white_balance_temperature=",
        "--set-ctrl=exposure_time_absolute=",
        "--set-ctrl=white_balance_automatic=0",
        "--set-ctrl=auto_exposure=1",
    )
    assert not any(
        command.startswith(fragment)
        for command in events
        for fragment in forbidden_fragments
    )

    # Khi không cấu hình chỉnh màu, phải xóa giá trị cũ còn lưu
    # trong driver bằng chính giá trị default mà driver công bố.
    default_events: list[str] = []
    default_camera = CameraService(
        device_index=0,
        width=1280,
        height=720,
        warmup_frames=0,
        retry_count=0,
        output_dir=Path("/tmp/captures"),
    )

    def fake_default_run(arguments: list[str], device: str | None = None):
        command = arguments[0]
        default_events.append(command)
        stdout = CONTROL_MENUS if command == "--list-ctrls-menus" else ""
        return subprocess.CompletedProcess(arguments, 0, stdout, "")

    default_camera._run_v4l2 = fake_default_run  # type: ignore[method-assign]
    default_camera.list_camera_controls("/dev/video0")
    default_camera._apply_initial_v4l2_controls()
    expected_driver_defaults = {
        "--set-ctrl=brightness=128",
        "--set-ctrl=contrast=128",
        "--set-ctrl=saturation=128",
        "--set-ctrl=gamma=100",
        "--set-ctrl=sharpness=128",
        "--set-ctrl=backlight_compensation=1",
    }
    assert expected_driver_defaults.issubset(set(default_events))
    print("Camera auto-only test: PASS")


if __name__ == "__main__":
    main()
