"""Raspberry Pi hardware services with lazy imports.

Không import camera/audio khi một script chỉ cần servo. Điều này giữ các test
phần cứng độc lập và tránh yêu cầu toàn bộ inference stack chỉ để điều khiển PWM.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


__all__ = ["AudioRecorder", "CameraService", "LCDDisplay", "LimitSwitch", "ServoController"]

_EXPORTS = {
    "AudioRecorder": (".audio_recorder", "AudioRecorder"),
    "CameraService": (".camera_service", "CameraService"),
    "LCDDisplay": (".lcd_display", "LCDDisplay"),
    "LimitSwitch": (".limit_switch", "LimitSwitch"),
    "ServoController": (".servo_controller", "ServoController"),
}


def __getattr__(name: str) -> Any:
    """Chỉ nạp module tương ứng khi thuộc tính thực sự được sử dụng."""
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value
