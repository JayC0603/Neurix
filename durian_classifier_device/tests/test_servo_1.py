#!/usr/bin/env python3
"""Safely exercise only physical servo 1 using the configured angles."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from config import Settings  # noqa: E402
from hardware.servo_controller import (  # noqa: E402
    HOME_ANGLE,
    STRIKE_ANGLE,
    ServoController,
)


def main() -> None:
    settings = Settings.load()
    if settings.hardware_mock:
        raise SystemExit("HARDWARE_MOCK must be false for this physical servo test")

    print(
        "WARNING: Servo 1 on BCM18 (physical pin 12) will move "
        f"HOME {HOME_ANGLE:.0f} -> STRIKE {STRIKE_ANGLE:.0f} degrees, 3 times in 3s."
    )
    print("Stop durian-classifier.service and keep hands clear of the mechanism.")
    answer = input("Type MOVE to continue: ").strip().upper()
    if answer != "MOVE":
        raise SystemExit("Servo test cancelled")

    servos = ServoController(
        mock=False,
        move_to_strike_time=settings.servo_move_to_strike_time,
        strike_hold_time=settings.servo_strike_hold_time,
        return_home_time=settings.servo_return_home_time,
        home_settle_time=settings.servo_home_settle_time,
        enabled_servos=(1,),
    )
    try:
        servos.init_servos()
        servos.tap_sequence()
        print("Physical servo 1 test completed")
    finally:
        servos.cleanup_servos()


if __name__ == "__main__":
    main()
