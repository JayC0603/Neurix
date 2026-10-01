#!/usr/bin/env python3
"""Test A/B Servo 1 trên BCM17, tách khỏi PWM0/BCM18 của ứng dụng chính."""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from config import Settings  # noqa: E402
from hardware.servo_controller import (  # noqa: E402
    HOME_ANGLE,
    MAX_PULSE_WIDTH,
    MIN_PULSE_WIDTH,
    STRIKE_ANGLE,
)


SERVO_1_TEST_GPIO = 17


def strike(servo: object, name: str, settings: Settings) -> None:
    """Gửi đúng một lần từng lệnh tuyệt đối 90 -> 150 -> 90."""
    logging.info("%s -> HOME %.0f", name, HOME_ANGLE)
    servo.angle = HOME_ANGLE
    time.sleep(settings.servo_return_home_time)
    time.sleep(settings.servo_home_settle_time)

    logging.info("%s -> STRIKE %.0f", name, STRIKE_ANGLE)
    servo.angle = STRIKE_ANGLE
    time.sleep(settings.servo_move_to_strike_time)
    time.sleep(settings.servo_strike_hold_time)

    logging.info("%s -> HOME %.0f", name, HOME_ANGLE)
    servo.angle = HOME_ANGLE
    time.sleep(settings.servo_return_home_time)
    time.sleep(settings.servo_home_settle_time)
    logging.info("%s HOME return/settle wait completed", name)


def main() -> int:
    settings = Settings.load()
    if settings.hardware_mock:
        raise SystemExit("HARDWARE_MOCK must be false for this physical GPIO test")

    print(
        "WARNING: move Servo 1 signal wire from physical pin 12/BCM18 "
        "to physical pin 11/BCM17 before continuing."
    )
    print("Stop durian-classifier.service and every other servo test first.")
    if input("Type GPIO17 to move Servo 1 once: ").strip().upper() != "GPIO17":
        raise SystemExit("GPIO17 test cancelled")

    from gpiozero import AngularServo
    from gpiozero.pins.lgpio import LGPIOFactory

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    factory = None
    servo = None
    home_wait_completed = False
    try:
        factory = LGPIOFactory()
        servo = AngularServo(
            SERVO_1_TEST_GPIO,
            min_angle=0,
            max_angle=180,
            min_pulse_width=MIN_PULSE_WIDTH,
            max_pulse_width=MAX_PULSE_WIDTH,
            initial_angle=None,
            pin_factory=factory,
        )
        strike(servo, "Servo 1 on BCM17", settings)
        home_wait_completed = True
        return 0
    except KeyboardInterrupt:
        logging.warning("Ctrl+C received; cleanup will command Servo 1 HOME")
        return 130
    except Exception:
        logging.exception("GPIO17 test failed; cleanup will attempt HOME")
        raise
    finally:
        if servo is not None:
            if not home_wait_completed:
                try:
                    logging.info("Servo 1 on BCM17 -> RECOVERY HOME %.0f", HOME_ANGLE)
                    servo.angle = HOME_ANGLE
                    time.sleep(settings.servo_return_home_time)
                    time.sleep(settings.servo_home_settle_time)
                    logging.info("GPIO17 recovery HOME wait completed")
                except Exception:
                    logging.exception("Could not return GPIO17 test servo HOME")
            try:
                servo.detach()
            finally:
                servo.close()
        if factory is not None:
            factory.close()
        logging.info("GPIO17 test cleanup completed")


if __name__ == "__main__":
    raise SystemExit(main())
