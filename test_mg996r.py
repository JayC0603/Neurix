#!/usr/bin/env python3
"""Gõ MG996R Servo 1 ở 150°, ba lần trong 2,5 giây."""

from __future__ import annotations

import logging
from time import sleep

from durian_classifier_device.config import Settings
from durian_classifier_device.hardware.servo_controller import ServoController


DEMO_COUNTDOWN_TIME = 3.0

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


def main() -> int:
    settings = Settings.load()
    controller = ServoController(
        mock=False,
        move_to_strike_time=settings.servo_move_to_strike_time,
        strike_hold_time=settings.servo_strike_hold_time,
        return_home_time=settings.servo_return_home_time,
        home_settle_time=settings.servo_home_settle_time,
        enabled_servos=(1,),
    )
    exit_code = 0
    try:
        controller.init_servos()
        logging.info("DEMO starts in %.1f seconds", DEMO_COUNTDOWN_TIME)
        sleep(DEMO_COUNTDOWN_TIME)
        controller.tap_sequence()
        logging.info("DEMO complete; HOME return/settle waits completed")
    except KeyboardInterrupt:
        logging.warning("Ctrl+C received; returning Servo 1 to HOME")
        exit_code = 130
    except Exception:
        logging.exception("Servo sequence failed")
        exit_code = 1
    finally:
        # cleanup chỉ command HOME nếu lệnh gần nhất chưa phải HOME.
        controller.cleanup_servos()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
