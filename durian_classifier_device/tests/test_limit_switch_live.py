#!/usr/bin/env python3
"""Live GPIO limit switch diagnostic without camera, model or servo."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from config import Settings  # noqa: E402
from hardware.limit_switch import LimitSwitch  # noqa: E402
from utils.logging_config import configure_logging  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=60.0)
    args = parser.parse_args()

    settings = Settings.load()
    configure_logging(PROJECT_DIR / "logs", settings.log_level)
    logger = logging.getLogger(__name__)

    switch = LimitSwitch(
        gpio_pin=settings.limit_switch_gpio,
        bounce_time=settings.limit_switch_bounce_time,
        active_low=settings.limit_switch_active_low,
        mock=False,
    )
    switch.register_on_pressed(lambda: logger.info("EVENT: PRESSED"))
    switch.register_on_released(lambda: logger.info("EVENT: RELEASED"))

    try:
        switch.initialize()
        last_state = switch.is_pressed()
        logger.info(
            "Reading BCM GPIO%s for %.1f seconds. active_low=%s. Current state: %s",
            settings.limit_switch_gpio,
            args.seconds,
            settings.limit_switch_active_low,
            "PRESSED" if last_state else "RELEASED",
        )
        if settings.limit_switch_active_low:
            logger.info("Expected wiring: COM -> GND, NO -> BCM GPIO%s", settings.limit_switch_gpio)
        else:
            logger.info("Expected wiring: COM -> 3.3V, NO -> BCM GPIO%s", settings.limit_switch_gpio)

        end_time = time.monotonic() + args.seconds
        while time.monotonic() < end_time:
            state = switch.is_pressed()
            if state != last_state:
                logger.info("POLL: %s", "PRESSED" if state else "RELEASED")
                last_state = state
            time.sleep(0.05)
        return 0
    finally:
        switch.close()


if __name__ == "__main__":
    raise SystemExit(main())
