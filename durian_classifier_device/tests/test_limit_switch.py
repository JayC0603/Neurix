#!/usr/bin/env python3
"""Standalone limit switch press, release and debounce test."""

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--seconds", type=float, default=30.0)
    args = parser.parse_args()
    settings = Settings.load()
    configure_logging(PROJECT_DIR / "logs", settings.log_level)
    logger = logging.getLogger(__name__)
    switch = LimitSwitch(
        settings.limit_switch_gpio,
        settings.limit_switch_bounce_time,
        settings.limit_switch_active_low,
        args.mock or settings.hardware_mock,
    )
    switch.register_on_pressed(lambda: logger.info("TEST callback: PRESSED"))
    switch.register_on_released(lambda: logger.info("TEST callback: RELEASED"))
    try:
        switch.initialize()
        if switch.mock:
            logger.info("Mock debounce test: rapid duplicate state transitions")
            switch.simulate_press()
            switch.simulate_press()
            switch.simulate_release()
            switch.simulate_release()
        else:
            logger.info("Press and release the switch for %.1f seconds", args.seconds)
            time.sleep(args.seconds)
    finally:
        switch.close()


if __name__ == "__main__":
    main()
