#!/usr/bin/env python3
"""Standalone LCD state sequence test."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from config import Settings  # noqa: E402
from hardware.lcd_display import LCDDisplay  # noqa: E402
from utils.logging_config import configure_logging  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--delay", type=float, default=1.5)
    args = parser.parse_args()
    settings = Settings.load()
    configure_logging(PROJECT_DIR / "logs", settings.log_level)
    lcd = LCDDisplay(
        settings.lcd_i2c_bus,
        settings.lcd_i2c_address,
        settings.lcd_columns,
        settings.lcd_rows,
        settings.lcd_enabled,
        args.mock or settings.hardware_mock,
    )
    try:
        lcd.initialize()
        states = [
            ("ĐH FPT Can Tho", "Team: Neurix"),
            ("Capturing...", "Please wait"),
            ("Analyzing...", "Please wait"),
        ]
        for line1, line2 in states:
            lcd.display_message(line1, line2)
            time.sleep(args.delay)
        lcd.display_result("ripe", 0.945)
        time.sleep(args.delay)
        lcd.display_message("Error", "System failed")
        time.sleep(args.delay)
    finally:
        lcd.close()
        logging.getLogger(__name__).info("LCD test complete")


if __name__ == "__main__":
    main()
