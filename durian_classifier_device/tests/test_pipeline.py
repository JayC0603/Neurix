#!/usr/bin/env python3
"""Interactive single-job camera -> model -> LCD pipeline test."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from config import Settings  # noqa: E402
from hardware import CameraService, LCDDisplay, LimitSwitch  # noqa: E402
from inference import create_image_classifier  # noqa: E402
from services import CaptureClassifyService  # noqa: E402
from utils.logging_config import configure_logging  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock-image", type=Path)
    args = parser.parse_args()
    settings = Settings.load()
    configure_logging(PROJECT_DIR / "logs", settings.log_level)
    logger = logging.getLogger(__name__)
    mock = settings.hardware_mock or args.mock_image is not None
    mock_image = args.mock_image or settings.mock_image_path

    lcd = LCDDisplay(
        settings.lcd_i2c_bus,
        settings.lcd_i2c_address,
        settings.lcd_columns,
        settings.lcd_rows,
        settings.lcd_enabled,
        mock,
    )
    camera = CameraService(
        settings.camera_device_index,
        settings.camera_width,
        settings.camera_height,
        settings.camera_warmup_frames,
        settings.camera_retry_count,
        settings.capture_output_dir,
        mock,
        mock_image,
    )
    classifier = create_image_classifier(settings)
    switch = LimitSwitch(settings.limit_switch_gpio, settings.limit_switch_bounce_time, mock)
    service = CaptureClassifyService(
        camera,
        classifier,
        lcd,
        settings.result_display_seconds,
        settings.error_display_seconds,
        settings.delete_image_after_inference,
    )
    try:
        lcd.initialize()
        camera.initialize()
        classifier.initialize()
        service.initialize()
        switch.initialize()
        switch.register_on_pressed(service.on_switch_pressed)
        lcd.display_message("ĐH FPT Can Tho", "Team: Neurix")
        if mock:
            input("Press Enter to process one mock image...")
            switch.simulate_press()
            switch.simulate_release()
            while service.is_busy():
                time.sleep(0.1)
        else:
            logger.info("Use the physical switch; Ctrl+C exits")
            while True:
                time.sleep(1.0)
    except KeyboardInterrupt:
        logger.info("Pipeline test interrupted")
    finally:
        switch.close()
        service.shutdown()
        camera.close()
        classifier.close()
        lcd.close()


if __name__ == "__main__":
    main()
