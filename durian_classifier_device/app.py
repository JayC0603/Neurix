#!/usr/bin/env python3
"""Durian classifier device application entry point."""

from __future__ import annotations

import logging
import signal
import sys
import threading
import time

from config import PROJECT_DIR, Settings
from hardware import AudioRecorder, CameraService, LCDDisplay, LimitSwitch, ServoController
from hardware.lcd_display import LCD_IDLE_LINE_1, LCD_IDLE_LINE_2
from inference import TorchAudioClassifier, create_image_classifier
from inference.exceptions import CameraError, ModelError
from inference.presence_detector import EmptySceneDetector
from services import CameraPreviewServer, CaptureClassifyService, LatestResultStore
from utils.logging_config import configure_logging
from utils.process_lock import AlreadyRunningError, ProcessLock


LOGGER = logging.getLogger(__name__)


def _mock_input_loop(limit_switch: LimitSwitch, stop_event: threading.Event) -> None:
    """Translate Enter presses into short mock switch pulses."""
    LOGGER.info("Mock switch ready: press Enter to trigger classification")
    while not stop_event.is_set():
        try:
            input()
        except (EOFError, KeyboardInterrupt):
            return
        if stop_event.is_set():
            return
        limit_switch.simulate_press()
        limit_switch.simulate_release()


def run() -> int:
    """Initialize services, wait for signals, and shut down in safe order."""
    settings = Settings.load()
    configure_logging(PROJECT_DIR / "logs", settings.log_level)
    process_lock = ProcessLock(PROJECT_DIR / "logs" / "app.lock")
    try:
        process_lock.acquire()
    except AlreadyRunningError as exc:
        LOGGER.error("%s", exc)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 7

    LOGGER.info("Durian classifier device starting; environment=%s", settings.app_env)
    LOGGER.info(
        "Hardware config LCD bus=%s address=0x%02X GPIO=%s camera=%s mock=%s",
        settings.lcd_i2c_bus,
        settings.lcd_i2c_address,
        settings.limit_switch_gpio,
        settings.camera_device_path or settings.camera_device or settings.camera_device_index,
        settings.hardware_mock,
    )
    LOGGER.info("Model path: %s", settings.model_path)

    stop_event = threading.Event()
    lcd = LCDDisplay(
        bus=settings.lcd_i2c_bus,
        address=settings.lcd_i2c_address,
        columns=settings.lcd_columns,
        rows=settings.lcd_rows,
        enabled=settings.lcd_enabled,
        mock=settings.hardware_mock,
    )
    camera = CameraService(
        device_index=(
            settings.camera_device_path
            or settings.camera_device
            or settings.camera_device_index
        ),
        width=settings.camera_width,
        height=settings.camera_height,
        warmup_frames=settings.camera_warmup_frames,
        retry_count=settings.camera_retry_count,
        output_dir=settings.capture_output_dir,
        capture_jpeg_quality=settings.camera_capture_jpeg_quality,
        focus_sample_frames=settings.camera_focus_sample_frames,
        blur_threshold=settings.camera_blur_threshold,
        mock=settings.hardware_mock,
        mock_image_path=settings.mock_image_path,
        fourcc=settings.camera_fourcc,
        fps=settings.camera_fps,
        # Các control hình ảnh mặc định là None: không ghi đè màu gốc.
        brightness=settings.camera_brightness,
        contrast=settings.camera_contrast,
        saturation=settings.camera_saturation,
        gamma=settings.camera_gamma,
        sharpness=settings.camera_sharpness,
        backlight_compensation=settings.camera_backlight_compensation,
        power_line_frequency=settings.camera_power_line_frequency,
        auto_white_balance=settings.camera_auto_white_balance,
        white_balance_temperature=settings.camera_white_balance_temperature,
        auto_focus=settings.camera_auto_focus,
        red_gain=settings.camera_red_gain,
        green_gain=settings.camera_green_gain,
        blue_gain=settings.camera_blue_gain,
        color_saturation_scale=settings.camera_color_saturation_scale,
        roi_path=settings.camera_roi_path,
        histogram_equalization_enabled=(
            settings.camera_histogram_equalization_enabled
        ),
        clahe_clip_limit=settings.camera_clahe_clip_limit,
        clahe_grid_size=settings.camera_clahe_grid_size,
        camera_device=settings.camera_device or "/dev/video0",
        warmup_seconds=settings.camera_warmup_seconds,
        lock_white_balance=settings.camera_lock_white_balance,
        lock_exposure=settings.camera_lock_exposure,
    )
    # Tự chọn worker PyTorch cho .pth hoặc giữ nguyên Keras cho .keras.
    classifier = create_image_classifier(settings)
    # Model âm thanh chạy độc lập, không thay đổi model ảnh hiện tại.
    audio_classifier = TorchAudioClassifier(
        model_path=settings.audio_model_path,
        python_executable=settings.audio_model_python,
        confidence_threshold=settings.audio_model_confidence_threshold,
    )
    limit_switch = LimitSwitch(
        gpio_pin=settings.limit_switch_gpio,
        bounce_time=settings.limit_switch_bounce_time,
        active_low=settings.limit_switch_active_low,
        mock=settings.hardware_mock,
    )
    servos = ServoController(
        mock=settings.hardware_mock,
        move_to_strike_time=settings.servo_move_to_strike_time,
        strike_hold_time=settings.servo_strike_hold_time,
        return_home_time=settings.servo_return_home_time,
        home_settle_time=settings.servo_home_settle_time,
        # Hệ thống chỉ sở hữu Servo 1 trên PWM0/BCM18.
        enabled_servos=(1,),
    )
    # USB microphone ghi vào record.wav ở thư mục gốc của project.
    audio_recorder = AudioRecorder(
        output_path=PROJECT_DIR / "record.wav",
        duration_seconds=settings.audio_record_seconds,
        mock=settings.hardware_mock,
    )
    presence_detector = (
        EmptySceneDetector(
            reference_path=settings.empty_scene_reference_path,
            difference_threshold=settings.durian_presence_difference_threshold,
        )
        if settings.durian_presence_enabled and not settings.hardware_mock
        else None
    )
    result_store = LatestResultStore()
    service = CaptureClassifyService(
        camera=camera,
        classifier=classifier,
        lcd=lcd,
        result_display_seconds=settings.result_display_seconds,
        error_display_seconds=settings.error_display_seconds,
        delete_image_after_inference=settings.delete_image_after_inference,
        servo_controller=servos,
        audio_recorder=audio_recorder,
        audio_classifier=audio_classifier,
        images_per_job=settings.capture_images_per_job,
        image_interval_seconds=settings.capture_image_interval_seconds,
        fusion_image_weight=settings.fusion_image_weight,
        fusion_audio_weight=settings.fusion_audio_weight,
        presence_detector=presence_detector,
        result_store=result_store,
    )
    preview = CameraPreviewServer(
        camera=camera,
        host=settings.camera_preview_host,
        port=settings.camera_preview_port,
        fps=settings.camera_preview_fps,
        jpeg_quality=settings.camera_preview_jpeg_quality,
        enabled=settings.camera_preview_enabled,
        result_store=result_store,
    )

    def request_shutdown(signum: int, _frame: object) -> None:
        LOGGER.info("Shutdown signal received: %s", signum)
        stop_event.set()

    signal.signal(signal.SIGINT, request_shutdown)
    signal.signal(signal.SIGTERM, request_shutdown)

    exit_code = 0
    try:
        lcd.initialize()
        # Màn hình chào: tên trường ở dòng 1, tên đội ở dòng 2.
        lcd.display_message(LCD_IDLE_LINE_1, LCD_IDLE_LINE_2)

        try:
            servos.init_servos()
        except Exception:
            lcd.display_message("Error", "Servo failed")
            LOGGER.exception("Servo initialization failed")
            return 5

        try:
            camera.initialize()
        except CameraError:
            lcd.display_message("Error", "Camera failed")
            LOGGER.exception("Camera initialization failed")
            return 2
        preview.start()

        try:
            classifier.initialize()
        except ModelError:
            lcd.display_message("Error", "Model failed")
            LOGGER.exception("Model initialization failed")
            return 3

        try:
            audio_classifier.initialize()
        except ModelError:
            lcd.display_message("Error", "Audio model")
            LOGGER.exception("Audio model initialization failed")
            return 6

        def handle_switch_pressed() -> None:
            # Worker chụp ảnh trước, mở micro rồi mới chạy servo để thu tiếng gõ.
            lcd.display_message("CHECKING", "PLEASE WAIT")
            service.on_switch_pressed()

        def handle_switch_released() -> None:
            # Tuyệt đối không phát lệnh servo từ callback GPIO.
            LOGGER.info("Switch released; no Servo 1 command issued")

        limit_switch.register_on_pressed(handle_switch_pressed)
        limit_switch.register_on_released(handle_switch_released)

        service.initialize()
        try:
            limit_switch.initialize()
        except Exception:
            lcd.display_message("Error", "System failed")
            LOGGER.exception("Limit switch is required but failed to initialize")
            return 4

        # Khi rảnh, LCD luôn hiển thị tên trường và tên đội.
        lcd.display_message(LCD_IDLE_LINE_1, LCD_IDLE_LINE_2)
        LOGGER.info(
            "System ready; preview address=%s",
            preview.display_host or "unavailable",
        )

        if settings.hardware_mock:
            threading.Thread(
                target=_mock_input_loop,
                args=(limit_switch, stop_event),
                name="mock-switch-input",
                daemon=True,
            ).start()

        while not stop_event.wait(0.5):
            pass
    except Exception:
        exit_code = 1
        lcd.display_message("Error", "System failed")
        LOGGER.exception("Fatal application error")
    finally:
        LOGGER.info("Application shutdown started")
        limit_switch.close()
        service.shutdown()
        servos.cleanup_servos()
        preview.stop()
        camera.close()
        classifier.close()
        audio_classifier.close()
        lcd.close()
        process_lock.release()
        LOGGER.info("Application shutdown complete")
    return exit_code


if __name__ == "__main__":
    sys.exit(run())
