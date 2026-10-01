"""Environment-backed application configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent

# Đổi thành 0x3F nếu module PCF8574 của LCD dùng địa chỉ đó.
LCD_I2C_ADDRESS = 0x27


def _load_environment(path: Path) -> None:
    """Load .env with python-dotenv, or a safe basic parser during bootstrap."""
    try:
        from dotenv import load_dotenv

        load_dotenv(path)
        return
    except ImportError:
        pass
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def _as_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _as_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)), 0)


def _as_optional_float(name: str, default: float | None) -> float | None:
    """Cho phép để trống control camera để giữ nguyên mặc định của driver."""
    fallback = "" if default is None else str(default)
    value = os.getenv(name, fallback).strip()
    return None if not value else float(value)


@dataclass(frozen=True)
class Settings:
    """Validated runtime settings loaded from environment variables."""

    app_env: str
    log_level: str
    lcd_enabled: bool
    lcd_i2c_bus: int
    lcd_i2c_address: int
    lcd_columns: int
    lcd_rows: int
    limit_switch_gpio: int
    limit_switch_bounce_time: float
    limit_switch_active_low: bool
    servo_move_to_strike_time: float
    servo_strike_hold_time: float
    servo_return_home_time: float
    servo_home_settle_time: float
    camera_device_index: int
    camera_device_path: str
    camera_device: str
    camera_width: int
    camera_height: int
    camera_fourcc: str
    camera_fps: float
    camera_brightness: float | None
    camera_contrast: float | None
    camera_saturation: float | None
    camera_gamma: float | None
    camera_sharpness: float | None
    camera_backlight_compensation: float | None
    camera_power_line_frequency: float | None
    camera_auto_white_balance: bool
    camera_white_balance_temperature: float | None
    camera_auto_focus: bool
    camera_red_gain: float
    camera_green_gain: float
    camera_blue_gain: float
    camera_color_saturation_scale: float
    camera_histogram_equalization_enabled: bool
    camera_clahe_clip_limit: float
    camera_clahe_grid_size: int
    camera_warmup_seconds: float
    camera_lock_white_balance: bool
    camera_lock_exposure: bool
    camera_warmup_frames: int
    camera_retry_count: int
    camera_preview_enabled: bool
    camera_preview_host: str
    camera_preview_port: int
    camera_preview_fps: float
    camera_preview_jpeg_quality: int
    camera_capture_jpeg_quality: int
    camera_focus_sample_frames: int
    camera_blur_threshold: float
    camera_roi_path: Path
    capture_output_dir: Path
    model_path: Path
    image_model_python: Path
    model_classes_path: Path
    model_input_width: int
    model_input_height: int
    model_normalization: str
    model_output_type: str
    model_confidence_threshold: float
    result_display_seconds: float
    error_display_seconds: float
    delete_image_after_inference: bool
    audio_model_path: Path
    audio_model_python: Path
    audio_model_confidence_threshold: float
    audio_record_seconds: float
    capture_images_per_job: int
    capture_image_interval_seconds: float
    fusion_image_weight: float
    fusion_audio_weight: float
    durian_presence_enabled: bool
    empty_scene_reference_path: Path
    durian_presence_difference_threshold: float
    hardware_mock: bool
    mock_image_path: Path | None

    @classmethod
    def load(cls) -> "Settings":
        """Load .env and return a configuration snapshot."""
        _load_environment(PROJECT_DIR / ".env")
        mock_path = os.getenv("MOCK_IMAGE_PATH", "").strip()
        settings = cls(
            app_env=os.getenv("APP_ENV", "production"),
            log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
            lcd_enabled=_as_bool("LCD_ENABLED", True),
            lcd_i2c_bus=_as_int("LCD_I2C_BUS", 1),
            lcd_i2c_address=_as_int("LCD_I2C_ADDRESS", LCD_I2C_ADDRESS),
            lcd_columns=_as_int("LCD_COLUMNS", 16),
            lcd_rows=_as_int("LCD_ROWS", 2),
            limit_switch_gpio=_as_int("LIMIT_SWITCH_GPIO", 5),
            limit_switch_bounce_time=float(os.getenv("LIMIT_SWITCH_BOUNCE_TIME", "0.05")),
            limit_switch_active_low=_as_bool("LIMIT_SWITCH_ACTIVE_LOW", True),
            servo_move_to_strike_time=float(
                os.getenv("SERVO_MOVE_TO_STRIKE_TIME", "0.30")
            ),
            servo_strike_hold_time=float(
                os.getenv("SERVO_STRIKE_HOLD_TIME", "0.06")
            ),
            servo_return_home_time=float(
                os.getenv("SERVO_RETURN_HOME_TIME", "0.30")
            ),
            servo_home_settle_time=float(
                os.getenv("SERVO_HOME_SETTLE_TIME", "0.18")
            ),
            camera_device_index=_as_int("CAMERA_DEVICE_INDEX", 0),
            camera_device_path=os.getenv("CAMERA_DEVICE_PATH", "").strip(),
            camera_device=(
                os.getenv("CAMERA_DEVICE", "/dev/video0").strip()
                or "/dev/video0"
            ),
            camera_width=_as_int("CAMERA_WIDTH", 1920),
            camera_height=_as_int("CAMERA_HEIGHT", 1080),
            # Mặc định không ghi đè các control hình ảnh của camera.
            # Frame camera trả về được giữ nguyên màu trong luồng capture/preview.
            camera_fourcc=os.getenv("CAMERA_FOURCC", "MJPG").strip().upper(),
            camera_fps=float(os.getenv("CAMERA_FPS", "15")),
            camera_brightness=_as_optional_float("CAMERA_BRIGHTNESS", None),
            camera_contrast=_as_optional_float("CAMERA_CONTRAST", None),
            camera_saturation=_as_optional_float("CAMERA_SATURATION", None),
            camera_gamma=_as_optional_float("CAMERA_GAMMA", None),
            camera_sharpness=_as_optional_float("CAMERA_SHARPNESS", None),
            camera_backlight_compensation=_as_optional_float(
                "CAMERA_BACKLIGHT_COMPENSATION", None
            ),
            camera_power_line_frequency=_as_optional_float(
                "CAMERA_POWER_LINE_FREQUENCY", 1.0
            ),
            camera_auto_white_balance=_as_bool("CAMERA_AUTO_WHITE_BALANCE", True),
            camera_white_balance_temperature=_as_optional_float(
                "CAMERA_WHITE_BALANCE_TEMPERATURE", None
            ),
            camera_auto_focus=_as_bool("CAMERA_AUTO_FOCUS", True),
            # Mặc định dùng màu nguyên bản của driver, không bù thêm bằng phần mềm.
            camera_red_gain=float(os.getenv("CAMERA_RED_GAIN", "1.0")),
            camera_green_gain=float(os.getenv("CAMERA_GREEN_GAIN", "1.0")),
            camera_blue_gain=float(os.getenv("CAMERA_BLUE_GAIN", "1.0")),
            camera_color_saturation_scale=float(
                os.getenv("CAMERA_COLOR_SATURATION_SCALE", "1.0")
            ),
            camera_histogram_equalization_enabled=_as_bool(
                "CAMERA_HISTOGRAM_EQUALIZATION", False
            ),
            camera_clahe_clip_limit=float(
                os.getenv("CAMERA_CLAHE_CLIP_LIMIT", "1.5")
            ),
            camera_clahe_grid_size=_as_int("CAMERA_CLAHE_GRID_SIZE", 8),
            camera_warmup_seconds=float(
                os.getenv("CAMERA_WARMUP_SECONDS", "3.0")
            ),
            camera_lock_white_balance=_as_bool(
                "CAMERA_LOCK_WHITE_BALANCE", False
            ),
            camera_lock_exposure=_as_bool("CAMERA_LOCK_EXPOSURE", False),
            camera_warmup_frames=_as_int("CAMERA_WARMUP_FRAMES", 5),
            camera_retry_count=_as_int("CAMERA_RETRY_COUNT", 2),
            camera_preview_enabled=_as_bool("CAMERA_PREVIEW_ENABLED", True),
            camera_preview_host=os.getenv("CAMERA_PREVIEW_HOST", "0.0.0.0"),
            camera_preview_port=_as_int("CAMERA_PREVIEW_PORT", 8081),
            camera_preview_fps=float(os.getenv("CAMERA_PREVIEW_FPS", "6")),
            camera_preview_jpeg_quality=_as_int("CAMERA_PREVIEW_JPEG_QUALITY", 95),
            camera_capture_jpeg_quality=_as_int(
                "CAMERA_CAPTURE_JPEG_QUALITY", 100
            ),
            camera_focus_sample_frames=_as_int("CAMERA_FOCUS_SAMPLE_FRAMES", 8),
            camera_blur_threshold=float(os.getenv("CAMERA_BLUR_THRESHOLD", "18.0")),
            camera_roi_path=Path(
                os.getenv(
                    "CAMERA_ROI_PATH",
                    str(PROJECT_DIR / "config" / "camera_roi.json"),
                )
            ).expanduser(),
            capture_output_dir=Path(
                os.getenv("CAPTURE_OUTPUT_DIR", str(PROJECT_DIR / "captures"))
            ).expanduser(),
            model_path=Path(
                os.getenv(
                    "MODEL_PATH",
                    str(PROJECT_DIR.parent / "model" / "mobilenetv1_best.keras"),
                )
            ).expanduser(),
            image_model_python=Path(
                os.getenv(
                    "IMAGE_MODEL_PYTHON",
                    str(PROJECT_DIR.parent / ".venv" / "bin" / "python"),
                )
            ).expanduser(),
            model_classes_path=Path(
                os.getenv("MODEL_CLASSES_PATH", str(PROJECT_DIR / "config" / "classes.txt"))
            ).expanduser(),
            model_input_width=_as_int("MODEL_INPUT_WIDTH", 224),
            model_input_height=_as_int("MODEL_INPUT_HEIGHT", 224),
            model_normalization=os.getenv("MODEL_NORMALIZATION", "zero_one").lower(),
            model_output_type=os.getenv("MODEL_OUTPUT_TYPE", "auto").lower(),
            model_confidence_threshold=float(
                os.getenv("MODEL_CONFIDENCE_THRESHOLD", "0.5")
            ),
            result_display_seconds=float(os.getenv("RESULT_DISPLAY_SECONDS", "15")),
            error_display_seconds=float(os.getenv("ERROR_DISPLAY_SECONDS", "2")),
            delete_image_after_inference=_as_bool("DELETE_IMAGE_AFTER_INFERENCE", False),
            audio_model_path=Path(
                os.getenv(
                    "AUDIO_MODEL_PATH",
                    str(PROJECT_DIR.parent / "model" / "model-audio.pt"),
                )
            ).expanduser(),
            audio_model_python=Path(
                os.getenv(
                    "AUDIO_MODEL_PYTHON",
                    str(PROJECT_DIR.parent / ".venv" / "bin" / "python"),
                )
            ).expanduser(),
            audio_model_confidence_threshold=float(
                os.getenv("AUDIO_MODEL_CONFIDENCE_THRESHOLD", "0.0")
            ),
            audio_record_seconds=float(os.getenv("AUDIO_RECORD_SECONDS", "5.0")),
            capture_images_per_job=_as_int("CAPTURE_IMAGES_PER_JOB", 3),
            capture_image_interval_seconds=float(
                os.getenv("CAPTURE_IMAGE_INTERVAL_SECONDS", "3.0")
            ),
            # Trọng số fusion có thể thay đổi sau khi tối ưu trên tập validation.
            fusion_image_weight=float(os.getenv("FUSION_IMAGE_WEIGHT", "0.6")),
            fusion_audio_weight=float(os.getenv("FUSION_AUDIO_WEIGHT", "0.4")),
            durian_presence_enabled=_as_bool("DURIAN_PRESENCE_ENABLED", True),
            empty_scene_reference_path=Path(
                os.getenv(
                    "EMPTY_SCENE_REFERENCE_PATH",
                    str(PROJECT_DIR / "config" / "empty_scene.jpg"),
                )
            ).expanduser(),
            durian_presence_difference_threshold=float(
                os.getenv("DURIAN_PRESENCE_DIFFERENCE_THRESHOLD", "6.0")
            ),
            hardware_mock=_as_bool("HARDWARE_MOCK", False),
            mock_image_path=Path(mock_path).expanduser() if mock_path else None,
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        """Reject invalid values early with actionable messages."""
        if self.lcd_columns <= 0 or self.lcd_rows <= 0:
            raise ValueError("LCD dimensions must be positive")
        if min(
            self.servo_move_to_strike_time,
            self.servo_strike_hold_time,
            self.servo_return_home_time,
            self.servo_home_settle_time,
        ) < 0.0:
            raise ValueError("Servo timing values cannot be negative")
        if self.camera_width <= 0 or self.camera_height <= 0:
            raise ValueError("Camera dimensions must be positive")
        if len(self.camera_fourcc) != 4:
            raise ValueError("CAMERA_FOURCC must contain exactly 4 characters")
        if self.camera_fps <= 0:
            raise ValueError("CAMERA_FPS must be positive")
        if self.camera_warmup_seconds < 0.0:
            raise ValueError("CAMERA_WARMUP_SECONDS cannot be negative")
        if min(
            self.camera_red_gain,
            self.camera_green_gain,
            self.camera_blue_gain,
            self.camera_color_saturation_scale,
        ) <= 0:
            raise ValueError("Camera colour gains and saturation scale must be positive")
        if (
            self.camera_white_balance_temperature is not None
            and not 2000.0 <= self.camera_white_balance_temperature <= 10000.0
        ):
            raise ValueError(
                "CAMERA_WHITE_BALANCE_TEMPERATURE must be between 2000 and 10000 K"
            )
        if self.camera_clahe_clip_limit <= 0.0:
            raise ValueError("CAMERA_CLAHE_CLIP_LIMIT must be positive")
        if self.camera_clahe_grid_size <= 0:
            raise ValueError("CAMERA_CLAHE_GRID_SIZE must be positive")
        if self.camera_retry_count < 0 or self.camera_warmup_frames < 0:
            raise ValueError("Camera retry and warm-up values cannot be negative")
        if self.camera_preview_port <= 0 or self.camera_preview_port > 65535:
            raise ValueError("CAMERA_PREVIEW_PORT must be between 1 and 65535")
        if self.camera_preview_fps <= 0:
            raise ValueError("CAMERA_PREVIEW_FPS must be positive")
        if not 1 <= self.camera_preview_jpeg_quality <= 100:
            raise ValueError("CAMERA_PREVIEW_JPEG_QUALITY must be between 1 and 100")
        if not 1 <= self.camera_capture_jpeg_quality <= 100:
            raise ValueError("CAMERA_CAPTURE_JPEG_QUALITY must be between 1 and 100")
        if self.camera_focus_sample_frames <= 0:
            raise ValueError("CAMERA_FOCUS_SAMPLE_FRAMES must be positive")
        if self.camera_blur_threshold < 0.0:
            raise ValueError("CAMERA_BLUR_THRESHOLD cannot be negative")
        if self.model_normalization not in {"zero_one", "minus_one_one", "none"}:
            raise ValueError(f"Unsupported MODEL_NORMALIZATION: {self.model_normalization}")
        if self.model_output_type not in {"auto", "softmax", "sigmoid", "logits"}:
            raise ValueError(f"Unsupported MODEL_OUTPUT_TYPE: {self.model_output_type}")
        if not 0.0 <= self.model_confidence_threshold <= 1.0:
            raise ValueError("MODEL_CONFIDENCE_THRESHOLD must be between 0 and 1")
        if not 0.0 <= self.audio_model_confidence_threshold <= 1.0:
            raise ValueError("AUDIO_MODEL_CONFIDENCE_THRESHOLD must be between 0 and 1")
        if self.audio_record_seconds <= 0.0:
            raise ValueError("AUDIO_RECORD_SECONDS must be positive")
        if self.capture_images_per_job <= 0:
            raise ValueError("CAPTURE_IMAGES_PER_JOB must be positive")
        if self.capture_image_interval_seconds < 0.0:
            raise ValueError("CAPTURE_IMAGE_INTERVAL_SECONDS cannot be negative")
        if self.fusion_image_weight < 0.0 or self.fusion_audio_weight < 0.0:
            raise ValueError("Fusion weights cannot be negative")
        if abs(self.fusion_image_weight + self.fusion_audio_weight - 1.0) > 1e-9:
            raise ValueError("FUSION_IMAGE_WEIGHT + FUSION_AUDIO_WEIGHT must equal 1")
        if self.durian_presence_difference_threshold < 0.0:
            raise ValueError("DURIAN_PRESENCE_DIFFERENCE_THRESHOLD cannot be negative")
