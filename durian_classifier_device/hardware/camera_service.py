"""Single-instance OpenCV camera capture service."""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from inference.exceptions import CameraBlurError, CameraError


LOGGER = logging.getLogger(__name__)


CONTROL_LINE_PATTERN = re.compile(
    r"^\s*([a-zA-Z0-9_]+)\s+0x[0-9a-fA-F]+\s+\(([^)]+)\)\s*:\s*(.*)$"
)
CONTROL_VALUE_PATTERN = re.compile(
    r"\b(min|max|step|default|value)=(-?\d+)\b"
)
CONTROL_MENU_PATTERN = re.compile(r"^\s+(-?\d+):\s+(.+?)\s*$")


class CameraService:
    """Own one camera instance and serialize all captures."""

    def __init__(
        self,
        device_index: int | str,
        width: int,
        height: int,
        warmup_frames: int,
        retry_count: int,
        output_dir: Path,
        mock: bool = False,
        mock_image_path: Path | None = None,
        fourcc: str = "MJPG",
        fps: float = 30.0,

        # Để None nhằm giữ nguyên giá trị mặc định của camera.
        brightness: float | None = None,
        contrast: float | None = None,
        saturation: float | None = None,
        gamma: float | None = None,
        sharpness: float | None = None,
        backlight_compensation: float | None = None,
        power_line_frequency: float | None = 1,

        # Bật tự động cân bằng trắng.
        auto_white_balance: bool | None = True,
        white_balance_temperature: float | None = None,
        # Tự bật lấy nét liên tục nếu driver camera có control tương ứng.
        auto_focus: bool = True,

        # Các giá trị bằng 1.0 nghĩa là không chỉnh màu bằng phần mềm.
        red_gain: float = 1.0,
        green_gain: float = 1.0,
        blue_gain: float = 1.0,
        color_saturation_scale: float = 1.0,
        roi_path: Path | None = None,
        histogram_equalization_enabled: bool = False,
        clahe_clip_limit: float = 1.5,
        clahe_grid_size: int = 8,
        camera_device: str = "/dev/video0",
        warmup_seconds: float = 3.0,
        lock_white_balance: bool = False,
        lock_exposure: bool = False,
        control_timeout_seconds: float = 3.0,
        capture_jpeg_quality: int = 100,
        focus_sample_frames: int = 8,
        blur_threshold: float = 18.0,
    ) -> None:
        self.device_index = device_index
        self.width = width
        self.height = height
        self.warmup_frames = warmup_frames
        self.retry_count = retry_count
        self.output_dir = output_dir
        self.capture_jpeg_quality = capture_jpeg_quality
        self.focus_sample_frames = focus_sample_frames
        self.blur_threshold = blur_threshold
        self.mock = mock
        self.mock_image_path = mock_image_path

        self.fourcc = fourcc
        self.fps = fps

        # None nghĩa là không ép thông số, giữ mặc định của driver camera.
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.gamma = gamma
        self.sharpness = sharpness
        self.backlight_compensation = backlight_compensation
        self.power_line_frequency = power_line_frequency
        self.auto_white_balance = auto_white_balance
        self.white_balance_temperature = white_balance_temperature
        self.auto_focus = auto_focus

        # Giữ các tham số cũ để tương thích cấu hình, nhưng phiên bản này
        # không áp dụng bất kỳ phép bù màu phần mềm nào lên frame camera.
        self.red_gain = red_gain
        self.green_gain = green_gain
        self.blue_gain = blue_gain
        self.color_saturation_scale = color_saturation_scale
        self.histogram_equalization_enabled = histogram_equalization_enabled
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_grid_size = clahe_grid_size

        # Hai cờ khóa được giữ để tương thích API cũ nhưng không còn được dùng.
        # Camera luôn giữ auto white balance và auto exposure trong suốt phiên.
        self.camera_device = camera_device
        self.warmup_seconds = warmup_seconds
        self.lock_white_balance = lock_white_balance
        self.lock_exposure = lock_exposure
        self.control_timeout_seconds = control_timeout_seconds
        self._active_control_device: str | None = None
        self._camera_controls: dict[str, dict[str, object]] = {}

        # ROI dùng tọa độ chuẩn hóa 0..1 để không phụ thuộc độ phân giải camera.
        self.roi_path = Path(roi_path) if roi_path is not None else None
        self._roi_lock = threading.Lock()
        self._roi = self._load_roi()

        self._capture = None
        self._lock = threading.Lock()
        self.is_connected = False
        # Ghi nhớ nguồn mạng để giữ nguyên màu đã được điện thoại xử lý.
        self._network_stream = False
        # Mỗi phiên camera chỉ ghi một cặp ảnh debug để không ghi đĩa liên tục.
        self._debug_raw_saved = False
        self._debug_preview_saved = False

    @staticmethod
    def _full_frame_roi() -> dict[str, float]:
        """Trả vùng mặc định bao phủ toàn bộ khung hình."""
        return {"x": 0.0, "y": 0.0, "width": 1.0, "height": 1.0}

    def _load_roi(self) -> dict[str, float]:
        """Đọc ROI đã lưu; cấu hình hỏng sẽ an toàn quay về toàn khung."""
        if self.roi_path is None or not self.roi_path.is_file():
            return self._full_frame_roi()
        try:
            payload = json.loads(self.roi_path.read_text(encoding="utf-8"))
            return self._validated_roi(payload)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            LOGGER.exception("Invalid camera ROI file; using the full frame")
            return self._full_frame_roi()

    @staticmethod
    def _validated_roi(payload: object) -> dict[str, float]:
        """Kiểm tra ROI chữ nhật hợp lệ trong hệ tọa độ chuẩn hóa."""
        if not isinstance(payload, dict):
            raise ValueError("ROI must be a JSON object")
        try:
            x = float(payload["x"])
            y = float(payload["y"])
            width = float(payload["width"])
            height = float(payload["height"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("ROI requires numeric x, y, width and height") from exc
        if x < 0.0 or y < 0.0 or width < 0.02 or height < 0.02:
            raise ValueError("ROI is outside the frame or too small")
        if x + width > 1.000001 or y + height > 1.000001:
            raise ValueError("ROI exceeds the frame")
        return {
            "x": min(x, 1.0),
            "y": min(y, 1.0),
            "width": min(width, 1.0 - x),
            "height": min(height, 1.0 - y),
        }

    def get_roi(self) -> dict[str, float]:
        """Lấy bản sao ROI hiện tại để preview vẽ đường bao."""
        with self._roi_lock:
            return dict(self._roi)

    def set_roi(self, payload: object) -> dict[str, float]:
        """Cập nhật ROI và lưu bền vững để lần khởi động sau dùng lại."""
        roi = self._validated_roi(payload)
        with self._roi_lock:
            self._roi = roi
            if self.roi_path is not None:
                self.roi_path.parent.mkdir(parents=True, exist_ok=True)
                temporary_path = self.roi_path.with_suffix(
                    self.roi_path.suffix + ".tmp"
                )
                temporary_path.write_text(
                    json.dumps(roi, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                temporary_path.replace(self.roi_path)
        LOGGER.info("Camera detection ROI updated: %s", roi)
        return dict(roi)

    def _crop_to_roi(self, frame):
        """Cắt frame theo ROI hiện tại trước khi lưu và đưa vào model."""
        left, top, right, bottom = self._roi_pixel_bounds(frame)
        # Model luôn nhận bản sao crop trực tiếp từ frame gốc, không từ preview.
        return frame[top:bottom, left:right].copy()

    def _roi_pixel_bounds(self, frame) -> tuple[int, int, int, int]:
        """Đổi ROI chuẩn hóa thành tọa độ pixel dùng chung cho crop/preview."""
        roi = self.get_roi()
        frame_height, frame_width = frame.shape[:2]
        left = max(0, min(frame_width - 1, round(roi["x"] * frame_width)))
        top = max(0, min(frame_height - 1, round(roi["y"] * frame_height)))
        right = max(
            left + 1,
            min(frame_width, round((roi["x"] + roi["width"]) * frame_width)),
        )
        bottom = max(
            top + 1,
            min(frame_height, round((roi["y"] + roi["height"]) * frame_height)),
        )
        return left, top, right, bottom

    def _draw_preview_roi(self, preview_frame) -> None:
        """Chỉ vẽ khung và chữ xanh lên bản sao dành riêng cho preview."""
        x1, y1, x2, y2 = self._roi_pixel_bounds(preview_frame)
        cv2.rectangle(
            preview_frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 80),
            3,
        )
        cv2.putText(
            preview_frame,
            "DETECTION AREA",
            (x1 + 10, min(y2 - 5, y1 + 25)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 80),
            2,
            cv2.LINE_AA,
        )

    def _control_device_path(self, open_device: int | str | None = None) -> str:
        """Đổi nguồn OpenCV thành node /dev/videoN dùng cho v4l2-ctl."""
        if self._active_control_device:
            return self._active_control_device
        if isinstance(open_device, int):
            return f"/dev/video{open_device}"
        if isinstance(open_device, str) and open_device.startswith("/dev/"):
            try:
                return str(Path(open_device).resolve())
            except OSError:
                return open_device
        return self.camera_device

    def _run_v4l2(
        self,
        arguments: list[str],
        device: str | None = None,
    ) -> subprocess.CompletedProcess[str] | None:
        """Chạy v4l2-ctl an toàn, không shell và không làm dừng ứng dụng."""
        executable = shutil.which("v4l2-ctl")
        if executable is None:
            LOGGER.warning("v4l2-ctl is unavailable; using OpenCV camera controls")
            return None
        command = [
            executable,
            "-d",
            device or self._control_device_path(),
            *arguments,
        ]
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=self.control_timeout_seconds,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            LOGGER.warning("Camera control command failed: %s (%s)", command, exc)
            return None
        if result.returncode != 0:
            error = result.stderr.strip() or result.stdout.strip() or "unknown error"
            LOGGER.warning(
                "Camera control command returned %s: %s (%s)",
                result.returncode,
                command,
                error,
            )
            return None
        return result

    def list_camera_controls(
        self,
        device: str | None = None,
    ) -> dict[str, dict[str, object]]:
        """Liệt kê control và menu thật sự trước khi đặt bất kỳ giá trị nào."""
        result = self._run_v4l2(["--list-ctrls-menus"], device)
        if result is None:
            self._camera_controls = {}
            return {}

        controls: dict[str, dict[str, object]] = {}
        current_control: str | None = None
        for raw_line in result.stdout.splitlines():
            match = CONTROL_LINE_PATTERN.match(raw_line)
            if match is not None:
                name, control_type, details = match.groups()
                metadata: dict[str, object] = {"type": control_type, "menus": {}}
                for key, raw_value in CONTROL_VALUE_PATTERN.findall(details):
                    metadata[key] = int(raw_value)
                controls[name] = metadata
                current_control = name
                continue

            menu_match = CONTROL_MENU_PATTERN.match(raw_line)
            if menu_match is None or current_control is None:
                continue
            menu_value, menu_label = menu_match.groups()
            menus = controls[current_control].get("menus")
            if isinstance(menus, dict):
                menus[int(menu_value)] = menu_label

        self._camera_controls = controls
        if controls:
            LOGGER.info(
                "Camera controls detected on %s: %s",
                device or self._control_device_path(),
                sorted(controls),
            )
        else:
            LOGGER.warning("No V4L2 camera controls were detected")
        return {name: dict(metadata) for name, metadata in controls.items()}

    def log_all_camera_controls(self, stage: str) -> None:
        """Ghi toàn bộ trạng thái V4L2 trước/sau cấu hình để đối chiếu."""
        result = self._run_v4l2(["--all"])
        if result is None:
            LOGGER.warning("Cannot read all camera controls (%s configuration)", stage)
            return
        LOGGER.info("Camera controls %s configuration:\n%s", stage, result.stdout.rstrip())

    def get_camera_control(self, name: str) -> int | None:
        """Đọc một control nếu camera hỗ trợ, nếu không trả None."""
        if name not in self._camera_controls:
            LOGGER.info("Camera control %s is unsupported; read skipped", name)
            return None
        result = self._run_v4l2([f"--get-ctrl={name}"])
        if result is None:
            return None
        match = re.search(rf"\b{re.escape(name)}\s*:\s*(-?\d+)", result.stdout)
        if match is None:
            LOGGER.warning("Cannot parse current camera control %s", name)
            return None
        value = int(match.group(1))
        LOGGER.info("Camera control %s read value=%s", name, value)
        return value

    def set_camera_control(self, name: str, value: int | float) -> bool:
        """Đặt một control được hỗ trợ và ghi log rõ thành công/thất bại."""
        if name not in self._camera_controls:
            LOGGER.info("Camera control %s is unsupported; setting skipped", name)
            return False
        numeric_value = int(round(float(value)))
        metadata = self._camera_controls[name]
        minimum = metadata.get("min")
        maximum = metadata.get("max")
        if (
            isinstance(minimum, int)
            and isinstance(maximum, int)
            and not minimum <= numeric_value <= maximum
        ):
            LOGGER.warning(
                "Camera control %s=%s is outside supported range %s..%s; skipped",
                name,
                numeric_value,
                minimum,
                maximum,
            )
            return False
        result = self._run_v4l2([f"--set-ctrl={name}={numeric_value}"])
        if result is None:
            LOGGER.warning("Camera control %s=%s could not be set", name, numeric_value)
            return False
        LOGGER.info("Camera control %s set successfully: %s", name, numeric_value)
        return True

    def _find_control(self, *names: str) -> str | None:
        """Tìm tên đầu tiên được driver hỗ trợ trong danh sách alias."""
        for name in names:
            if name in self._camera_controls:
                return name
        LOGGER.info("Camera controls %s are unsupported; skipped", names)
        return None

    def _apply_initial_v4l2_controls(self) -> None:
        """Khôi phục mặc định driver, trừ control được cấu hình rõ."""
        configured_controls = {
            "brightness": self.brightness,
            "contrast": self.contrast,
            "saturation": self.saturation,
            "gamma": self.gamma,
            "sharpness": self.sharpness,
            "backlight_compensation": self.backlight_compensation,
            "power_line_frequency": self.power_line_frequency,
        }
        for name, value in configured_controls.items():
            if value is None:
                metadata = self._camera_controls.get(name, {})
                default_value = metadata.get("default")
                if not isinstance(default_value, int):
                    LOGGER.info(
                        "Camera control %s has no driver default; left unchanged",
                        name,
                    )
                    continue
                value = default_value
                LOGGER.info(
                    "Restoring camera control %s to driver default=%s",
                    name,
                    default_value,
                )
            self.set_camera_control(name, value)

    def _auto_control_value(self, name: str) -> int | None:
        """Chọn chế độ auto từ kiểu/menu driver, không giả định mã menu."""
        metadata = self._camera_controls.get(name, {})
        if metadata.get("type") == "bool":
            return 1

        menus = metadata.get("menus")
        if not isinstance(menus, dict):
            LOGGER.warning("Control %s has no driver menu; auto mode not changed", name)
            return None

        # UVC thường gọi auto exposure là Auto Mode hoặc Aperture Priority Mode.
        # Chỉ dùng nhãn do chính driver trả về và tuyệt đối không chọn Manual Mode.
        priorities = ("auto mode", "aperture priority mode")
        normalized = {
            int(value): str(label).strip().lower()
            for value, label in menus.items()
        }
        for wanted_label in priorities:
            for value, label in normalized.items():
                if label == wanted_label:
                    LOGGER.info(
                        "Camera control %s auto menu detected: %s=%s",
                        name,
                        value,
                        menus[value],
                    )
                    return value
        # Một số camera đặt tên menu focus là Continuous Auto Focus/Autofocus.
        for value, label in normalized.items():
            if "manual" not in label and (
                "auto focus" in label
                or "autofocus" in label
                or "continuous focus" in label
            ):
                LOGGER.info(
                    "Camera control %s auto menu detected: %s=%s",
                    name,
                    value,
                    menus[value],
                )
                return value
        LOGGER.warning(
            "Control %s has no recognized auto menu in %s; value not changed",
            name,
            menus,
        )
        return None

    def warmup_camera(self, capture) -> None:
        """Đọc bỏ frame trong thời gian warm-up để auto WB/exposure ổn định."""
        LOGGER.info("Camera warm-up started: %.1f seconds", self.warmup_seconds)
        deadline = time.monotonic() + self.warmup_seconds
        discarded_frames = 0
        while time.monotonic() < deadline:
            success, frame = capture.read()
            if success and frame is not None:
                discarded_frames += 1
            else:
                time.sleep(0.02)
        LOGGER.info(
            "Camera warm-up completed; discarded_frames=%s",
            discarded_frames,
        )

    def configure_automatic_controls(self, capture, open_device: int | str) -> None:
        """Bật WB/exposure tự động, warm-up và giữ auto trong suốt phiên."""
        self._active_control_device = self._control_device_path(open_device)
        controls = self.list_camera_controls(self._active_control_device)
        self.log_all_camera_controls("before")
        if not controls:
            LOGGER.warning(
                "V4L2 controls unavailable; leaving current controls unchanged"
            )
            self.warmup_camera(capture)
            self.log_all_camera_controls("after")
            LOGGER.info("Camera is ready for classification with current controls")
            return

        self._apply_initial_v4l2_controls()
        white_balance_auto = self._find_control(
            "white_balance_temperature_auto",
            "white_balance_automatic",
            "auto_white_balance_temperature",
        )
        exposure_auto = self._find_control("exposure_auto", "auto_exposure")
        focus_auto = self._find_control(
            "focus_automatic_continuous",
            "focus_auto",
            "auto_focus",
        )

        if white_balance_auto is not None:
            auto_value = self._auto_control_value(white_balance_auto)
            if auto_value is not None and self.set_camera_control(
                white_balance_auto, auto_value
            ):
                LOGGER.info("Auto white balance enabled")

        if exposure_auto is not None:
            auto_value = self._auto_control_value(exposure_auto)
            if auto_value is not None and self.set_camera_control(
                exposure_auto, auto_value
            ):
                LOGGER.info("Auto exposure enabled")

        if self.auto_focus and focus_auto is not None:
            auto_value = self._auto_control_value(focus_auto)
            if auto_value is not None and self.set_camera_control(
                focus_auto, auto_value
            ):
                LOGGER.info("Continuous auto focus enabled")
        elif not self.auto_focus:
            LOGGER.info("Auto focus disabled by configuration")
        else:
            # Camera fixed-focus vẫn tiếp tục hoạt động bình thường.
            LOGGER.info("Auto focus unsupported; camera uses fixed focus")

        # Chỉ đọc bỏ frame; không đọc hay gán WB/exposure thủ công sau warm-up.
        self.warmup_camera(capture)
        self.log_all_camera_controls("after")
        LOGGER.info(
            "Camera is ready for classification; auto white balance and auto exposure remain enabled"
        )

    def initialize(self) -> None:
        """Open the configured camera exactly once, or validate the mock image."""

        self.output_dir.mkdir(parents=True, exist_ok=True)

        if self.mock:
            if self.mock_image_path is None or not self.mock_image_path.is_file():
                raise CameraError(
                    "HARDWARE_MOCK requires an existing MOCK_IMAGE_PATH"
                )

            self.is_connected = True

            LOGGER.info(
                "Camera mock initialized with image %s",
                self.mock_image_path,
            )
            return

        with self._lock:
            if self._capture is not None:
                return

            self._open_capture_locked()

    def _open_capture_locked(self) -> None:
        """Mở camera và cấu hình định dạng hình ảnh."""

        open_device = self._resolve_device_index()
        is_network_stream = self._is_network_stream(open_device)

        # Camera điện thoại dùng luồng HTTP/RTSP qua FFmpeg;
        # camera USB trên Raspberry Pi tiếp tục dùng V4L2 như trước.
        backend = cv2.CAP_FFMPEG if is_network_stream else cv2.CAP_V4L2
        capture = cv2.VideoCapture(open_device, backend)

        # Một số bản OpenCV không bật FFmpeg nhưng vẫn tự nhận được luồng mạng.
        if is_network_stream and not capture.isOpened():
            capture.release()
            LOGGER.warning(
                "FFmpeg backend could not open phone camera; trying automatic backend"
            )
            capture = cv2.VideoCapture(open_device)

        if not capture.isOpened():
            capture.release()

            raise CameraError(
                f"Cannot open camera {self.device_index}; "
                "it may be used by another process"
            )

        capture.set(
            cv2.CAP_PROP_BUFFERSIZE,
            1,
        )

        if is_network_stream:
            # Độ phân giải, FPS, exposure và màu của luồng mạng do điện thoại đặt.
            LOGGER.info(
                "Phone camera stream opened; controls are managed by the phone"
            )
        else:
            # Chọn MJPEG trước khi cấu hình độ phân giải cho camera USB.
            requested_fourcc = cv2.VideoWriter_fourcc(*self.fourcc)
            capture.set(cv2.CAP_PROP_FOURCC, requested_fourcc)
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            capture.set(cv2.CAP_PROP_FPS, self.fps)
            # Chỉ dùng menu control thật của V4L2 và luôn giữ hai chế độ auto.
            self._active_control_device = None
            try:
                self.configure_automatic_controls(capture, open_device)
            except Exception:
                # Lỗi control không được phép làm dừng camera/pipeline phân loại.
                LOGGER.exception(
                    "Camera automatic control setup failed; continuing with current controls"
                )

        self._capture = capture
        self._network_stream = is_network_stream
        self._debug_raw_saved = False
        self._debug_preview_saved = False
        self.is_connected = True

        actual_fourcc = self._decode_fourcc(
            capture.get(cv2.CAP_PROP_FOURCC)
        )

        LOGGER.info(
            "Camera initialized: "
            "device=%s open_device=%s backend=%s format=%s size=%sx%s fps=%.1f",
            self.device_index,
            open_device,
            "FFMPEG/AUTO" if is_network_stream else "V4L2",
            actual_fourcc,
            int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            capture.get(cv2.CAP_PROP_FPS),
        )

    @classmethod
    def _set_property_if_configured(
        cls,
        capture,
        property_id: int,
        value: float | None,
        name: str,
    ) -> None:
        """Giữ mặc định driver nếu giá trị cấu hình là None."""

        if value is None:
            LOGGER.info(
                "Camera control %s uses driver default",
                name,
            )
            return

        cls._set_property(
            capture,
            property_id,
            value,
            name,
        )

    @staticmethod
    def _set_property(
        capture,
        property_id: int,
        value: float,
        name: str,
    ) -> None:
        """Áp dụng thông số camera và ghi lại kết quả."""

        accepted = capture.set(
            property_id,
            value,
        )

        actual = capture.get(property_id)

        if not accepted:
            LOGGER.warning(
                "Camera rejected %s=%s; actual=%s",
                name,
                value,
                actual,
            )
        else:
            LOGGER.info(
                "Camera control %s requested=%s actual=%s",
                name,
                value,
                actual,
            )

    @staticmethod
    def _decode_fourcc(value: float) -> str:
        """Đổi mã FOURCC số thành chuỗi dễ đọc."""

        code = int(value)

        return "".join(
            chr((code >> (8 * index)) & 0xFF)
            for index in range(4)
        )

    def _resolve_device_index(self) -> int | str:
        """Resolve cấu hình hoặc tự tìm node capture chính của camera USB UVC."""

        if not isinstance(self.device_index, str):
            if self._is_usb_capture_index(self.device_index):
                return self.device_index
            discovered_index = self._discover_usb_camera_index()
            if discovered_index is not None:
                LOGGER.info(
                    "Configured camera index %s is not a USB capture device; "
                    "auto-detected /dev/video%s",
                    self.device_index,
                    discovered_index,
                )
                return discovered_index
            return self.device_index

        if not self.device_index:
            discovered_index = self._discover_usb_camera_index()
            return discovered_index if discovered_index is not None else self.device_index

        if self._is_network_stream(self.device_index):
            return self.device_index

        device_path = Path(self.device_index)

        if not device_path.exists():
            discovered_index = self._discover_usb_camera_index()
            if discovered_index is not None:
                LOGGER.warning(
                    "Configured camera %s does not exist; auto-detected /dev/video%s",
                    self.device_index,
                    discovered_index,
                )
                return discovered_index
            return self.device_index

        try:
            resolved_path = device_path.resolve()
        except OSError:
            return self.device_index

        resolved = str(resolved_path)

        if resolved != self.device_index:
            LOGGER.info(
                "Resolved camera device %s -> %s",
                self.device_index,
                resolved,
            )

        if (
            resolved_path.name.startswith("video")
            and resolved_path.name[5:].isdigit()
        ):
            resolved_index = int(resolved_path.name[5:])

            # Một USB camera có thể tạo thêm node metadata như /dev/video1.
            # Chỉ dùng node capture chính; nếu cấu hình nhầm thì tự dò lại.
            if self._is_usb_capture_index(resolved_index):
                return resolved_index

            discovered_index = self._discover_usb_camera_index()
            if discovered_index is not None:
                LOGGER.warning(
                    "Configured camera %s is not a USB video capture node; "
                    "auto-detected /dev/video%s",
                    self.device_index,
                    discovered_index,
                )
                return discovered_index

        return resolved

    @classmethod
    def _discover_usb_camera_index(cls) -> int | None:
        """Tìm interface capture index 0 được bind với driver uvcvideo."""
        candidates: list[int] = []
        for video_class_path in Path("/sys/class/video4linux").glob("video*"):
            suffix = video_class_path.name[5:]
            if not suffix.isdigit():
                continue
            try:
                interface_index = (video_class_path / "index").read_text().strip()
                driver_name = (video_class_path / "device" / "driver").resolve().name
            except OSError:
                continue
            if interface_index == "0" and driver_name == "uvcvideo":
                candidates.append(int(suffix))
        return min(candidates) if candidates else None

    @classmethod
    def _is_usb_capture_index(cls, device_index: int) -> bool:
        """Kiểm tra video index có phải node capture chính của USB camera không."""
        video_class_path = Path(f"/sys/class/video4linux/video{device_index}")
        try:
            return (
                (video_class_path / "index").read_text().strip() == "0"
                and (video_class_path / "device" / "driver").resolve().name
                == "uvcvideo"
            )
        except OSError:
            return False

    @staticmethod
    def _is_network_stream(device: int | str) -> bool:
        """Nhận diện URL camera điện thoại dùng HTTP, HTTPS hoặc RTSP."""

        if not isinstance(device, str):
            return False

        return device.strip().lower().startswith(
            ("http://", "https://", "rtsp://")
        )

    def _reopen_capture_locked(self) -> None:
        """Đóng và mở lại camera sau khi đọc hình thất bại."""

        LOGGER.warning(
            "Reopening camera after read failure: %s",
            self.device_index,
        )

        if self._capture is not None:
            self._capture.release()

        self._capture = None
        self.is_connected = False

        time.sleep(0.3)

        self._open_capture_locked()

    def _read_frame_locked(
        self,
        attempts: int,
        label: str,
    ):
        """Đọc một frame từ camera với cơ chế thử lại."""

        if self._capture is None or not self._capture.isOpened():
            self._reopen_capture_locked()

        frame = None

        for attempt in range(attempts):
            success, frame = self._capture.read()

            if success and frame is not None:
                # Kiểm tra frame BGR gốc ngay sau read(), trước ROI hay preview.
                self._validate_raw_frame(frame)
                return frame

            LOGGER.warning(
                "Camera %s read attempt %s failed",
                label,
                attempt + 1,
            )

            time.sleep(0.1)

        # Nếu đọc thất bại, mở lại camera và thử thêm lần nữa.
        self._reopen_capture_locked()

        for attempt in range(attempts):
            success, frame = self._capture.read()

            if success and frame is not None:
                self._validate_raw_frame(frame)
                return frame

            LOGGER.warning(
                "Camera %s read attempt %s failed after reopen",
                label,
                attempt + 1,
            )

            time.sleep(0.1)

        raise CameraError(
            f"Camera {label} failed after reopen"
        )

    def _debug_image_path(self, filename: str) -> Path:
        """Đặt ảnh debug cạnh thư mục captures để dễ so sánh trực tiếp."""
        return self.output_dir.parent / filename

    def _validate_raw_frame(self, frame) -> None:
        """Kiểm tra frame gốc luôn là uint8 hợp lệ ngay sau OpenCV read()."""
        if not isinstance(frame, np.ndarray) or frame.size == 0:
            raise CameraError("Camera returned an empty or invalid frame")
        if frame.dtype != np.uint8:
            raise CameraError(f"Camera raw frame must be uint8, got {frame.dtype}")

        minimum = int(frame.min())
        maximum = int(frame.max())
        if minimum < 0 or maximum > 255:
            raise CameraError(
                f"Camera raw frame values outside 0..255: {minimum}..{maximum}"
            )

        if self._debug_raw_saved:
            return
        LOGGER.info(
            "Raw frame: dtype=%s min=%s max=%s mean=%.2f",
            frame.dtype,
            minimum,
            maximum,
            float(frame.mean()),
        )
        # Cờ này cũng giới hạn log thống kê xuống một lần cho mỗi phiên camera.
        self._debug_raw_saved = True

    def _save_preview_debug_frames(self, frame, preview_frame) -> None:
        """Lưu cùng một frame trước/sau khi chỉ vẽ khung ROI để so sánh."""
        if self._debug_preview_saved:
            return
        raw_path = self._debug_image_path("debug_raw.jpg")
        preview_path = self._debug_image_path("debug_preview.jpg")
        jpeg_params = [
            int(cv2.IMWRITE_JPEG_QUALITY),
            int(self.capture_jpeg_quality),
        ]
        raw_saved = cv2.imwrite(str(raw_path), frame, jpeg_params)
        preview_saved = cv2.imwrite(str(preview_path), preview_frame, jpeg_params)
        if raw_saved and preview_saved:
            self._debug_preview_saved = True
            LOGGER.info("Raw preview source saved: %s", raw_path)
            LOGGER.info("Rendered preview frame saved: %s", preview_path)
        else:
            LOGGER.warning(
                "Could not save preview debug frames: raw=%s preview=%s",
                raw_saved,
                preview_saved,
            )

    def _new_path(self) -> Path:
        """Tạo đường dẫn ảnh mới theo thời gian."""

        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S_%f"
        )

        return self.output_dir / f"durian_{timestamp}.jpg"

    @staticmethod
    def _sharpness_score(frame) -> float:
        """Measure edge detail; larger Laplacian variance means a sharper frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def _capture_sharpest_roi_frame(self):
        """Read a short burst and return the sharpest ROI frame."""
        best_frame = None
        best_score = -1.0
        scores: list[float] = []
        for _ in range(self.focus_sample_frames):
            frame = self._read_frame_locked(
                self.retry_count + 1,
                "focus sample",
            )
            roi_frame = self._crop_to_roi(frame)
            score = self._sharpness_score(roi_frame)
            scores.append(score)
            if score > best_score:
                best_frame = roi_frame
                best_score = score
        LOGGER.info(
            "Capture sharpness best=%.2f threshold=%.2f samples=%s",
            best_score,
            self.blur_threshold,
            [round(score, 2) for score in scores],
        )
        if best_frame is None:
            raise CameraError("Camera did not return a focus sample")
        if best_score < self.blur_threshold:
            raise CameraBlurError(
                f"Camera image is too blurry: sharpness={best_score:.2f}, "
                f"required={self.blur_threshold:.2f}"
            )
        return best_frame

    def capture_image(self) -> str:
        """Khởi động camera, chụp và lưu ảnh JPEG."""

        output_path = self._new_path()

        with self._lock:
            if self.mock:
                if self.mock_image_path is None:
                    raise CameraError(
                        "Mock image path is not configured"
                    )

                shutil.copy2(
                    self.mock_image_path,
                    output_path,
                )

                LOGGER.info(
                    "Mock image captured: %s",
                    output_path,
                )

                return str(output_path)

            # Đọc một số frame đầu để camera ổn định ánh sáng và màu.
            for _ in range(self.warmup_frames):
                try:
                    self._read_frame_locked(
                        1,
                        "warmup",
                    )
                except CameraError:
                    LOGGER.debug(
                        "Camera warmup frame failed"
                    )

            # Fixed-focus camera cannot be refocused in software. Select the
            # sharpest frame from a short burst to avoid motion-blurred input.
            frame = self._capture_sharpest_roi_frame()

            if not cv2.imwrite(
                str(output_path),
                frame,
                [
                    int(cv2.IMWRITE_JPEG_QUALITY),
                    int(self.capture_jpeg_quality),
                ],
            ):
                raise CameraError(
                    f"cv2.imwrite failed for {output_path}"
                )

        LOGGER.info(
            "Image captured: %s",
            output_path,
        )

        return str(output_path)

    def read_jpeg(
        self,
        quality: int = 70,
    ) -> bytes:
        """Đọc frame camera và mã hóa thành JPEG cho preview web."""

        with self._lock:
            if self.mock:
                if self.mock_image_path is None:
                    raise CameraError(
                        "Mock image path is not configured"
                    )

                return self.mock_image_path.read_bytes()

            frame = self._read_frame_locked(
                3,
                "preview",
            )
            # Tạo preview trực tiếp từ frame gốc; chỉ thêm khung và chữ ROI.
            preview_frame = frame.copy()
            self._draw_preview_roi(preview_frame)
            self._save_preview_debug_frames(frame, preview_frame)

            encode_params = [
                int(cv2.IMWRITE_JPEG_QUALITY),
                int(quality),
            ]

            ok, encoded = cv2.imencode(
                ".jpg",
                preview_frame,
                encode_params,
            )

            if not ok:
                raise CameraError(
                    "Camera preview JPEG encode failed"
                )

            return encoded.tobytes()

    def close(self) -> None:
        """Đóng camera an toàn."""

        with self._lock:
            try:
                if self._capture is not None:
                    self._capture.release()
            finally:
                self._capture = None
                self._network_stream = False
                self._active_control_device = None
                self._camera_controls = {}
                self._debug_raw_saved = False
                self._debug_preview_saved = False
                self.is_connected = False

                LOGGER.info(
                    "Camera closed"
                )
