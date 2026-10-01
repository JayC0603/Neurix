"""Điều khiển duy nhất Servo 1 MG996R bằng hardware PWM Raspberry Pi 4."""

from __future__ import annotations

import fcntl
import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path


LOGGER = logging.getLogger(__name__)

# BCM numbering: GPIO18 là physical pin 12 và hardware PWM0.
SERVO_1_GPIO_BCM = 18

# Góc logic tuyệt đối. Không bù sai lệch cơ khí bằng cách đổi hai hằng này.
HOME_ANGLE = 90.0
STRIKE_ANGLE = 120.0

SERVO_MIN_ANGLE = 0.0
SERVO_MAX_ANGLE = 180.0
# Calibration riêng Servo 1: logical 90° ánh xạ 1500 us.
SERVO_1_MIN_PULSE = 0.001
SERVO_1_MAX_PULSE = 0.002
MIN_PULSE_WIDTH = SERVO_1_MIN_PULSE
MAX_PULSE_WIDTH = SERVO_1_MAX_PULSE
SERVO_FREQUENCY_HZ = 50
PWM_PERIOD_NS = 1_000_000_000 // SERVO_FREQUENCY_HZ
PWM_SYSFS_ROOT = Path("/sys/class/pwm")
PWM_PROCESS_LOCK = Path("/tmp/durian-mg996r-pwm.lock")

# Ba nhịp gõ, tính cả lần đưa servo về HOME ban đầu, kéo dài 3 giây.
STRIKE_COUNT = 3
STRIKE_SEQUENCE_TIME = 3.0
MOVE_TO_STRIKE_TIME = 0.30
STRIKE_HOLD_TIME = 0.06
RETURN_HOME_TIME = 0.30
HOME_SETTLE_TIME = 0.18

# Tên tương thích với import cũ.
SERVO_1_PIN = SERVO_1_GPIO_BCM
SERVO_MIN_PULSE_WIDTH = MIN_PULSE_WIDTH
SERVO_MAX_PULSE_WIDTH = MAX_PULSE_WIDTH


def angle_to_pulse_width_us(
    angle: float,
    min_pulse_width: float = MIN_PULSE_WIDTH,
    max_pulse_width: float = MAX_PULSE_WIDTH,
) -> int:
    """Đổi góc logic tuyệt đối 0..180 sang pulse width."""
    value = float(angle)
    if not SERVO_MIN_ANGLE <= value <= SERVO_MAX_ANGLE:
        raise ValueError(f"Servo angle must be in [0, 180], got {value}")
    if not 0.0 < min_pulse_width < max_pulse_width:
        raise ValueError("Pulse widths must be positive and min must be below max")
    position = value / (SERVO_MAX_ANGLE - SERVO_MIN_ANGLE)
    pulse_width = min_pulse_width + position * (max_pulse_width - min_pulse_width)
    return round(pulse_width * 1_000_000)


def servo_angle_to_pulse_width_us(servo_number: int, angle: float) -> int:
    """Đổi góc theo calibration của duy nhất Servo 1."""
    if servo_number != 1:
        raise ValueError("Only Servo 1 is supported")
    return angle_to_pulse_width_us(angle, SERVO_1_MIN_PULSE, SERVO_1_MAX_PULSE)


class _HardwarePWMServo:
    """Một kênh kernel hardware PWM; MG996R không trả feedback vị trí."""

    def __init__(
        self,
        channel_path: Path,
        pin: int,
        min_pulse_width: float,
        max_pulse_width: float,
    ) -> None:
        self._pin = pin
        self._closed = False
        self._min_pulse_width = min_pulse_width
        self._max_pulse_width = max_pulse_width
        self._enable_path = channel_path / "enable"
        self._duty_cycle_path = channel_path / "duty_cycle"
        self._period_path = channel_path / "period"
        for path in (self._enable_path, self._duty_cycle_path, self._period_path):
            if not path.is_file():
                raise RuntimeError(f"Hardware PWM control is missing: {path}")
            if not os.access(path, os.R_OK | os.W_OK):
                raise RuntimeError(f"Hardware PWM control is not writable: {path}")
        if self._enable_path.read_text(encoding="ascii").strip() != "0":
            self._write(self._enable_path, 0)

    @staticmethod
    def _write(path: Path, value: int) -> None:
        path.write_text(str(value), encoding="ascii")

    @property
    def angle(self) -> None:
        return None

    @angle.setter
    def angle(self, value: float) -> None:
        if self._closed:
            raise RuntimeError(f"Servo on BCM{self._pin} is closed")
        pulse_us = angle_to_pulse_width_us(
            value, self._min_pulse_width, self._max_pulse_width
        )
        self._write(self._duty_cycle_path, pulse_us * 1_000)
        if self._enable_path.read_text(encoding="ascii").strip() != "1":
            self._write(self._enable_path, 1)

    def detach(self) -> None:
        """Disable PWM rồi clear duty; không để signal tiếp tục phát xung."""
        if self._closed:
            return
        if self._enable_path.read_text(encoding="ascii").strip() != "0":
            self._write(self._enable_path, 0)
        if self._duty_cycle_path.read_text(encoding="ascii").strip() != "0":
            self._write(self._duty_cycle_path, 0)

    def close(self) -> None:
        if not self._closed:
            self.detach()
            self._closed = True


class _HardwarePWMChip:
    """Backend độc quyền cho PWM0/BCM18 của Servo 1."""

    def __init__(self) -> None:
        self._closed = False
        self._lock_file = PWM_PROCESS_LOCK.open("a+", encoding="ascii")
        try:
            fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self._lock_file.close()
            raise RuntimeError(
                "Servo PWM is already owned by another process. Stop the "
                "durian-classifier service or other servo test first."
            ) from exc
        candidates = sorted(PWM_SYSFS_ROOT.glob("pwmchip*"))
        self._chip_path = next(
            (
                path
                for path in candidates
                if (path / "npwm").is_file()
                and int((path / "npwm").read_text(encoding="ascii").strip()) >= 1
                and (path / "pwm0").is_dir()
            ),
            None,
        )
        if self._chip_path is None:
            self.close()
            raise RuntimeError(
                "PWM0 hardware PWM is not ready. Run "
                "scripts/configure-hardware-pwm.sh as root and reboot."
            )

    def create_servo(self, pin: int) -> _HardwarePWMServo:
        if self._closed:
            raise RuntimeError("Hardware PWM chip is closed")
        if pin != SERVO_1_GPIO_BCM:
            raise ValueError("Only Servo 1 on BCM18/PWM0 is supported")
        return _HardwarePWMServo(
            self._chip_path / "pwm0", pin, SERVO_1_MIN_PULSE, SERVO_1_MAX_PULSE
        )

    def close(self) -> None:
        if self._closed:
            return
        fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_UN)
        self._lock_file.close()
        self._closed = True


@dataclass(frozen=True)
class ServoCommand:
    """Một lệnh đã gửi để audit; angle không phải feedback vật lý."""

    servo_number: int
    angle: float
    purpose: str


class ServoController:
    """Single-owner controller cho duy nhất Servo 1."""

    def __init__(
        self,
        *,
        mock: bool,
        move_to_strike_time: float = MOVE_TO_STRIKE_TIME,
        strike_hold_time: float = STRIKE_HOLD_TIME,
        return_home_time: float = RETURN_HOME_TIME,
        home_settle_time: float = HOME_SETTLE_TIME,
        between_strikes_time: float | None = None,
        enabled_servos: tuple[int, ...] = (1,),
        single_servo_sequence_time: float | None = None,
    ) -> None:
        # Tham số legacy bị bỏ qua; timing dùng bốn khoảng rõ ràng ở trên.
        del between_strikes_time, single_servo_sequence_time
        if enabled_servos != (1,):
            raise ValueError("Only Servo 1 is supported; enabled_servos must be (1,)")
        self.mock = bool(mock)
        self.enabled_servos = (1,)
        self.center_angles = (HOME_ANGLE,)
        self.move_to_strike_time = self._validate_duration(
            move_to_strike_time, "move_to_strike_time"
        )
        self.strike_hold_time = self._validate_duration(
            strike_hold_time, "strike_hold_time"
        )
        self.return_home_time = self._validate_duration(
            return_home_time, "return_home_time"
        )
        self.home_settle_time = self._validate_duration(
            home_settle_time, "home_settle_time"
        )
        self.is_initialized = False
        self._factory: _HardwarePWMChip | None = None
        self._servo_1: _HardwarePWMServo | None = None
        self._commanded_angle: float | None = None
        self._attached = False
        self._commands: list[ServoCommand] = []
        self._lock = threading.RLock()
        self._sequence_lock = threading.Lock()

    @staticmethod
    def _validate_duration(duration: float, label: str) -> float:
        value = float(duration)
        if value < 0.0:
            raise ValueError(f"{label} cannot be negative, got {value}")
        return value

    @staticmethod
    def _validate_angle(angle: float, label: str) -> float:
        value = float(angle)
        if not SERVO_MIN_ANGLE <= value <= SERVO_MAX_ANGLE:
            raise ValueError(f"{label} must be between 0 and 180, got {value}")
        return value

    @property
    def angles(self) -> tuple[float | None]:
        """Góc command gần nhất; không phải vị trí vật lý đo được."""
        with self._lock:
            return (self._commanded_angle,)

    @property
    def command_history(self) -> tuple[ServoCommand, ...]:
        with self._lock:
            return tuple(self._commands)

    @property
    def is_pwm_attached(self) -> bool:
        with self._lock:
            return self._attached

    def clear_command_history(self) -> None:
        with self._lock:
            self._commands.clear()

    def is_centered(self) -> bool:
        """Chỉ xác nhận lệnh gần nhất là HOME, không xác nhận góc vật lý."""
        with self._lock:
            return self._commanded_angle == HOME_ANGLE

    def is_servo_centered(self, servo_number: int) -> bool:
        self._require_servo_1(servo_number)
        return self.is_centered()

    def init_servos(self) -> None:
        """Khởi tạo backend Servo 1; chưa phát xung cho tới một lệnh rõ ràng."""
        with self._lock:
            if self.is_initialized:
                return
            try:
                if not self.mock:
                    self._factory = _HardwarePWMChip()
                    self._servo_1 = self._factory.create_servo(SERVO_1_GPIO_BCM)
                self.is_initialized = True
                LOGGER.info("Servo 1 PWM backend initialized; no angle commanded")
            except Exception:
                LOGGER.exception("Could not initialize Servo 1 at HOME")
                self._close_resources_locked()
                raise

    def set_servo_angle(self, servo: object | None, angle: float) -> float:
        """Gửi một góc tuyệt đối; không nội suy và không loop cập nhật."""
        value = self._validate_angle(angle, "servo command")
        if not self.mock:
            if servo is None:
                raise RuntimeError("Servo 1 has not been initialized")
            servo.angle = value
        return value

    def tap_sequence(self) -> None:
        """Gõ Servo 1 ba lần ở 120° và trở về HOME 90°."""
        if not self._sequence_lock.acquire(blocking=False):
            raise RuntimeError("Servo 1 strike sequence is already running")
        try:
            with self._lock:
                self._require_initialized()
                LOGGER.info(
                    "START %d strikes at %.0f degrees in %.2f seconds",
                    STRIKE_COUNT,
                    STRIKE_ANGLE,
                    STRIKE_SEQUENCE_TIME,
                )
                self._set_one_servo_locked(HOME_ANGLE, "HOME")
                time.sleep(self.return_home_time)
                time.sleep(self.home_settle_time)
                for strike_number in range(1, STRIKE_COUNT + 1):
                    self._set_one_servo_locked(
                        STRIKE_ANGLE, f"STRIKE_{strike_number}"
                    )
                    time.sleep(self.move_to_strike_time)
                    time.sleep(self.strike_hold_time)
                    self._set_one_servo_locked(HOME_ANGLE, "HOME")
                    time.sleep(self.return_home_time)
                    time.sleep(self.home_settle_time)
                LOGGER.info("Servo 1 completed %d strikes and returned HOME", STRIKE_COUNT)
        except BaseException:
            LOGGER.exception("Strike sequence interrupted; restoring HOME")
            with self._lock:
                self._recover_to_home_locked()
            raise
        finally:
            try:
                # Chuỗi luôn kết thúc bằng detach sau khi HOME đã chờ đủ.
                # Biến cấu hình legacy không được phép tạo background hold.
                self.release_servos()
                LOGGER.info("Servo 1 cleanup completed")
                LOGGER.info("STOP")
            finally:
                self._sequence_lock.release()

    # Tên cũ được giữ để các caller ngoài project không bị hỏng.
    def strike_once(self) -> None:
        self.tap_sequence()

    def strike(self, servo: int, name: str) -> None:
        del name
        self._require_servo_1(servo)
        self.tap_sequence()

    def move_to_home(
        self, servo_number: int, *, force: bool = False, purpose: str = "HOME"
    ) -> None:
        self._require_servo_1(servo_number)
        with self._lock:
            self._require_initialized()
            if force or self._commanded_angle != HOME_ANGLE or not self._attached:
                self._set_one_servo_locked(HOME_ANGLE, purpose)
                time.sleep(self.return_home_time)
                time.sleep(self.home_settle_time)

    def center_servos(
        self, *, wait: bool = True, release: bool = False, force: bool = False
    ) -> None:
        self.center_servo(1, wait=wait, release=release, force=force)

    def center_servo(
        self,
        servo_number: int,
        *,
        wait: bool = True,
        release: bool = False,
        force: bool = False,
    ) -> None:
        self._require_servo_1(servo_number)
        with self._lock:
            self._require_initialized()
            if force or self._commanded_angle != HOME_ANGLE or not self._attached:
                self._set_one_servo_locked(HOME_ANGLE, "HOME")
                time.sleep(self.return_home_time)
                if wait:
                    time.sleep(self.home_settle_time)
            if release:
                self.release_servos()

    def release_servos(self) -> None:
        """Disable PWM và clear duty; chỉ gọi sau khi HOME đã chờ đủ."""
        with self._lock:
            if not self.mock and self._servo_1 is not None:
                self._servo_1.detach()
            self._attached = False
            LOGGER.info("Servo 1 PWM disabled and duty cleared")

    def release_servo(self, servo_number: int) -> None:
        self._require_servo_1(servo_number)
        self.release_servos()

    def cleanup_servos(self) -> None:
        """Nếu đang ngoài HOME thì HOME -> wait -> settle; sau đó detach/close."""
        with self._lock:
            try:
                if self.is_initialized:
                    self._recover_to_home_locked()
                    self.release_servos()
            except Exception:
                LOGGER.exception("Error while parking/detaching Servo 1")
            finally:
                self._close_resources_locked()
            LOGGER.info("Servo 1 cleanup completed")

    def _set_one_servo_locked(self, angle: float, purpose: str) -> None:
        commanded = self.set_servo_angle(self._servo_1, angle)
        self._commanded_angle = commanded
        self._attached = True
        self._commands.append(ServoCommand(1, commanded, purpose))
        LOGGER.info(
            "Servo 1 -> %s %.0f (%d us)",
            purpose,
            commanded,
            servo_angle_to_pulse_width_us(1, commanded),
        )

    def _recover_to_home_locked(self) -> None:
        if not self.is_initialized or self._commanded_angle is None:
            return
        # Không re-attach nếu chuỗi đã HOME, chờ đủ và detach.
        if self._commanded_angle != HOME_ANGLE:
            self._set_one_servo_locked(HOME_ANGLE, "HOME")
            time.sleep(self.return_home_time)
            time.sleep(self.home_settle_time)
            LOGGER.info("Servo 1 HOME return wait completed")

    def _close_resources_locked(self) -> None:
        if self._servo_1 is not None:
            try:
                self._servo_1.close()
            except Exception:
                LOGGER.exception("Error while closing Servo 1 PWM channel")
        self._servo_1 = None
        if self._factory is not None:
            try:
                self._factory.close()
            except Exception:
                LOGGER.exception("Error while closing hardware PWM backend")
        self._factory = None
        self._attached = False
        self.is_initialized = False

    @staticmethod
    def _require_servo_1(servo_number: int) -> None:
        if servo_number != 1:
            raise RuntimeError("Only Servo 1 is supported")

    def _require_initialized(self) -> None:
        if not self.is_initialized:
            raise RuntimeError("Servo 1 has not been initialized")
