#!/usr/bin/env python3
"""Kiểm thử chuỗi ba nhịp gõ bằng mock, không truy cập GPIO."""

from __future__ import annotations

import logging
import math
import sys
import tempfile
import threading
from pathlib import Path
from unittest.mock import patch


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from hardware.servo_controller import (  # noqa: E402
    HOME_ANGLE,
    STRIKE_COUNT,
    STRIKE_ANGLE,
    STRIKE_SEQUENCE_TIME,
    ServoController,
    _HardwarePWMServo,
    angle_to_pulse_width_us,
    servo_angle_to_pulse_width_us,
)


def make_controller(move_seconds: float = 0.0) -> ServoController:
    return ServoController(
        mock=True,
        move_to_strike_time=move_seconds,
        strike_hold_time=0.0,
        return_home_time=move_seconds,
        home_settle_time=0.0,
        enabled_servos=(1,),
    )


def test_exact_three_strike_sequence() -> None:
    servos = make_controller()
    try:
        servos.init_servos()
        assert servos.angles == (None,)
        assert not servos.is_pwm_attached
        servos.clear_command_history()
        servos.tap_sequence()
        assert [
            (command.servo_number, command.angle, command.purpose)
            for command in servos.command_history
        ] == [
            (1, HOME_ANGLE, "HOME"),
            (1, STRIKE_ANGLE, "STRIKE_1"),
            (1, HOME_ANGLE, "HOME"),
            (1, STRIKE_ANGLE, "STRIKE_2"),
            (1, HOME_ANGLE, "HOME"),
            (1, STRIKE_ANGLE, "STRIKE_3"),
            (1, HOME_ANGLE, "HOME"),
        ]
        assert servos.angles == (HOME_ANGLE,)
        assert servos.is_centered()
        assert not servos.is_pwm_attached
    finally:
        servos.cleanup_servos()


def test_concurrent_sequence_is_rejected() -> None:
    servos = make_controller(move_seconds=0.03)
    servos.init_servos()
    errors: list[BaseException] = []
    first_command_sent = threading.Event()
    original_set_angle = servos.set_servo_angle

    def signal_first_command(servo: object | None, angle: float) -> float:
        result = original_set_angle(servo, angle)
        first_command_sent.set()
        return result

    servos.set_servo_angle = signal_first_command  # type: ignore[method-assign]

    def run_first() -> None:
        try:
            servos.tap_sequence()
        except BaseException as exc:  # pragma: no cover
            errors.append(exc)

    worker = threading.Thread(target=run_first)
    worker.start()
    assert first_command_sent.wait(timeout=1.0)
    try:
        servos.tap_sequence()
    except RuntimeError as exc:
        assert "already running" in str(exc)
    else:
        raise AssertionError("Concurrent Servo 1 sequence must be rejected")
    worker.join(timeout=2.0)
    assert not worker.is_alive() and not errors
    servos.cleanup_servos()


def test_interrupted_strike_recovers_home() -> None:
    servos = make_controller()
    servos.init_servos()
    original_set_angle = servos.set_servo_angle
    failed = False

    def fail_return_home(servo: object | None, angle: float) -> float:
        nonlocal failed
        if not failed and servos.angles == (STRIKE_ANGLE,) and angle == HOME_ANGLE:
            failed = True
            raise RuntimeError("injected HOME failure")
        return original_set_angle(servo, angle)

    servos.set_servo_angle = fail_return_home  # type: ignore[method-assign]
    logging.disable(logging.CRITICAL)
    try:
        try:
            servos.tap_sequence()
        except RuntimeError as exc:
            assert "injected" in str(exc)
        else:
            raise AssertionError("Injected failure must propagate")
    finally:
        logging.disable(logging.NOTSET)
    assert servos.angles == (HOME_ANGLE,)
    assert not servos.is_pwm_attached
    servos.cleanup_servos()


def test_only_servo_1_is_supported() -> None:
    try:
        ServoController(mock=True, enabled_servos=(1, 2))
    except ValueError as exc:
        assert "Only Servo 1" in str(exc)
    else:
        raise AssertionError("Servo 2 configuration must be rejected")
    try:
        servo_angle_to_pulse_width_us(2, HOME_ANGLE)
    except ValueError as exc:
        assert "Only Servo 1" in str(exc)
    else:
        raise AssertionError("Servo 2 pulse mapping must be rejected")


def test_angle_mapping_and_detach() -> None:
    assert angle_to_pulse_width_us(0.0) == 1000
    assert angle_to_pulse_width_us(60.0) == 1333
    assert angle_to_pulse_width_us(90.0) == 1500
    assert angle_to_pulse_width_us(120.0) == 1667
    assert angle_to_pulse_width_us(135.0) == 1750
    assert angle_to_pulse_width_us(150.0) == 1833
    assert angle_to_pulse_width_us(180.0) == 2000
    assert servo_angle_to_pulse_width_us(1, HOME_ANGLE) == 1500

    with tempfile.TemporaryDirectory() as temporary_directory:
        channel = Path(temporary_directory)
        (channel / "enable").write_text("0", encoding="ascii")
        (channel / "duty_cycle").write_text("0", encoding="ascii")
        (channel / "period").write_text("20000000", encoding="ascii")
        servo = _HardwarePWMServo(channel, 18, 0.001, 0.002)
        servo.angle = HOME_ANGLE
        assert (channel / "enable").read_text(encoding="ascii") == "1"
        assert (channel / "duty_cycle").read_text(encoding="ascii") == "1500000"
        servo.detach()
        assert (channel / "enable").read_text(encoding="ascii") == "0"
        assert (channel / "duty_cycle").read_text(encoding="ascii") == "0"


def test_cleanup_does_not_send_duplicate_home_after_completed_sequence() -> None:
    servos = make_controller()
    servos.init_servos()
    servos.clear_command_history()
    servos.tap_sequence()
    history_before_cleanup = servos.command_history
    servos.cleanup_servos()
    assert servos.command_history == history_before_cleanup


def test_default_sequence_timing_is_three_strikes_in_3_seconds() -> None:
    servos = ServoController(mock=True)
    servos.init_servos()
    sleeps: list[float] = []
    with patch("hardware.servo_controller.time.sleep", side_effect=sleeps.append):
        servos.tap_sequence()
    assert sum(command.angle == STRIKE_ANGLE for command in servos.command_history) == STRIKE_COUNT
    assert math.isclose(sum(sleeps), STRIKE_SEQUENCE_TIME, abs_tol=1e-9)
    servos.cleanup_servos()


def main() -> None:
    test_exact_three_strike_sequence()
    test_concurrent_sequence_is_rejected()
    test_interrupted_strike_recovers_home()
    test_only_servo_1_is_supported()
    test_angle_mapping_and_detach()
    test_cleanup_does_not_send_duplicate_home_after_completed_sequence()
    test_default_sequence_timing_is_three_strikes_in_3_seconds()
    print("Servo 1 three-strike mock test: PASS")


if __name__ == "__main__":
    main()
