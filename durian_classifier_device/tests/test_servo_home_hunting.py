#!/usr/bin/env python3
"""So sánh Servo 1 tại HOME khi giữ PWM và khi detach, không thực hiện strike."""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from config import Settings  # noqa: E402
from hardware.servo_controller import HOME_ANGLE, ServoController  # noqa: E402


HOLD_OBSERVE_TIME = 15.0
DETACHED_OBSERVE_TIME = 10.0


def _read_observation(prompt: str) -> str:
    allowed = {"YEN", "RUNG-CHU-DONG", "LAC-THU-DONG", "YEN-NGHIENG"}
    while True:
        answer = input(prompt).strip().upper()
        if answer in allowed:
            return answer
        print("Nhập YEN, RUNG-CHU-DONG, LAC-THU-DONG hoặc YEN-NGHIENG.")


def main() -> int:
    settings = Settings.load()
    if settings.hardware_mock:
        raise SystemExit("HARDWARE_MOCK must be false for this physical servo test")

    print("WARNING: test chỉ Servo 1 tại HOME 90°, không thực hiện strike.")
    print("Dừng durian-classifier.service và giữ tay tránh cơ cấu trước khi chạy.")
    if input("Type HOME-TEST to continue: ").strip().upper() != "HOME-TEST":
        raise SystemExit("HOME hunting test cancelled")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    controller = ServoController(
        mock=False,
        move_to_strike_time=settings.servo_move_to_strike_time,
        strike_hold_time=settings.servo_strike_hold_time,
        return_home_time=settings.servo_return_home_time,
        home_settle_time=settings.servo_home_settle_time,
    )
    try:
        controller.init_servos()
        controller.move_to_home(1, force=False, purpose="HOME HOLD TEST")
        logging.info(
            "Servo 1 commanded HOME %.0f with PWM attached; observe for %.0fs",
            HOME_ANGLE,
            HOLD_OBSERVE_TIME,
        )
        time.sleep(HOLD_OBSERVE_TIME)
        held_result = _read_observation(
            "PWM đang giữ HOME [YEN/RUNG-CHU-DONG/LAC-THU-DONG/YEN-NGHIENG]: "
        )

        controller.release_servo(1)
        logging.info(
            "Servo 1 PWM detached; observe vibration/position for %.0fs",
            DETACHED_OBSERVE_TIME,
        )
        time.sleep(DETACHED_OBSERVE_TIME)
        detached_result = _read_observation(
            "PWM đã detach [YEN/RUNG-CHU-DONG/LAC-THU-DONG/YEN-NGHIENG]: "
        )

        logging.info(
            "HOME hunting diagnostic result: pwm_attached=%s pwm_detached=%s",
            held_result,
            detached_result,
        )
        print(
            "RESULT "
            f"PWM_ATTACHED={held_result} PWM_DETACHED={detached_result}"
        )
        if held_result == "RUNG-CHU-DONG" and detached_result != "RUNG-CHU-DONG":
            print("Kết luận: rung là hunting khi servo giữ torque tại HOME.")
        elif detached_result == "RUNG-CHU-DONG":
            print(
                "Kết luận: servo vẫn tự phát lực khi không còn pulse; kiểm tra "
                "nhiễu dây signal, nguồn/GND hoặc mạch điện bên trong servo."
            )
        elif detached_result == "LAC-THU-DONG":
            print(
                "Kết luận: PWM đã mất và servo không còn giữ lực; dao động còn "
                "lại thuộc tải/cơ cấu, độ rơ hoặc trọng lực."
            )
        if detached_result == "YEN-NGHIENG":
            print(
                "Thanh bị nghiêng khi detach; cần điểm tựa/lò xo/đối trọng tại HOME."
            )
        return 0
    except KeyboardInterrupt:
        logging.warning("Ctrl+C received; cleanup will command HOME before detach")
        return 130
    finally:
        controller.cleanup_servos()


if __name__ == "__main__":
    raise SystemExit(main())
