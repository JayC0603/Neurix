#!/usr/bin/env python3
"""Mock regression test: LCD shows the result only after tapping has finished."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from inference.result import ClassificationResult  # noqa: E402
from services.capture_classify_service import CaptureClassifyService  # noqa: E402


class FakeLCD:
    def __init__(self, servo: FakeServo | None = None) -> None:
        self.events: list[tuple[str, str, float | str]] = []
        self.servo = servo

    def display_message(self, line1: str, line2: str) -> None:
        self.events.append(("message", line1, line2))

    def display_result(self, class_name: str, confidence: float) -> None:
        if self.servo is not None:
            assert self.servo.completed, "LCD result was shown before tapping finished"
        self.events.append(("result", class_name, confidence))


class FakeServo:
    def __init__(self) -> None:
        self.completed = False

    def tap_sequence(self) -> None:
        self.completed = True


class FakeCamera:
    def __init__(self) -> None:
        self.number = 0

    def capture_image(self) -> str:
        self.number += 1
        return f"/tmp/mock-durian-{self.number}.jpg"


class FakeClassifier:
    def predict(self, _path: str) -> ClassificationResult:
        return ClassificationResult(
            class_index=0,
            class_name="unripe",
            confidence=0.9,
            inference_time_ms=1.0,
            probabilities=[0.9, 0.05, 0.05],
        )


def main() -> None:
    servo = FakeServo()
    lcd = FakeLCD(servo)
    service = CaptureClassifyService(
        camera=FakeCamera(),
        classifier=FakeClassifier(),
        lcd=lcd,
        result_display_seconds=0.0,
        error_display_seconds=0.0,
        delete_image_after_inference=False,
        servo_controller=servo,
        images_per_job=3,
    )

    service.process_once()

    assert any(event[0] == "result" for event in lcd.events), lcd.events
    assert servo.completed
    assert lcd.events[-1] == (
        "message",
        "ĐH FPT Can Tho",
        "Team: Neurix",
    ), lcd.events
    print("LCD idle message mock test: PASS")


if __name__ == "__main__":
    main()
