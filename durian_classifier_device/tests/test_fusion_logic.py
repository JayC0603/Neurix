#!/usr/bin/env python3
"""Kiểm tra fusion đủ ba lớp và hai nhánh model chạy song song."""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from inference.result import ClassificationResult  # noqa: E402
from services.capture_classify_service import CaptureClassifyService  # noqa: E402


def result(probabilities: list[float]) -> ClassificationResult:
    """Tạo kết quả tối giản theo thứ tự unripe, ripe, Overripe."""
    class_index = max(range(len(probabilities)), key=probabilities.__getitem__)
    return ClassificationResult(
        class_index=class_index,
        class_name=("unripe", "ripe", "Overripe")[class_index],
        confidence=probabilities[class_index],
        inference_time_ms=10.0,
        probabilities=probabilities,
    )


def make_service() -> CaptureClassifyService:
    """Tạo service tối giản chỉ để gọi trực tiếp phép fusion."""
    service = object.__new__(CaptureClassifyService)
    service.fusion_image_weight = 0.6
    service.fusion_audio_weight = 0.4
    return service


def assert_fused(
    result: ClassificationResult,
    expected_class: str,
    expected_probabilities: list[float],
) -> None:
    """Check that the actual weighted model probabilities are returned."""
    expected_index = ("unripe", "ripe", "Overripe").index(expected_class)
    assert result.class_index == expected_index
    assert result.class_name == expected_class
    assert all(
        abs(actual - expected) < 1e-9
        for actual, expected in zip(result.probabilities, expected_probabilities)
    )
    assert abs(result.confidence - expected_probabilities[expected_index]) < 1e-9
    assert abs(sum(result.probabilities) - 1.0) < 1e-9


def main() -> None:
    """Confirm the displayed output is the real image/audio fusion result."""
    service = make_service()

    # Mỗi xác suất cuối = 60% trung bình ảnh + 40% audio.
    unripe_images = [result([0.8, 0.1, 0.1]) for _ in range(3)]
    unripe_audio = result([0.1, 0.6, 0.3])
    unripe = service._fuse_results(unripe_audio, unripe_images)
    assert_fused(unripe, "unripe", [0.52, 0.30, 0.18])

    # Hai model cùng nghiêng về Overripe.
    overripe_images = [result([0.1, 0.1, 0.8]) for _ in range(3)]
    overripe_audio = result([0.3, 0.1, 0.6])
    overripe = service._fuse_results(overripe_audio, overripe_images)
    assert_fused(overripe, "Overripe", [0.18, 0.10, 0.72])

    # Fusion vẫn chọn xác suất lớn nhất, không còn ngưỡng audio nhị phân.
    boundary_audio = result([0.5, 0.1, 0.4])
    boundary = service._fuse_results(boundary_audio, overripe_images)
    assert_fused(boundary, "Overripe", [0.26, 0.10, 0.64])

    # Trường hợp quan trọng: lớp ripe phải có thể thắng fusion.
    conflicting_images = [result([0.4, 0.5, 0.1]) for _ in range(3)]
    below_gate_audio = result([0.1, 0.51, 0.39])
    ripe = service._fuse_results(below_gate_audio, conflicting_images)
    assert_fused(ripe, "ripe", [0.28, 0.504, 0.216])

    # Không có audio thì dùng nguyên vector ảnh, không ép về unripe.
    ripe_images = [result([0.1, 0.8, 0.1]) for _ in range(3)]
    image_only = service._fuse_results(None, ripe_images)
    assert_fused(image_only, "ripe", [0.1, 0.8, 0.1])

    stored_audio: dict[str, object] = {}
    service.audio_classifier = SimpleNamespace(predict=lambda _path: image_only)
    service.result_store = SimpleNamespace(
        set_audio=lambda path, result=None: stored_audio.update(
            path=path, result=result
        )
    )
    service._record_audio_during_servo_strike = lambda: Path("/tmp/demo.wav")
    audio_result = service._run_audio_pipeline()
    assert audio_result is not None
    assert audio_result == image_only
    assert stored_audio["result"] == audio_result

    # Confidence thấp vẫn chỉ chạy một lượt: ba ảnh và một lần nhận diện audio.
    counters = {"captures": 0, "audio_records": 0, "audio_predictions": 0}
    first_capture_started = threading.Event()
    capture_times: list[float] = []

    def capture_image() -> str:
        counters["captures"] += 1
        capture_times.append(time.monotonic())
        first_capture_started.set()
        return f"/tmp/fusion_image_{counters['captures']}.jpg"

    def record_audio() -> Path:
        counters["audio_records"] += 1
        return Path("/tmp/fusion_audio.wav")

    def predict_audio(_path: Path) -> ClassificationResult:
        # Nếu hai nhánh không chạy song song, chờ này sẽ timeout.
        assert first_capture_started.wait(timeout=1.0)
        counters["audio_predictions"] += 1
        return result([0.34, 0.33, 0.33])

    retry_service = CaptureClassifyService(
        camera=SimpleNamespace(capture_image=capture_image),
        classifier=SimpleNamespace(
            predict=lambda _path: result([0.34, 0.33, 0.33]),
        ),
        lcd=SimpleNamespace(
            display_message=lambda *_args: None,
            display_result=lambda *_args: None,
        ),
        result_display_seconds=0.0,
        error_display_seconds=0.0,
        delete_image_after_inference=False,
        audio_classifier=SimpleNamespace(predict=predict_audio),
        images_per_job=3,
        image_interval_seconds=0.01,
        fusion_image_weight=0.6,
        fusion_audio_weight=0.4,
    )
    retry_service._record_audio_during_servo_strike = record_audio
    retry_service._shutdown_event = threading.Event()
    retry_service.process_once()
    assert counters == {
        "captures": 3,
        "audio_records": 1,
        "audio_predictions": 1,
    }
    assert all(
        later - earlier >= 0.008
        for earlier, later in zip(capture_times, capture_times[1:])
    )
    print("Fusion 3-class logic test: PASS")


if __name__ == "__main__":
    main()
