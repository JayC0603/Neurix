"""Thread-safe latest-job state shared by the classifier and web dashboard."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

from inference.classifier import CLASS_NAMES
from inference.result import ClassificationResult


def _result_payload(result: ClassificationResult | None) -> dict[str, Any] | None:
    if result is None:
        return None
    return {
        "class_name": result.class_name,
        "confidence": round(float(result.confidence), 6),
        "inference_time_ms": round(float(result.inference_time_ms), 2),
        "probabilities": {
            name: round(float(probability), 6)
            for name, probability in zip(CLASS_NAMES, result.probabilities)
        },
    }


class LatestResultStore:
    """Keep the latest job in memory without exposing arbitrary file paths."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._job_sequence = 0
        self._state: dict[str, Any] = self._initial_state()
        self._image_paths: list[Path] = []
        self._audio_path: Path | None = None

    @staticmethod
    def _initial_state() -> dict[str, Any]:
        now = time.time()
        return {
            "job_id": None,
            "status": "idle",
            "message": "Sẵn sàng kiểm tra",
            "started_at": None,
            "updated_at": now,
            "duration_ms": None,
            "images": [],
            "audio": None,
            "final_result": None,
        }

    def begin_job(self) -> int:
        with self._lock:
            self._job_sequence += 1
            now = time.time()
            self._state = {
                "job_id": self._job_sequence,
                "status": "processing",
                "message": "Đang bắt đầu kiểm tra",
                "started_at": now,
                "updated_at": now,
                "duration_ms": None,
                "images": [],
                "audio": None,
                "final_result": None,
            }
            self._image_paths = []
            self._audio_path = None
            return self._job_sequence

    def update_status(self, status: str, message: str) -> None:
        with self._lock:
            self._state["status"] = status
            self._state["message"] = message
            self._state["updated_at"] = time.time()

    def add_image(
        self,
        path: str | Path,
        *,
        present: bool | None,
        presence_score: float | None,
    ) -> int:
        with self._lock:
            index = len(self._image_paths)
            self._image_paths.append(Path(path))
            self._state["images"].append(
                {
                    "index": index,
                    "name": Path(path).name,
                    "url": f"/media/image/{index}",
                    "captured_at": time.time(),
                    "durian_present": present,
                    "presence_score": (
                        round(float(presence_score), 3)
                        if presence_score is not None
                        else None
                    ),
                    "result": None,
                }
            )
            self._state["updated_at"] = time.time()
            return index

    def set_image_result(self, index: int, result: ClassificationResult) -> None:
        with self._lock:
            if 0 <= index < len(self._state["images"]):
                self._state["images"][index]["result"] = _result_payload(result)
                self._state["updated_at"] = time.time()

    def set_audio(
        self,
        path: str | Path,
        result: ClassificationResult | None = None,
    ) -> None:
        with self._lock:
            self._audio_path = Path(path)
            self._state["audio"] = {
                "name": self._audio_path.name,
                "url": "/media/audio",
                "result": _result_payload(result),
            }
            self._state["updated_at"] = time.time()

    def complete(self, result: ClassificationResult, duration_ms: float) -> None:
        with self._lock:
            self._state["status"] = "completed"
            self._state["message"] = "Đã phân loại xong"
            self._state["final_result"] = _result_payload(result)
            self._state["duration_ms"] = round(float(duration_ms), 2)
            self._state["updated_at"] = time.time()

    def fail(self, status: str, message: str, duration_ms: float) -> None:
        with self._lock:
            self._state["status"] = status
            self._state["message"] = message
            self._state["duration_ms"] = round(float(duration_ms), 2)
            self._state["updated_at"] = time.time()

    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-safe copy while holding the lock briefly."""
        with self._lock:
            images = [
                {**image, "result": dict(image["result"]) if image["result"] else None}
                for image in self._state["images"]
            ]
            audio = dict(self._state["audio"]) if self._state["audio"] else None
            return {**self._state, "images": images, "audio": audio}

    def image_path(self, index: int) -> Path | None:
        with self._lock:
            if 0 <= index < len(self._image_paths):
                return self._image_paths[index]
            return None

    def audio_path(self) -> Path | None:
        with self._lock:
            return self._audio_path
