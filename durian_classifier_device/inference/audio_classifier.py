"""Điều khiển worker PyTorch audio tách biệt khỏi venv TensorFlow chính."""

from __future__ import annotations

import json
import logging
import subprocess
import threading
from pathlib import Path
from typing import Any

from .classifier import CLASS_NAMES
from .exceptions import ModelConfigurationError, ModelInferenceError
from .result import ClassificationResult


LOGGER = logging.getLogger(__name__)


class TorchAudioClassifier:
    """Giao tiếp đồng bộ với một process giữ model audio trong bộ nhớ."""

    def __init__(
        self,
        model_path: Path,
        python_executable: Path,
        confidence_threshold: float = 0.0,
    ) -> None:
        self.model_path = Path(model_path)
        self.python_executable = Path(python_executable)
        self.confidence_threshold = confidence_threshold
        self._process: subprocess.Popen[str] | None = None
        # Khóa bảo đảm mỗi lần chỉ có một yêu cầu inference đi qua pipe.
        self._lock = threading.Lock()

    def initialize(self) -> None:
        """Khởi động worker và chờ model EfficientNet-B0 sẵn sàng."""
        if self._process is not None and self._process.poll() is None:
            return
        if not self.model_path.is_file():
            raise ModelConfigurationError(
                f"Audio model file does not exist: {self.model_path}"
            )
        if not self.python_executable.is_file():
            raise ModelConfigurationError(
                f"Audio Python executable does not exist: {self.python_executable}"
            )

        worker_path = Path(__file__).with_name("audio_model_worker.py")
        try:
            process = subprocess.Popen(
                [
                    str(self.python_executable),
                    str(worker_path),
                    "--model",
                    str(self.model_path),
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=None,
                text=True,
                bufsize=1,
                # Worker không nhận Ctrl+C trực tiếp; process chính sẽ đóng qua pipe.
                start_new_session=True,
            )
            self._process = process
            response = self._read_response()
        except Exception as exc:
            self.close()
            raise ModelConfigurationError(
                f"Cannot start audio model worker: {exc}"
            ) from exc

        if response.get("status") != "ready":
            self.close()
            raise ModelConfigurationError(
                f"Audio model worker failed: {response.get('error', response)}"
            )
        LOGGER.info(
            "PyTorch audio model worker ready: model=%s python=%s filter=%s",
            self.model_path,
            self.python_executable,
            response.get("audio_filter"),
        )

    def _read_response(self) -> dict[str, Any]:
        """Đọc đúng một phản hồi JSON từ worker và phát hiện worker bị dừng."""
        process = self._process
        if process is None or process.stdout is None:
            raise RuntimeError("Audio model worker is not running")
        line = process.stdout.readline()
        if not line:
            return_code = process.poll()
            raise RuntimeError(
                f"Audio model worker stopped unexpectedly; exit_code={return_code}"
            )
        try:
            return json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid audio worker response: {line!r}") from exc

    def predict(self, audio_path: Path | str) -> ClassificationResult:
        """Gửi một WAV tới worker và nhận kết quả phân loại ba lớp."""
        path = Path(audio_path)
        if not path.is_file():
            raise ModelInferenceError(f"Audio file does not exist: {path}")

        with self._lock:
            process = self._process
            if process is None or process.poll() is not None or process.stdin is None:
                raise ModelInferenceError("Audio classifier is not initialized")
            try:
                request = {"command": "predict", "audio_path": str(path)}
                process.stdin.write(json.dumps(request) + "\n")
                process.stdin.flush()
                response = self._read_response()
            except Exception as exc:
                raise ModelInferenceError(
                    f"Audio model worker communication failed: {exc}"
                ) from exc

        if response.get("status") != "ok":
            raise ModelInferenceError(
                f"Audio model prediction failed: {response.get('error', response)}"
            )
        probabilities = [float(value) for value in response["probabilities"]]
        if len(probabilities) != len(CLASS_NAMES):
            raise ModelInferenceError(
                f"Audio model returned {len(probabilities)} classes; "
                f"expected {len(CLASS_NAMES)}"
            )

        class_index = int(response["class_index"])
        confidence = float(response["confidence"])
        class_name = (
            CLASS_NAMES[class_index]
            if confidence >= self.confidence_threshold
            else "Unknown"
        )
        result = ClassificationResult(
            class_index=class_index,
            class_name=class_name,
            confidence=confidence,
            inference_time_ms=float(response["inference_time_ms"]),
            probabilities=probabilities,
        )
        LOGGER.info(
            "Audio inference result index=%s class=%s confidence=%.2f%% "
            "probabilities_pct=%s time_ms=%.2f",
            result.class_index,
            result.class_name,
            result.confidence * 100.0,
            {
                class_name: round(probability * 100.0, 2)
                for class_name, probability in zip(CLASS_NAMES, result.probabilities)
            },
            result.inference_time_ms,
        )
        return result

    def close(self) -> None:
        """Yêu cầu worker dừng và giải phóng process/pipe."""
        process = self._process
        self._process = None
        if process is None:
            return
        try:
            if process.poll() is None and process.stdin is not None:
                process.stdin.write(json.dumps({"command": "close"}) + "\n")
                process.stdin.flush()
                process.wait(timeout=5.0)
        except Exception:
            LOGGER.debug("Audio worker graceful shutdown failed", exc_info=True)
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=3.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=1.0)
        finally:
            if process.stdin is not None:
                process.stdin.close()
            if process.stdout is not None:
                process.stdout.close()
        LOGGER.info("Audio classifier worker closed")
