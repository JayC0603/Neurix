"""Giao tiếp với worker MobileNetV1 PyTorch tách khỏi venv TensorFlow."""

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


class TorchImageClassifier:
    """Cung cấp interface giống classifier Keras cũ qua một worker PyTorch."""

    def __init__(
        self,
        model_path: Path,
        python_executable: Path,
        confidence_threshold: float,
    ) -> None:
        self.model_path = Path(model_path)
        self.python_executable = Path(python_executable)
        self.confidence_threshold = confidence_threshold
        self._process: subprocess.Popen[str] | None = None
        self._lock = threading.Lock()
        self.framework_version = "unknown"
        self.metadata: dict[str, Any] = {}

    def initialize(self) -> None:
        """Khởi động worker và chờ checkpoint MobileNetV1 sẵn sàng."""

        if self._process is not None and self._process.poll() is None:
            return
        if not self.model_path.is_file():
            raise ModelConfigurationError(
                f"Image model file does not exist: {self.model_path}"
            )
        if not self.python_executable.is_file():
            raise ModelConfigurationError(
                f"Image Python executable does not exist: {self.python_executable}"
            )

        worker_path = Path(__file__).with_name("torch_image_model_worker.py")
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
                # Worker không nhận Ctrl+C trực tiếp; app đóng worker qua pipe.
                start_new_session=True,
            )
            self._process = process
            response = self._read_response()
        except Exception as exc:
            self.close()
            raise ModelConfigurationError(
                f"Cannot start PyTorch image worker: {exc}"
            ) from exc

        if response.get("status") != "ready":
            self.close()
            raise ModelConfigurationError(
                f"PyTorch image worker failed: {response.get('error', response)}"
            )
        self.framework_version = str(response.get("torch_version", "unknown"))
        self.metadata = dict(response.get("metadata", {}))
        LOGGER.info(
            "PyTorch image worker ready: model=%s python=%s torch=%s metadata=%s",
            self.model_path,
            self.python_executable,
            self.framework_version,
            self.metadata,
        )
        LOGGER.info(
            "Primary validation metric: Macro-F1=%.2f%% "
            "(Accuracy=%.2f%%, Balanced Accuracy=%.2f%%)",
            float(self.metadata.get("val_macro_f1", 0.0)) * 100.0,
            float(self.metadata.get("val_accuracy", 0.0)) * 100.0,
            float(self.metadata.get("val_balanced_accuracy", 0.0)) * 100.0,
        )

    def _read_response(self) -> dict[str, Any]:
        """Đọc đúng một JSON response và phát hiện worker dừng bất thường."""

        process = self._process
        if process is None or process.stdout is None:
            raise RuntimeError("PyTorch image worker is not running")
        line = process.stdout.readline()
        if not line:
            raise RuntimeError(
                "PyTorch image worker stopped unexpectedly; "
                f"exit_code={process.poll()}"
            )
        try:
            return json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid image worker response: {line!r}") from exc

    def predict(self, image_path: str | Path) -> ClassificationResult:
        """Gửi một ảnh tới worker và nhận xác suất theo class chuẩn của app."""

        path = Path(image_path)
        if not path.is_file():
            raise ModelInferenceError(f"Image file does not exist: {path}")

        with self._lock:
            process = self._process
            if process is None or process.poll() is not None or process.stdin is None:
                raise ModelInferenceError("PyTorch image classifier is not initialized")
            try:
                process.stdin.write(
                    json.dumps({"command": "predict", "image_path": str(path)})
                    + "\n"
                )
                process.stdin.flush()
                response = self._read_response()
            except Exception as exc:
                raise ModelInferenceError(
                    f"PyTorch image worker communication failed: {exc}"
                ) from exc

        if response.get("status") != "ok":
            raise ModelInferenceError(
                f"PyTorch image prediction failed: {response.get('error', response)}"
            )
        probabilities = [float(value) for value in response["probabilities"]]
        if len(probabilities) != len(CLASS_NAMES):
            raise ModelInferenceError(
                f"PyTorch model returned {len(probabilities)} classes; "
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
            "PyTorch image result index=%s class=%s prediction_confidence=%.2f%% "
            "probabilities_pct=%s time_ms=%.2f",
            result.class_index,
            result.class_name,
            result.confidence * 100.0,
            {
                name: round(probability * 100.0, 2)
                for name, probability in zip(CLASS_NAMES, probabilities)
            },
            result.inference_time_ms,
        )
        return result

    def close(self) -> None:
        """Dừng worker và đóng toàn bộ pipe."""

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
            LOGGER.debug("PyTorch image worker graceful shutdown failed", exc_info=True)
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
        LOGGER.info("PyTorch image worker closed")
