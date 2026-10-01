"""ONNX image classifier running in the configured model Python environment."""

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


class OnnxImageClassifier:
    """Keep one ONNX Runtime worker alive and expose the app classifier API."""

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
        if self._process is not None and self._process.poll() is None:
            return
        if not self.model_path.is_file():
            raise ModelConfigurationError(
                f"ONNX image model does not exist: {self.model_path}"
            )
        if not self.python_executable.is_file():
            raise ModelConfigurationError(
                f"Image Python executable does not exist: {self.python_executable}"
            )

        worker_path = Path(__file__).with_name("onnx_image_model_worker.py")
        try:
            self._process = subprocess.Popen(
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
                start_new_session=True,
            )
            response = self._read_response()
        except Exception as exc:
            self.close()
            raise ModelConfigurationError(
                f"Cannot start ONNX image worker: {exc}"
            ) from exc

        if response.get("status") != "ready":
            self.close()
            raise ModelConfigurationError(
                f"ONNX image worker failed: {response.get('error', response)}"
            )
        self.framework_version = str(response.get("onnxruntime_version", "unknown"))
        self.metadata = dict(response.get("metadata", {}))
        LOGGER.info(
            "ONNX image worker ready: model=%s python=%s onnxruntime=%s metadata=%s",
            self.model_path,
            self.python_executable,
            self.framework_version,
            self.metadata,
        )

    def _read_response(self) -> dict[str, Any]:
        process = self._process
        if process is None or process.stdout is None:
            raise RuntimeError("ONNX image worker is not running")
        line = process.stdout.readline()
        if not line:
            raise RuntimeError(
                "ONNX image worker stopped unexpectedly; "
                f"exit_code={process.poll()}"
            )
        try:
            return json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid ONNX image worker response: {line!r}") from exc

    def predict(self, image_path: str | Path) -> ClassificationResult:
        path = Path(image_path)
        if not path.is_file():
            raise ModelInferenceError(f"Image file does not exist: {path}")

        with self._lock:
            process = self._process
            if process is None or process.poll() is not None or process.stdin is None:
                raise ModelInferenceError("ONNX image classifier is not initialized")
            try:
                process.stdin.write(
                    json.dumps({"command": "predict", "image_path": str(path)})
                    + "\n"
                )
                process.stdin.flush()
                response = self._read_response()
            except Exception as exc:
                raise ModelInferenceError(
                    f"ONNX image worker communication failed: {exc}"
                ) from exc

        if response.get("status") != "ok":
            raise ModelInferenceError(
                f"ONNX image prediction failed: {response.get('error', response)}"
            )
        probabilities = [float(value) for value in response["probabilities"]]
        if len(probabilities) != len(CLASS_NAMES):
            raise ModelInferenceError(
                f"ONNX model returned {len(probabilities)} classes; "
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
            "ONNX image result index=%s class=%s prediction_confidence=%.2f%% "
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
            LOGGER.debug("ONNX image worker graceful shutdown failed", exc_info=True)
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
        LOGGER.info("ONNX image worker closed")
