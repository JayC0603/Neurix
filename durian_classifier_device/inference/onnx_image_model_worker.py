#!/usr/bin/env python3
"""Persistent ONNX Runtime worker for the MobileNetV1 image model."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np


def _import_onnxruntime_without_device_probe_noise():
    """Import ORT without its harmless Raspberry Pi DRM probe warnings.

    ONNX Runtime 1.27 probes DRM cards while its native module is imported,
    even when this worker later selects CPUExecutionProvider explicitly. The
    Pi exposes card entries without the PCI ``device/vendor`` files expected
    by that probe. Temporarily silence native stderr only for the import; the
    descriptor is restored before model loading, inference, and error output.
    """

    stderr_fd = sys.stderr.fileno()
    saved_stderr_fd = os.dup(stderr_fd)
    null_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(null_fd, stderr_fd)
        import onnxruntime  # type: ignore[import-not-found]
    finally:
        os.dup2(saved_stderr_fd, stderr_fd)
        os.close(null_fd)
        os.close(saved_stderr_fd)
    return onnxruntime


ort = _import_onnxruntime_without_device_probe_noise()
# Keep later native warnings out of the production console. Model-load and
# inference failures still travel through the worker's structured JSON errors.
ort.set_default_logger_severity(3)


CLASS_NAMES = ["unripe", "ripe", "Overripe"]


def send(payload: dict[str, object]) -> None:
    print(json.dumps(payload, ensure_ascii=True), flush=True)


def _fixed_dimension(value: object, label: str) -> int:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"ONNX {label} must be a fixed positive integer, got {value!r}")
    return value


def load_session(model_path: Path):
    if not model_path.is_file():
        raise FileNotFoundError(f"ONNX image model not found: {model_path}")
    session = ort.InferenceSession(
        str(model_path),
        providers=["CPUExecutionProvider"],
    )
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if len(inputs) != 1 or len(outputs) != 1:
        raise ValueError(
            f"Expected one ONNX input/output, got {len(inputs)}/{len(outputs)}"
        )
    shape = inputs[0].shape
    if len(shape) != 4:
        raise ValueError(f"Expected NHWC image input, got {shape}")
    height = _fixed_dimension(shape[1], "input height")
    width = _fixed_dimension(shape[2], "input width")
    channels = _fixed_dimension(shape[3], "input channels")
    if channels != 3:
        raise ValueError(f"Expected 3 image channels, got {channels}")
    output_shape = outputs[0].shape
    if not output_shape or output_shape[-1] != len(CLASS_NAMES):
        raise ValueError(
            f"Expected {len(CLASS_NAMES)} output classes, got {output_shape}"
        )
    metadata = {
        "input_name": inputs[0].name,
        "input_shape": list(shape),
        "input_layout": "NHWC",
        "normalization": "minus_one_one",
        "output_name": outputs[0].name,
        "output_shape": list(output_shape),
        "classes": CLASS_NAMES,
        "providers": session.get_providers(),
    }
    return session, inputs[0].name, width, height, metadata


def preprocess(image_path: Path, width: int, height: int) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    tensor = image.astype(np.float32) / 127.5 - 1.0
    return np.expand_dims(tensor, axis=0)


def predict(session, input_name: str, width: int, height: int, path: Path):
    tensor = preprocess(path, width, height)
    started = time.perf_counter()
    values = np.asarray(session.run(None, {input_name: tensor})[0], dtype=np.float64).reshape(-1)
    inference_time_ms = (time.perf_counter() - started) * 1000.0
    if values.size != len(CLASS_NAMES) or not np.all(np.isfinite(values)):
        raise ValueError(f"Invalid ONNX output: {values.tolist()}")
    if np.any(values < 0.0) or np.any(values > 1.0) or not np.isclose(values.sum(), 1.0, atol=1e-3):
        raise ValueError(
            "ONNX image output is not the expected softmax probability vector"
        )
    values = np.clip(values, 0.0, 1.0)
    values /= values.sum()
    class_index = int(np.argmax(values))
    return {
        "status": "ok",
        "class_index": class_index,
        "confidence": float(values[class_index]),
        "probabilities": [float(value) for value in values],
        "inference_time_ms": inference_time_ms,
    }


def run(model_path: Path) -> int:
    try:
        session, input_name, width, height, metadata = load_session(model_path)
    except Exception as exc:
        send({"status": "error", "error": f"Cannot load ONNX image model: {exc}"})
        return 1
    send(
        {
            "status": "ready",
            "onnxruntime_version": ort.__version__,
            "metadata": metadata,
        }
    )
    for raw_line in sys.stdin:
        try:
            request = json.loads(raw_line)
            command = request.get("command")
            if command == "close":
                return 0
            if command != "predict":
                raise ValueError(f"Unsupported command: {command}")
            send(
                predict(
                    session,
                    input_name,
                    width,
                    height,
                    Path(request["image_path"]),
                )
            )
        except Exception as exc:
            send({"status": "error", "error": str(exc)})
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()
    return run(args.model)


if __name__ == "__main__":
    raise SystemExit(main())
