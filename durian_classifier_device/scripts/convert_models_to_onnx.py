#!/usr/bin/env python3
"""Convert the production image and audio models to validated ONNX files."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_DIR = PROJECT_DIR.parent
DEFAULT_IMAGE_MODEL = WORKSPACE_DIR / "model" / "MobileNetV1" / "mobilenetv1_best.keras"
DEFAULT_AUDIO_MODEL = WORKSPACE_DIR / "model" / "model-audio.pt"
DEFAULT_OUTPUT_DIR = WORKSPACE_DIR / "model" / "onnx"


def metadata_path(path: Path) -> str:
    """Store repository artifacts without embedding a developer machine path."""
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(WORKSPACE_DIR.resolve()))
    except ValueError:
        return str(resolved)


def validate_onnx(path: Path) -> None:
    """Load the graph with ONNX and create a CPU inference session."""
    import onnx
    import onnxruntime as ort

    graph = onnx.load(str(path))
    onnx.checker.check_model(graph)
    ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def export_image_model(source: Path, target: Path) -> dict[str, object]:
    """Export the Keras MobileNetV1 classifier with a dynamic batch axis."""
    import onnxruntime as ort
    import tensorflow as tf

    if not source.is_file():
        raise FileNotFoundError(f"Image model not found: {source}")
    model = tf.keras.models.load_model(source)
    shape = model.input_shape
    if isinstance(shape, list):
        if len(shape) != 1:
            raise ValueError(f"Only one image input is supported, got {shape}")
        shape = shape[0]
    if len(shape) != 4 or any(value is None for value in shape[1:]):
        raise ValueError(f"Expected image input (None, H, W, C), got {shape}")

    input_shape = tuple(int(value) for value in shape[1:])
    target.parent.mkdir(parents=True, exist_ok=True)
    model.export(
        str(target),
        format="onnx",
        input_signature=[
            tf.TensorSpec((None, *input_shape), tf.float32, name="image")
        ],
        verbose=False,
    )
    validate_onnx(target)

    sample = np.zeros((1, *input_shape), dtype=np.float32)
    keras_output = np.asarray(model(sample, training=False)).reshape(-1)
    session = ort.InferenceSession(str(target), providers=["CPUExecutionProvider"])
    onnx_output = np.asarray(
        session.run(None, {session.get_inputs()[0].name: sample})[0]
    ).reshape(-1)
    max_error = float(np.max(np.abs(keras_output - onnx_output)))
    if not np.allclose(keras_output, onnx_output, rtol=1e-4, atol=1e-5):
        raise RuntimeError(f"Image ONNX output mismatch; max_error={max_error:.8f}")

    return {
        "source": metadata_path(source),
        "onnx": metadata_path(target),
        "input_name": session.get_inputs()[0].name,
        "input_shape": [None, *input_shape],
        "input_layout": "NHWC",
        "normalization": "minus_one_one",
        "output_type": "softmax_probabilities",
        "classes": ["unripe", "ripe", "Overripe"],
        "max_validation_error": max_error,
    }


def export_audio_model(source: Path, target: Path, opset: int) -> dict[str, object]:
    """Export the PyTorch audio spectrogram classifier to ONNX."""
    import onnxruntime as ort
    import torch

    if not source.is_file():
        raise FileNotFoundError(f"Audio model not found: {source}")
    model = torch.load(source, map_location="cpu", weights_only=False)
    if not callable(model) or not hasattr(model, "eval"):
        raise TypeError("Audio checkpoint is not a callable PyTorch model")
    model = model.to("cpu").eval()
    sample = torch.zeros((1, 3, 224, 224), dtype=torch.float32)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        sample,
        str(target),
        input_names=["mel_image"],
        output_names=["logits"],
        dynamic_axes={
            "mel_image": {0: "batch"},
            "logits": {0: "batch"},
        },
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )
    validate_onnx(target)

    with torch.inference_mode():
        torch_output = model(sample)
        if isinstance(torch_output, (tuple, list)):
            torch_output = torch_output[0]
        torch_output = torch_output.detach().cpu().numpy()
    session = ort.InferenceSession(str(target), providers=["CPUExecutionProvider"])
    onnx_output = session.run(None, {"mel_image": sample.numpy()})[0]
    max_error = float(np.max(np.abs(torch_output - onnx_output)))
    if not np.allclose(torch_output, onnx_output, rtol=1e-4, atol=1e-5):
        raise RuntimeError(f"Audio ONNX output mismatch; max_error={max_error:.8f}")

    return {
        "source": metadata_path(source),
        "onnx": metadata_path(target),
        "input_name": "mel_image",
        "input_shape": [None, 3, 224, 224],
        "input_layout": "NCHW",
        "preprocessing": "3-second 16kHz mono mel spectrogram",
        "output_type": "logits_apply_softmax",
        "classes": ["unripe", "ripe", "Overripe"],
        "opset": opset,
        "max_validation_error": max_error,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert production durian image/audio models to ONNX."
    )
    parser.add_argument(
        "model",
        choices=("image", "audio", "all"),
        nargs="?",
        default="all",
    )
    parser.add_argument("--image-model", type=Path, default=DEFAULT_IMAGE_MODEL)
    parser.add_argument("--audio-model", type=Path, default=DEFAULT_AUDIO_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--opset", type=int, default=17)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.opset < 13:
        raise ValueError("ONNX opset must be at least 13")

    # TensorFlow and PyTorch together can exceed Raspberry Pi memory. Run each
    # export in a fresh process so the first framework releases all of its RAM.
    if args.model == "all":
        for model_name in ("image", "audio"):
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    model_name,
                    "--image-model",
                    str(args.image_model),
                    "--audio-model",
                    str(args.audio_model),
                    "--output-dir",
                    str(args.output_dir),
                    "--opset",
                    str(args.opset),
                ],
                check=True,
            )
        print(f"All ONNX models validated in: {args.output_dir}", flush=True)
        return 0

    metadata_path = args.output_dir / "models.json"
    metadata: dict[str, object] = {}
    if metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            metadata = {}

    if args.model == "image":
        target = args.output_dir / "mobilenetv1_image.onnx"
        print(f"Exporting image model: {args.image_model} -> {target}", flush=True)
        metadata["image"] = export_image_model(args.image_model, target)
        print(f"Image ONNX validated: {target}", flush=True)

    if args.model == "audio":
        target = args.output_dir / "efficientnet_audio.onnx"
        print(f"Exporting audio model: {args.audio_model} -> {target}", flush=True)
        metadata["audio"] = export_audio_model(args.audio_model, target, args.opset)
        print(f"Audio ONNX validated: {target}", flush=True)

    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Metadata saved: {metadata_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
