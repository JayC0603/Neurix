#!/usr/bin/env python3
"""Worker giữ MobileNetV1 PyTorch và nhận yêu cầu ảnh qua JSON lines."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import timm
import torch
from PIL import Image, ImageOps
from timm.data import create_transform, resolve_model_data_config


CANONICAL_CLASS_NAMES = ["unripe", "ripe", "Overripe"]


def send(payload: dict[str, object]) -> None:
    """Gửi đúng một phản hồi JSON và flush ngay cho process chính."""

    print(json.dumps(payload, ensure_ascii=True), flush=True)


class MobileNetV1Classifier(torch.nn.Module):
    """Khớp cấu trúc backbone và classifier trong checkpoint."""

    def __init__(self, model_name: str, class_count: int) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=0,
        )
        self.classifier = torch.nn.Linear(
            self.backbone.num_features,
            class_count,
        )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """Trích đặc trưng MobileNetV1 rồi phân loại ba lớp."""

        return self.classifier(self.backbone(tensor))


class ELAAttention(torch.nn.Module):
    """ELA dùng depthwise Conv1D và GroupNorm theo checkpoint FairPlay."""

    def __init__(self, channels: int, kernel_size: int, groups: int) -> None:
        super().__init__()
        padding = kernel_size // 2
        # Depthwise Conv1D giữ nguyên từng kênh, GroupNorm cân bằng theo nhóm.
        self.conv_h = torch.nn.Conv1d(
            channels,
            channels,
            kernel_size,
            padding=padding,
            groups=channels,
            bias=False,
        )
        self.conv_w = torch.nn.Conv1d(
            channels,
            channels,
            kernel_size,
            padding=padding,
            groups=channels,
            bias=False,
        )
        self.norm_h = torch.nn.GroupNorm(groups, channels)
        self.norm_w = torch.nn.GroupNorm(groups, channels)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """Tạo attention độc lập theo chiều cao và chiều rộng."""
        height_context = tensor.mean(dim=3)
        width_context = tensor.mean(dim=2)
        height_attention = torch.sigmoid(
            self.norm_h(self.conv_h(height_context))
        ).unsqueeze(3)
        width_attention = torch.sigmoid(
            self.norm_w(self.conv_w(width_context))
        ).unsqueeze(2)
        return tensor * height_attention * width_attention


class ResidualELA(torch.nn.Module):
    """Cộng đặc trưng ELA vào đặc trưng gốc theo residual scale."""

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        groups: int,
        residual_scale: float,
    ) -> None:
        super().__init__()
        self.ela = ELAAttention(channels, kernel_size, groups)
        self.residual_scale = residual_scale

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor + self.residual_scale * self.ela(tensor)


class MobileNetV1ResidualELAClassifier(torch.nn.Module):
    """MobileNetV1 chèn ResidualELA trước global average pooling."""

    def __init__(
        self,
        model_name: str,
        class_count: int,
        kernel_size: int,
        groups: int,
        residual_scale: float,
        dropout_rate: float,
    ) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=0,
            global_pool="",
        )
        self.feature_module = ResidualELA(
            self.backbone.num_features,
            kernel_size,
            groups,
            residual_scale,
        )
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.classifier = torch.nn.Linear(
            self.backbone.num_features,
            class_count,
        )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """Trích feature map, áp ELA residual rồi phân loại."""
        features = self.backbone.forward_features(tensor)
        features = self.feature_module(features)
        features = features.mean(dim=(2, 3))
        return self.classifier(self.dropout(features))


def load_checkpoint(model_path: Path):
    """Nạp state_dict an toàn và dựng transform validation của timm."""

    # Metrics trong checkpoint dùng scalar NumPy; chỉ allowlist đúng các kiểu đó.
    torch.serialization.add_safe_globals(
        [
            np._core.multiarray.scalar,
            np.dtype,
            np.dtypes.Float64DType,
        ]
    )
    checkpoint = torch.load(
        model_path,
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise TypeError("Image checkpoint must contain model_state_dict")

    config = checkpoint.get("config", {})
    model_name = str(config.get("model_name", "mobilenetv1_100"))
    checkpoint_classes = [str(value) for value in checkpoint.get("class_names", [])]
    if not checkpoint_classes:
        checkpoint_classes = ["Overripe", "Ripe", "Unripe"]

    # Remap thứ tự checkpoint về đúng thứ tự class đang dùng trong toàn hệ thống.
    checkpoint_lookup = {
        class_name.strip().lower(): index
        for index, class_name in enumerate(checkpoint_classes)
    }
    canonical_to_checkpoint: list[int] = []
    for class_name in CANONICAL_CLASS_NAMES:
        checkpoint_index = checkpoint_lookup.get(class_name.lower())
        if checkpoint_index is None:
            raise ValueError(
                f"Checkpoint classes {checkpoint_classes!r} do not contain {class_name!r}"
            )
        canonical_to_checkpoint.append(checkpoint_index)

    feature_module = str(
        checkpoint.get("feature_module", config.get("feature_module", ""))
    )
    if feature_module == "ResidualELA":
        # Dựng đúng kiến trúc custom từ metadata lưu trong checkpoint.
        model = MobileNetV1ResidualELAClassifier(
            model_name=model_name,
            class_count=len(checkpoint_classes),
            kernel_size=int(checkpoint.get("ela_kernel_size", 7)),
            groups=int(checkpoint.get("ela_groups", 16)),
            residual_scale=float(checkpoint.get("residual_scale", 1.0)),
            dropout_rate=float(config.get("dropout_rate", 0.3)),
        )
    else:
        model = MobileNetV1Classifier(model_name, len(checkpoint_classes))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to("cpu")
    model.eval()

    # Dùng preprocessing validation chuẩn của timm cho đúng backbone.
    data_config = resolve_model_data_config(model.backbone)
    transform = create_transform(**data_config, is_training=False)
    input_size = tuple(int(value) for value in data_config["input_size"])
    with torch.inference_mode():
        model(torch.zeros((1, *input_size), dtype=torch.float32))

    validation_metrics = checkpoint.get("val_metrics", {})
    metadata = {
        "model_name": model_name,
        "checkpoint_classes": checkpoint_classes,
        "canonical_classes": CANONICAL_CLASS_NAMES,
        "input_size": list(input_size),
        "mean": [float(value) for value in data_config["mean"]],
        "std": [float(value) for value in data_config["std"]],
        "epoch": int(checkpoint.get("epoch", -1)),
        "feature_module": feature_module or "none",
        "residual_scale": float(checkpoint.get("residual_scale", 0.0)),
        # Macro-F1 là metric chính vì ba lớp trong dataset bị mất cân bằng.
        "val_macro_f1": float(checkpoint.get("val_macro_f1", 0.0)),
        "val_accuracy": float(validation_metrics.get("accuracy", 0.0)),
        "val_balanced_accuracy": float(
            validation_metrics.get("balanced_accuracy", 0.0)
        ),
    }
    return model, transform, canonical_to_checkpoint, metadata


def preprocess(image_path: Path, transform) -> torch.Tensor:
    """Đọc ảnh, sửa EXIF orientation, đổi RGB và áp dụng transform validation."""

    with Image.open(image_path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        tensor = transform(image)
    return tensor.unsqueeze(0).to("cpu")


def predict(
    model: torch.nn.Module,
    transform,
    canonical_to_checkpoint: list[int],
    image_path: Path,
) -> dict[str, object]:
    """Chạy inference và trả xác suất theo thứ tự class chuẩn của app."""

    if not image_path.is_file():
        raise FileNotFoundError(f"Image file does not exist: {image_path}")
    tensor = preprocess(image_path, transform)
    started = time.perf_counter()
    with torch.inference_mode():
        logits = model(tensor).reshape(-1)
        checkpoint_probabilities = torch.softmax(logits, dim=0)
    inference_time_ms = (time.perf_counter() - started) * 1000.0

    canonical_probabilities = checkpoint_probabilities[
        canonical_to_checkpoint
    ].detach().cpu().numpy()
    class_index = int(np.argmax(canonical_probabilities))
    return {
        "status": "ok",
        "class_index": class_index,
        "confidence": float(canonical_probabilities[class_index]),
        "probabilities": [float(value) for value in canonical_probabilities],
        "inference_time_ms": inference_time_ms,
    }


def run(model_path: Path) -> int:
    """Nạp model một lần rồi xử lý tuần tự yêu cầu từ process chính."""

    try:
        model, transform, mapping, metadata = load_checkpoint(model_path)
    except Exception as exc:
        send({"status": "error", "error": f"Cannot load image model: {exc}"})
        return 1

    send(
        {
            "status": "ready",
            "torch_version": torch.__version__,
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
            send(predict(model, transform, mapping, Path(request["image_path"])))
        except Exception as exc:
            send({"status": "error", "error": str(exc)})
    return 0


def main() -> int:
    """Đọc đường dẫn checkpoint và chạy worker."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()
    return run(args.model)


if __name__ == "__main__":
    raise SystemExit(main())
