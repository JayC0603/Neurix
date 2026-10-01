"""Dựng MobileNetV1 + ELA + BatchNorm để nạp checkpoint chỉ chứa weights."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_mobilenetv1_ela_weights(
    weights_path: Path,
    expected_classes: list[str],
) -> tuple[Any, dict[str, Any]]:
    """Đọc metadata, dựng đúng kiến trúc và nạp toàn bộ trọng số tốt nhất."""
    import tensorflow as tf

    config_path = weights_path.parent / "experiment_config.json"
    if not config_path.is_file():
        raise ValueError(f"Missing ELA experiment config: {config_path}")

    metadata = json.loads(config_path.read_text(encoding="utf-8"))
    configured_classes = metadata.get("classes")
    if not isinstance(configured_classes, list) or [
        str(name).lower() for name in configured_classes
    ] != [name.lower() for name in expected_classes]:
        raise ValueError(
            f"ELA class order {configured_classes!r} does not match "
            f"application classes {expected_classes!r}"
        )
    if metadata.get("backbone") != "MobileNetV1":
        raise ValueError(f"Unsupported ELA backbone: {metadata.get('backbone')!r}")

    image_size = int(metadata.get("image_size", 224))
    dropout = float(metadata.get("dropout", 0.3))
    kernel_size = int(metadata.get("ela_kernel_size", 7))
    groups = int(metadata.get("ela_groups", 16))
    residual = bool(metadata.get("ela_residual", False))

    class ELAAttention(tf.keras.layers.Layer):
        """ELA tách attention theo chiều cao/rộng bằng grouped Conv1D."""

        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            # Checkpoint được huấn luyện với feature map 1024 kênh của MobileNetV1.
            channels = 1024
            self.conv_h = tf.keras.layers.Conv1D(
                channels,
                kernel_size,
                padding="same",
                groups=groups,
                use_bias=False,
                name="conv_h",
            )
            self.conv_w = tf.keras.layers.Conv1D(
                channels,
                kernel_size,
                padding="same",
                groups=groups,
                use_bias=False,
                name="conv_w",
            )
            self.bn_h = tf.keras.layers.BatchNormalization(name="bn_h")
            self.bn_w = tf.keras.layers.BatchNormalization(name="bn_w")

        def call(self, inputs: Any, training: bool | None = None) -> Any:
            # Gom trung bình từng trục để học attention độc lập cho H và W.
            height_context = tf.reduce_mean(inputs, axis=2)
            width_context = tf.reduce_mean(inputs, axis=1)
            height_attention = tf.nn.sigmoid(
                self.bn_h(self.conv_h(height_context), training=training)
            )[:, :, None, :]
            width_attention = tf.nn.sigmoid(
                self.bn_w(self.conv_w(width_context), training=training)
            )[:, None, :, :]
            attended = inputs * height_attention * width_attention
            return inputs + attended if residual else attended

    # Giữ backbone thành một Functional layer lồng nhau để khớp cấu trúc HDF5.
    backbone = tf.keras.applications.MobileNet(
        include_top=False,
        weights=None,
        input_shape=(image_size, image_size, 3),
    )
    inputs = tf.keras.Input(shape=(image_size, image_size, 3))
    features = backbone(inputs)
    features = ELAAttention(name="ela_attention")(features)
    features = tf.keras.layers.GlobalAveragePooling2D()(features)
    features = tf.keras.layers.BatchNormalization()(features)
    features = tf.keras.layers.Dropout(dropout)(features)
    outputs = tf.keras.layers.Dense(
        len(expected_classes), activation="softmax"
    )(features)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.load_weights(weights_path)
    return model, metadata
