#!/usr/bin/env python3
"""Worker giữ model audio PyTorch và nhận lệnh JSON qua stdin/stdout."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import librosa
import numpy as np
import torch
import torchvision  # noqa: F401 - cần để unpickle EfficientNet.
from scipy import signal


AUDIO_SAMPLE_RATE = 16_000
AUDIO_DURATION_SECONDS = 3.0
AUDIO_IMAGE_SIZE = 224
AUDIO_N_FFT = 1024
AUDIO_HOP_LENGTH = 256
AUDIO_N_MELS = 128
AUDIO_HIGHPASS_HZ = 80.0
AUDIO_SERVO_NOTCH_HZ = 4_900.0
AUDIO_SERVO_NOTCH_Q = 20.0


def send(payload: dict[str, object]) -> None:
    """Gửi một phản hồi JSON duy nhất và flush ngay."""
    print(json.dumps(payload, ensure_ascii=True), flush=True)


def load_model(model_path: Path):
    """Nạp full torchvision model trên CPU và chạy warm-up một lần."""
    model = torch.load(model_path, map_location="cpu", weights_only=False)
    if not callable(model) or not hasattr(model, "eval"):
        raise TypeError("Audio checkpoint is not a callable PyTorch model")
    model = model.to("cpu")
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, 3, AUDIO_IMAGE_SIZE, AUDIO_IMAGE_SIZE))
    return model


def filter_servo_noise(audio: np.ndarray) -> np.ndarray:
    """Giảm rung thấp và tiếng rít hẹp của servo, giữ phổ va đập chính."""
    if audio.size < 32:
        return audio

    # Rung cơ khí/nguồn nằm chủ yếu dưới 80 Hz, không mang nhiều thông tin độ chín.
    highpass = signal.butter(
        4,
        AUDIO_HIGHPASS_HZ,
        btype="highpass",
        fs=AUDIO_SAMPLE_RATE,
        output="sos",
    )
    filtered = signal.sosfiltfilt(highpass, audio)

    # Chỉ cắt rất hẹp quanh 4.9 kHz để không xóa dải cộng hưởng cú gõ 0.6–3 kHz.
    notch_b, notch_a = signal.iirnotch(
        AUDIO_SERVO_NOTCH_HZ,
        AUDIO_SERVO_NOTCH_Q,
        fs=AUDIO_SAMPLE_RATE,
    )
    filtered = signal.filtfilt(notch_b, notch_a, filtered)
    return np.asarray(filtered, dtype=np.float32)


def preprocess(audio_path: Path):
    """Chuyển WAV thành tensor Mel RGB giống pipeline lúc huấn luyện."""
    audio, _ = librosa.load(str(audio_path), sr=AUDIO_SAMPLE_RATE, mono=True)
    audio = filter_servo_noise(audio)
    target_length = int(AUDIO_SAMPLE_RATE * AUDIO_DURATION_SECONDS)
    if audio.size < target_length:
        audio = np.pad(audio, (0, target_length - audio.size), mode="constant")
    else:
        audio = audio[:target_length]

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=AUDIO_SAMPLE_RATE,
        n_fft=AUDIO_N_FFT,
        hop_length=AUDIO_HOP_LENGTH,
        n_mels=AUDIO_N_MELS,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_min = float(mel_db.min())
    mel_max = float(mel_db.max())
    mel_normalized = (mel_db - mel_min) / (mel_max - mel_min + 1e-8)
    mel_image = (mel_normalized * 255).astype(np.uint8)
    mel_image = cv2.resize(mel_image, (AUDIO_IMAGE_SIZE, AUDIO_IMAGE_SIZE))
    mel_image = cv2.cvtColor(mel_image, cv2.COLOR_GRAY2RGB)
    mel_image = mel_image.astype(np.float32) / 255.0
    mel_image = np.transpose(mel_image, (2, 0, 1))
    return torch.from_numpy(mel_image).unsqueeze(0).to("cpu")


def predict(model, audio_path: Path) -> dict[str, object]:
    """Chạy inference và trả xác suất ba lớp dưới dạng JSON."""
    tensor = preprocess(audio_path)
    started = time.perf_counter()
    with torch.no_grad():
        output = model(tensor)
    inference_time_ms = (time.perf_counter() - started) * 1000.0
    if isinstance(output, (tuple, list)):
        output = output[0]
    probabilities_tensor = torch.softmax(output.reshape(-1), dim=0)
    probabilities = probabilities_tensor.detach().cpu().numpy()
    class_index = int(np.argmax(probabilities))
    return {
        "status": "ok",
        "class_index": class_index,
        "confidence": float(probabilities[class_index]),
        "probabilities": [float(value) for value in probabilities],
        "inference_time_ms": inference_time_ms,
    }


def run(model_path: Path) -> int:
    """Nạp model một lần rồi xử lý tuần tự các yêu cầu từ process chính."""
    try:
        model = load_model(model_path)
    except Exception as exc:
        send({"status": "error", "error": f"Cannot load audio model: {exc}"})
        return 1

    send(
        {
            "status": "ready",
            "audio_filter": {
                "highpass_hz": AUDIO_HIGHPASS_HZ,
                "servo_notch_hz": AUDIO_SERVO_NOTCH_HZ,
                "servo_notch_q": AUDIO_SERVO_NOTCH_Q,
            },
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
            send(predict(model, Path(request["audio_path"])))
        except Exception as exc:
            send({"status": "error", "error": str(exc)})
    return 0


def main() -> int:
    """Đọc tham số đường dẫn model và chạy worker."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()
    return run(args.model)


if __name__ == "__main__":
    raise SystemExit(main())
