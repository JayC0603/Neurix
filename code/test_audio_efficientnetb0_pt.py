import argparse
import os
import time
from pathlib import Path

import numpy as np
import librosa
import cv2
import torch
import torch.nn.functional as F
import torchvision


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "model" / "best (3).pt"
DEFAULT_INPUT_PATH = PROJECT_ROOT / "audio" / "durian_016_unripe_iphone.wav"

IMG_SIZE = 224
SR = 16000
DURATION = 3.0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# 1. Cấu hình class
# =========================
# Phải đúng thứ tự lúc train model
CLASS_NAMES = [
    "unripe",
    "ripe",
    "Overripe"
]


# =========================
# 2. Load model PyTorch .pt
# =========================
def load_model_pt(model_path):
    print("Đang load model...")
    print(f"Model path: {model_path}")
    print(f"Device: {DEVICE}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Không tìm thấy model: {model_path}")

    # PyTorch 2.6 mặc định weights_only=True
    # File của bạn là full model nên cần weights_only=False
    model = torch.load(
        model_path,
        map_location=DEVICE,
        weights_only=False
    )

    # Trường hợp model lưu dạng checkpoint dictionary
    if isinstance(model, dict):
        print("File .pt đang là checkpoint dictionary.")
        print("Các key có trong checkpoint:")
        print(model.keys())

        raise ValueError(
            "File .pt này không phải full model trực tiếp.\n"
            "Cần tạo lại kiến trúc EfficientNetB0 rồi load_state_dict.\n"
            "Nhưng lỗi trước đó của bạn cho thấy khả năng cao file là full model."
        )

    model = model.to(DEVICE)
    model.eval()

    print("Load model thành công.")
    print("Model type:", type(model))
    return model


# =========================
# 3. Load audio
# =========================
def load_audio(audio_path, sr=16000, duration=3.0):
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Không tìm thấy audio: {audio_path}")

    audio, _ = librosa.load(str(audio_path), sr=sr, mono=True)

    target_length = int(sr * duration)

    if len(audio) < target_length:
        audio = np.pad(audio, (0, target_length - len(audio)), mode="constant")
    else:
        audio = audio[:target_length]

    return audio


# =========================
# 4. Audio -> Mel Spectrogram Tensor
# =========================
def audio_to_mel_tensor(audio, sr=16000, img_size=224):
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=1024,
        hop_length=256,
        n_mels=128
    )

    mel_db = librosa.power_to_db(mel, ref=np.max)

    mel_min = mel_db.min()
    mel_max = mel_db.max()

    mel_norm = (mel_db - mel_min) / (mel_max - mel_min + 1e-8)
    mel_img = (mel_norm * 255).astype(np.uint8)

    mel_img = cv2.resize(mel_img, (img_size, img_size))

    # 1 channel -> 3 channel
    mel_img = cv2.cvtColor(mel_img, cv2.COLOR_GRAY2RGB)

    # Normalize [0, 1]
    mel_img = mel_img.astype(np.float32) / 255.0

    # Nếu lúc train dùng ImageNet normalize thì mở 3 dòng này
    # mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    # std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    # mel_img = (mel_img - mean) / std

    # HWC -> CHW
    mel_img = np.transpose(mel_img, (2, 0, 1))

    # CHW -> BCHW
    tensor = torch.tensor(mel_img, dtype=torch.float32).unsqueeze(0)

    return tensor.to(DEVICE)


# =========================
# 5. Xử lý output model
# =========================
def parse_output(output):
    # Một số model trả về tuple/list
    if isinstance(output, (tuple, list)):
        output = output[0]

    output = output.squeeze()

    # Multi-class
    if output.numel() > 1:
        probs = F.softmax(output, dim=0)

        class_id = int(torch.argmax(probs).item())
        confidence = float(probs[class_id].item())
        raw_output = probs.detach().cpu().numpy()

    # Binary
    else:
        score = float(torch.sigmoid(output).item())

        class_id = 1 if score >= 0.5 else 0
        confidence = score if class_id == 1 else 1 - score
        raw_output = np.array([score])

    class_name = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"class_{class_id}"

    return class_id, class_name, confidence, raw_output


# =========================
# 6. Dự đoán 1 audio
# =========================
def predict_audio(model, audio_path):
    total_start = time.perf_counter()

    audio = load_audio(audio_path, sr=SR, duration=DURATION)
    input_tensor = audio_to_mel_tensor(audio, sr=SR, img_size=IMG_SIZE)

    with torch.no_grad():
        # Warm-up
        _ = model(input_tensor)

        infer_start = time.perf_counter()
        output = model(input_tensor)
        infer_end = time.perf_counter()

    total_end = time.perf_counter()

    inference_time = infer_end - infer_start
    total_time = total_end - total_start
    fps = 1.0 / inference_time if inference_time > 0 else 0

    class_id, class_name, confidence, raw_output = parse_output(output)

    return {
        "audio": audio_path,
        "class_id": class_id,
        "class_name": class_name,
        "confidence": confidence,
        "inference_time": inference_time,
        "total_time": total_time,
        "fps": fps,
        "raw_output": raw_output
    }


# =========================
# 7. Lấy danh sách audio
# =========================
def get_audio_paths(path):
    valid_exts = [".wav", ".mp3", ".m4a", ".flac", ".ogg"]

    if os.path.isfile(path):
        return [path]

    if os.path.isdir(path):
        audio_paths = []

        for file in os.listdir(path):
            ext = os.path.splitext(file)[1].lower()

            if ext in valid_exts:
                audio_paths.append(os.path.join(path, file))

        return sorted(audio_paths)

    raise ValueError(f"Đường dẫn không tồn tại: {path}")


# =========================
# 8. Main
# =========================
def main():
    parser = argparse.ArgumentParser(description="Test EfficientNet-B0 audio model")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    args = parser.parse_args()

    print("=" * 60)
    print("TEST AUDIO EFFICIENTNETB0 PYTORCH")
    print("=" * 60)
    print(f"Input path: {args.input}")
    print(f"Sample rate: {SR}")
    print(f"Duration: {DURATION} giây")
    print(f"Image size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"Class names: {CLASS_NAMES}")
    print("=" * 60)

    model = load_model_pt(args.model)

    audio_paths = get_audio_paths(args.input)

    if len(audio_paths) == 0:
        print("Không tìm thấy audio nào.")
        return

    total_infer_time = 0
    total_process_time = 0
    success_count = 0

    print("-" * 60)

    for audio_path in audio_paths:
        try:
            result = predict_audio(model, audio_path)

            total_infer_time += result["inference_time"]
            total_process_time += result["total_time"]
            success_count += 1

            print(f"Audio: {os.path.basename(result['audio'])}")
            print(f"Class ID: {result['class_id']}")
            print(f"Class name: {result['class_name']}")
            print(f"Confidence: {result['confidence']:.4f}")
            print(f"Inference time: {result['inference_time'] * 1000:.2f} ms")
            print(f"Total processing time: {result['total_time'] * 1000:.2f} ms")
            print(f"Speed: {result['fps']:.2f} FPS")
            print(f"Raw output: {result['raw_output']}")
            print("-" * 60)

        except Exception as e:
            print(f"Lỗi khi xử lý audio {audio_path}: {e}")
            print("-" * 60)

    if success_count > 0:
        avg_infer = total_infer_time / success_count
        avg_total = total_process_time / success_count
        avg_fps = 1.0 / avg_infer if avg_infer > 0 else 0

        print("TỔNG KẾT")
        print(f"Số audio test thành công: {success_count}/{len(audio_paths)}")
        print(f"Inference trung bình: {avg_infer * 1000:.2f} ms/file")
        print(f"Xử lý tổng trung bình: {avg_total * 1000:.2f} ms/file")
        print(f"Tốc độ model trung bình: {avg_fps:.2f} FPS")
    else:
        print("Không có audio nào test thành công.")


if __name__ == "__main__":
    main()
