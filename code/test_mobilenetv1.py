import argparse
import os
import time
from pathlib import Path

import numpy as np
import cv2
import tensorflow as tf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "model" / "mobilenetv1_best.keras"
DEFAULT_INPUT_PATH = (
    PROJECT_ROOT / "video" / "durian_181_Overmature_C_SA_Overripe_rgb.jpg"
)

IMG_SIZE = 224


# =========================
# 1. Cấu hình class
# =========================
# Phải đúng thứ tự class lúc train model
CLASS_NAMES = [
    "unripe",
    "ripe",
    "Overripe"
]


# =========================
# 2. Load ảnh
# =========================
def load_image(image_path, input_size):
    img_bgr = cv2.imread(str(image_path))

    if img_bgr is None:
        raise ValueError(f"Không đọc được ảnh: {image_path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, input_size)

    # Nếu lúc train dùng rescale=1./255 thì giữ dòng này
    img_array = img_resized.astype(np.float32) / 255.0

    img_array = np.expand_dims(img_array, axis=0)

    return img_array


# =========================
# 3. Dự đoán 1 ảnh
# =========================
def predict_image(model, image_path, input_size):
    img_array = load_image(image_path, input_size)

    # Warm-up để lần đầu không bị tính chậm bất thường
    _ = model.predict(img_array, verbose=0)

    start_time = time.perf_counter()
    preds = model.predict(img_array, verbose=0)
    end_time = time.perf_counter()

    inference_time = end_time - start_time
    fps = 1.0 / inference_time if inference_time > 0 else 0

    preds = preds[0]

    class_id = int(np.argmax(preds))
    confidence = float(preds[class_id])

    class_name = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"class_{class_id}"

    return {
        "image": image_path,
        "class_id": class_id,
        "class_name": class_name,
        "confidence": confidence,
        "inference_time": inference_time,
        "fps": fps,
        "raw_output": preds
    }


# =========================
# 4. Lấy danh sách ảnh
# =========================
def get_image_paths(path):
    valid_exts = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

    if os.path.isfile(path):
        return [path]

    if os.path.isdir(path):
        image_paths = []

        for file in os.listdir(path):
            ext = os.path.splitext(file)[1].lower()

            if ext in valid_exts:
                image_paths.append(os.path.join(path, file))

        return sorted(image_paths)

    raise ValueError(f"Đường dẫn không tồn tại: {path}")


# =========================
# 5. Main
# =========================
def main():
    parser = argparse.ArgumentParser(description="Test MobileNetV1 on an image or folder")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    args = parser.parse_args()

    input_size = (IMG_SIZE, IMG_SIZE)

    print("Đang load model...")
    print(f"Model path: {args.model}")
    print(f"Input path: {args.input}")

    model = tf.keras.models.load_model(str(args.model))

    print("Load model thành công.")
    print("Input shape:", model.input_shape)
    print("Output shape:", model.output_shape)
    print("-" * 60)

    image_paths = get_image_paths(args.input)

    if len(image_paths) == 0:
        print("Không tìm thấy ảnh nào.")
        return

    total_time = 0
    success_count = 0

    for image_path in image_paths:
        try:
            result = predict_image(model, image_path, input_size)

            total_time += result["inference_time"]
            success_count += 1

            print(f"Ảnh: {os.path.basename(result['image'])}")
            print(f"Class ID: {result['class_id']}")
            print(f"Class name: {result['class_name']}")
            print(f"Confidence: {result['confidence']:.4f}")
            print(f"Inference time: {result['inference_time'] * 1000:.2f} ms")
            print(f"Speed: {result['fps']:.2f} FPS")
            print(f"Raw output: {result['raw_output']}")
            print("-" * 60)

        except Exception as e:
            print(f"Lỗi khi xử lý ảnh {image_path}: {e}")

    if success_count > 0:
        avg_time = total_time / success_count
        avg_fps = 1.0 / avg_time if avg_time > 0 else 0

        print("TỔNG KẾT")
        print(f"Số ảnh test thành công: {success_count}/{len(image_paths)}")
        print(f"Thời gian trung bình: {avg_time * 1000:.2f} ms/ảnh")
        print(f"Tốc độ trung bình: {avg_fps:.2f} FPS")


if __name__ == "__main__":
    main()
