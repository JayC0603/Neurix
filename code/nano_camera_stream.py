import time
import cv2
from flask import Flask, Response
from ultralytics import YOLO

# =========================
# Config
# =========================
CAMERA_SOURCE = "/dev/video0"
MODEL_PATH = "yolov8n.pt"
IMG_SIZE = 320
CONF_THRES = 0.25
STREAM_WIDTH = 640
STREAM_HEIGHT = 480

app = Flask(__name__)

print("Loading YOLO model...")
model = YOLO(MODEL_PATH)
print("YOLO model loaded.")

camera = cv2.VideoCapture(CAMERA_SOURCE)

if not camera.isOpened():
    raise RuntimeError(f"Không mở được camera: {CAMERA_SOURCE}")

# Giảm độ phân giải camera để Raspberry Pi xử lý nhẹ hơn
camera.set(cv2.CAP_PROP_FRAME_WIDTH, STREAM_WIDTH)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, STREAM_HEIGHT)
camera.set(cv2.CAP_PROP_FPS, 15)


def generate_frames():
    prev_time = time.time()

    while True:
        success, frame = camera.read()

        if not success:
            print("Không đọc được frame từ camera.")
            break

        frame = cv2.resize(frame, (STREAM_WIDTH, STREAM_HEIGHT))

        # YOLO inference
        results = model.predict(
            frame,
            imgsz=IMG_SIZE,
            conf=CONF_THRES,
            verbose=False
        )

        # Vẽ bounding box lên frame
        annotated_frame = results[0].plot()

        # Tính FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time

        cv2.putText(
            annotated_frame,
            f"FPS: {fps:.2f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        # Encode frame thành JPEG để stream qua web
        ret, buffer = cv2.imencode(".jpg", annotated_frame)

        if not ret:
            continue

        frame_bytes = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )


@app.route("/")
def index():
    return """
    <html>
    <head>
        <title>Raspberry Pi YOLO AI Stream</title>
    </head>
    <body>
        <h2>Raspberry Pi Camera + YOLO AI</h2>
        <p>Model: YOLOv8n | Camera: /dev/video0</p>
        <img src="/video" width="640">
    </body>
    </html>
    """


@app.route("/video")
def video():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


if __name__ == "__main__":
    print("Starting web server...")
    print("Open browser: http://192.168.10.8:8000")
    app.run(host="0.0.0.0", port=8000, threaded=True)