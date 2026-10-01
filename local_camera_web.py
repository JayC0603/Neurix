from flask import Flask, Response, render_template_string
import cv2
import time

app = Flask(__name__)

CAMERA_INDEX = 0  # /dev/video0
WIDTH = 640
HEIGHT = 480
FPS = 30

HTML = """
<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <title>Raspberry Pi Camera</title>
    <style>
        body {
            margin: 0;
            background: #111;
            color: white;
            font-family: Arial, sans-serif;
            text-align: center;
        }
        h1 {
            margin-top: 20px;
        }
        .camera-box {
            margin: 20px auto;
            width: fit-content;
            border: 4px solid #333;
            border-radius: 12px;
            overflow: hidden;
            background: black;
        }
        img {
            display: block;
            max-width: 95vw;
            height: auto;
        }
        .info {
            color: #aaa;
            margin-top: 10px;
        }
    </style>
</head>
<body>
    <h1>Raspberry Pi Camera Live</h1>
    <div class="camera-box">
        <img src="/video_feed">
    </div>
    <div class="info">
        Camera: /dev/video0 | Stream: MJPEG | Local Web
    </div>
</body>
</html>
"""


def open_camera():
    cap = cv2.VideoCapture(CAMERA_INDEX)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)

    return cap


def generate_frames():
    cap = open_camera()

    if not cap.isOpened():
        print("ERROR: Không mở được camera /dev/video0")
        return

    print("Camera opened OK")

    while True:
        success, frame = cap.read()

        if not success:
            print("Không đọc được frame, thử lại...")
            time.sleep(0.1)
            continue

        ret, buffer = cv2.imencode(".jpg", frame)

        if not ret:
            continue

        frame_bytes = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


if __name__ == "__main__":
    print("Web camera đang chạy...")
    print("Mở trên trình duyệt:")
    print("http://192.168.10.11:8000")
    app.run(host="0.0.0.0", port=8000, debug=False)

