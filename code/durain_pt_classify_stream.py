import argparse
import re
import signal
import threading
import time
from pathlib import Path

import cv2
from flask import Flask, Response, render_template_string
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FPS_SMOOTHING_ALPHA = 0.15

HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Durain Classification Stream</title>
  <style>
    body {
      margin: 0;
      background: #101418;
      color: #f5f7fb;
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    main {
      width: min(960px, calc(100vw - 24px));
      margin: 16px auto;
    }
    header {
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 12px;
    }
    h1 {
      font-size: 22px;
      margin: 0;
    }
    .meta {
      color: #b8c0cc;
      font-size: 14px;
    }
    img {
      width: 100%;
      max-height: calc(100vh - 96px);
      object-fit: contain;
      background: #05070a;
      border: 1px solid #2d3744;
      border-radius: 8px;
    }
  </style>
</head>
<body>
  <main>
    <header>
      <h1>Durain Classification</h1>
      <div class="meta">{{ camera }} | {{ model }}</div>
    </header>
    <img src="/video" alt="classification stream">
  </main>
</body>
</html>
"""


def open_camera(source):
    candidates = []
    if isinstance(source, str) and source.isdigit():
        candidates.append(int(source))
    candidates.append(source)

    match = re.fullmatch(r"/dev/video(\d+)", str(source))
    if match:
        candidates.append(int(match.group(1)))

    seen = set()
    for candidate in candidates:
        key = (type(candidate), candidate)
        if key in seen:
            continue
        seen.add(key)

        for backend in (cv2.CAP_V4L2, cv2.CAP_ANY):
            camera = cv2.VideoCapture(candidate, backend)
            if camera.isOpened():
                return camera
            camera.release()

    return cv2.VideoCapture()


class DurainClassifyStream:
    def __init__(self, args):
        model_path = Path(args.model)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        self.model = YOLO(str(model_path))
        self.image_size = args.imgsz
        self.jpeg_quality = args.jpeg_quality
        self.process_every = max(1, args.process_every)
        self.frame_index = 0
        self.lock = threading.Lock()
        self.smoothed_fps = 0.0
        self.last_label = "warming up"
        self.last_conf = 0.0
        self.last_infer_ms = 0.0
        self.last_top3 = []

        self.camera = open_camera(args.camera)
        if not self.camera.isOpened():
            raise RuntimeError(f"Cannot open camera: {args.camera}")

        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        self.camera.set(cv2.CAP_PROP_FPS, args.fps)
        self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def close(self):
        with self.lock:
            if self.camera.isOpened():
                self.camera.release()

    def read_annotated_jpeg(self):
        frame_started = time.perf_counter()

        with self.lock:
            ok, frame = self.camera.read()

        if not ok:
            return None

        if self.frame_index % self.process_every == 0:
            started = time.perf_counter()
            result = self.model.predict(frame, imgsz=self.image_size, verbose=False)[0]
            self.last_infer_ms = (time.perf_counter() - started) * 1000

            probs = result.probs
            self.last_label = result.names[probs.top1]
            self.last_conf = float(probs.top1conf)
            self.last_top3 = [
                (result.names[index], float(probs.data[index]))
                for index in probs.top5[:3]
            ]

        self.frame_index += 1
        self._draw_overlay(frame, frame_started)

        ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
        if not ok:
            return None
        return buffer.tobytes()

    def _draw_overlay(self, frame, frame_started):
        total_ms = (time.perf_counter() - frame_started) * 1000
        fps = 1000 / total_ms if total_ms > 0 else 0.0
        self.smoothed_fps = (
            fps
            if self.smoothed_fps == 0
            else (1.0 - FPS_SMOOTHING_ALPHA) * self.smoothed_fps
            + FPS_SMOOTHING_ALPHA * fps
        )

        label = f"{self.last_label} {self.last_conf:.2f}"
        status = (
            f"FPS {self.smoothed_fps:.1f} | AI {self.last_infer_ms:.0f} ms"
            f" | every {self.process_every}"
        )
        top3 = " | ".join(f"{name}:{conf:.2f}" for name, conf in self.last_top3)

        cv2.rectangle(frame, (0, 0), (frame.shape[1], 94), (0, 0, 0), -1)
        cv2.putText(frame, label, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (40, 220, 90), 2, cv2.LINE_AA)
        cv2.putText(frame, status, (12, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (230, 235, 245), 2, cv2.LINE_AA)
        cv2.putText(frame, top3, (12, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (185, 195, 210), 1, cv2.LINE_AA)


def build_app(stream, args):
    app = Flask(__name__)

    @app.route("/")
    def index():
        return render_template_string(HTML, camera=args.camera, model=Path(args.model).name)

    @app.route("/video")
    def video():
        def generate():
            while True:
                frame = stream.read_annotated_jpeg()
                if frame is None:
                    time.sleep(0.05)
                    continue
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"

        return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/health")
    def health():
        return {"ok": True}

    return app


def parse_args():
    parser = argparse.ArgumentParser(description="Stream USB camera with durain.pt classification.")
    parser.add_argument("--model", default=str(PROJECT_ROOT / "model" / "durain.pt"), help="Path to .pt model.")
    parser.add_argument("--camera", default="/dev/video1", help="Camera device or video file.")
    parser.add_argument("--host", default="0.0.0.0", help="Flask host.")
    parser.add_argument("--port", type=int, default=8001, help="Flask port.")
    parser.add_argument("--width", type=int, default=640, help="Camera capture width.")
    parser.add_argument("--height", type=int, default=480, help="Camera capture height.")
    parser.add_argument("--fps", type=int, default=15, help="Requested camera FPS.")
    parser.add_argument("--imgsz", type=int, default=320, help="Model input image size.")
    parser.add_argument("--jpeg-quality", type=int, default=70, help="Stream JPEG quality.")
    parser.add_argument("--process-every", type=int, default=3, help="Run model every N frames.")
    return parser.parse_args()


def main():
    args = parse_args()
    stream = DurainClassifyStream(args)

    def shutdown(_signum, _frame):
        stream.close()
        raise SystemExit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    app = build_app(stream, args)
    print(f"Open in browser: http://<raspberry-pi-ip>:{args.port}")
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
