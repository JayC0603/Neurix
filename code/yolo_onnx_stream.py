import argparse
import re
import signal
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from flask import Flask, Response, render_template_string


COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]
FPS_SMOOTHING_ALPHA = 0.15


HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Raspberry Pi YOLO ONNX Stream</title>
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
      font-weight: 700;
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
      <h1>YOLO ONNX Camera Stream</h1>
      <div class="meta">{{ camera }} | {{ model }}</div>
    </header>
    <img src="/video" alt="YOLO camera stream">
  </main>
</body>
</html>
"""


def letterbox(image, new_shape=(320, 320), color=(114, 114, 114)):
    height, width = image.shape[:2]
    scale = min(new_shape[0] / height, new_shape[1] / width)
    resized_width = int(round(width * scale))
    resized_height = int(round(height * scale))

    pad_width = new_shape[1] - resized_width
    pad_height = new_shape[0] - resized_height
    pad_left = int(round(pad_width / 2 - 0.1))
    pad_right = int(round(pad_width / 2 + 0.1))
    pad_top = int(round(pad_height / 2 - 0.1))
    pad_bottom = int(round(pad_height / 2 + 0.1))

    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    padded = cv2.copyMakeBorder(
        resized, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=color
    )
    return padded, scale, (pad_left, pad_top)


def preprocess(frame, image_size):
    padded, scale, pad = letterbox(frame, (image_size, image_size))
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    tensor = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
    return np.expand_dims(tensor, axis=0), scale, pad


def xywh_to_xyxy(boxes):
    converted = np.empty_like(boxes)
    converted[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    converted[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    converted[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    converted[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return converted


def nms(boxes, scores, iou_threshold):
    if len(boxes) == 0:
        return []

    x1, y1, x2, y2 = boxes.T
    areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        current = order[0]
        keep.append(current)

        xx1 = np.maximum(x1[current], x1[order[1:]])
        yy1 = np.maximum(y1[current], y1[order[1:]])
        xx2 = np.minimum(x2[current], x2[order[1:]])
        yy2 = np.minimum(y2[current], y2[order[1:]])

        inter_w = np.maximum(0, xx2 - xx1)
        inter_h = np.maximum(0, yy2 - yy1)
        intersection = inter_w * inter_h
        union = areas[current] + areas[order[1:]] - intersection
        iou = intersection / np.maximum(union, 1e-6)

        order = order[1:][iou <= iou_threshold]

    return keep


def postprocess(output, frame_shape, scale, pad, conf_threshold, iou_threshold):
    predictions = np.squeeze(output)

    if predictions.ndim != 2:
        return []

    if predictions.shape[0] < predictions.shape[1]:
        predictions = predictions.T

    boxes_xywh = predictions[:, :4]
    if predictions.shape[1] == 85:
        objectness = predictions[:, 4]
        class_scores = predictions[:, 5:] * objectness[:, None]
    else:
        class_scores = predictions[:, 4:]
    class_ids = np.argmax(class_scores, axis=1)
    confidences = class_scores[np.arange(class_scores.shape[0]), class_ids]

    mask = confidences >= conf_threshold
    boxes_xywh = boxes_xywh[mask]
    confidences = confidences[mask]
    class_ids = class_ids[mask]

    boxes = xywh_to_xyxy(boxes_xywh)
    pad_x, pad_y = pad
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / scale

    frame_h, frame_w = frame_shape[:2]
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, frame_w - 1)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, frame_h - 1)

    detections = []
    for class_id in np.unique(class_ids):
        indexes = np.where(class_ids == class_id)[0]
        kept = nms(boxes[indexes], confidences[indexes], iou_threshold)
        for kept_index in kept:
            source_index = indexes[kept_index]
            detections.append(
                (
                    boxes[source_index].astype(int),
                    int(class_ids[source_index]),
                    float(confidences[source_index]),
                )
            )
    return detections


def draw_detections(frame, detections):
    for box, class_id, confidence in detections:
        x1, y1, x2, y2 = box.tolist()
        name = COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else f"class {class_id}"
        label = f"{name} {confidence:.2f}"
        color = (40, 220, 90)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        label_y = max(y1, label_size[1] + 8)
        cv2.rectangle(
            frame,
            (x1, label_y - label_size[1] - baseline - 6),
            (x1 + label_size[0] + 8, label_y + baseline - 2),
            color,
            -1,
        )
        cv2.putText(
            frame,
            label,
            (x1 + 4, label_y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )


class YoloOnnxStream:
    def __init__(self, args):
        model_path = Path(args.model)
        if not model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {model_path}")

        providers = ["CPUExecutionProvider"]
        session_options = ort.SessionOptions()
        session_options.log_severity_level = 3
        self.session = ort.InferenceSession(
            str(model_path), sess_options=session_options, providers=providers
        )
        model_input = self.session.get_inputs()[0]
        self.input_name = model_input.name
        self.image_size = args.imgsz or self._detect_image_size(model_input.shape)
        self.conf = args.conf
        self.iou = args.iou
        self.jpeg_quality = args.jpeg_quality
        self.lock = threading.Lock()
        self.smoothed_fps = 0.0
        self.frame_index = 0
        self.process_every = max(1, args.process_every)
        self.last_detections = []
        self.last_infer_ms = 0.0

        self.camera = self._open_camera(args.camera)
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

    @staticmethod
    def _detect_image_size(input_shape):
        if len(input_shape) == 4 and isinstance(input_shape[2], int) and input_shape[2] == input_shape[3]:
            return input_shape[2]
        return 320

    @staticmethod
    def _open_camera(source):
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

    def read_annotated_jpeg(self):
        frame_started = time.perf_counter()

        with self.lock:
            ok, frame = self.camera.read()

        if not ok:
            return None

        should_process = self.frame_index % self.process_every == 0
        self.frame_index += 1

        if should_process:
            tensor, scale, pad = preprocess(frame, self.image_size)
            started = time.perf_counter()
            output = self.session.run(None, {self.input_name: tensor})[0]
            self.last_detections = postprocess(output, frame.shape, scale, pad, self.conf, self.iou)
            self.last_infer_ms = (time.perf_counter() - started) * 1000

        draw_detections(frame, self.last_detections)
        total_ms = (time.perf_counter() - frame_started) * 1000
        fps = 1000 / total_ms if total_ms > 0 else 0.0
        self.smoothed_fps = (
            fps
            if self.smoothed_fps == 0
            else (1.0 - FPS_SMOOTHING_ALPHA) * self.smoothed_fps
            + FPS_SMOOTHING_ALPHA * fps
        )

        cv2.putText(
            frame,
            (
                f"FPS {self.smoothed_fps:.1f} | AI {self.last_infer_ms:.0f} ms"
                f" | every {self.process_every} | {len(self.last_detections)} objects"
            ),
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (40, 220, 90),
            2,
            cv2.LINE_AA,
        )

        ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
        if not ok:
            return None
        return buffer.tobytes()


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
    parser = argparse.ArgumentParser(description="Stream USB camera with YOLO ONNX detections.")
    parser.add_argument("--model", default="model/yolov8n.onnx", help="Path to YOLO ONNX model.")
    parser.add_argument("--camera", default="/dev/video0", help="Camera device or video file.")
    parser.add_argument("--host", default="0.0.0.0", help="Flask host.")
    parser.add_argument("--port", type=int, default=8000, help="Flask port.")
    parser.add_argument("--width", type=int, default=640, help="Camera capture width.")
    parser.add_argument("--height", type=int, default=480, help="Camera capture height.")
    parser.add_argument("--fps", type=int, default=15, help="Requested camera FPS.")
    parser.add_argument("--imgsz", type=int, default=None, help="YOLO input image size.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold.")
    parser.add_argument("--jpeg-quality", type=int, default=80, help="Stream JPEG quality.")
    parser.add_argument(
        "--process-every",
        type=int,
        default=1,
        help="Run YOLO every N frames and reuse the last boxes between inference frames.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    stream = YoloOnnxStream(args)

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
