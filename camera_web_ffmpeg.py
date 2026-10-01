#!/usr/bin/env python3
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import subprocess
import threading
import time
from urllib.parse import parse_qs, urlparse

HOST = "0.0.0.0"
PORT = 8000

CAMERA_DEVICE = "/dev/video0"

# Camera gốc: không zoom, không crop
WIDTH = 640
HEIGHT = 480
FPS = 24

ACTIVE_PROCESS = None
ACTIVE_PROCESS_LOCK = threading.Lock()

HTML = f"""
<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <title>Raspberry Pi Camera</title>
    <style>
        body {{
            margin: 0;
            background: #111;
            color: white;
            font-family: Arial, sans-serif;
            text-align: center;
        }}

        h1 {{
            margin-top: 20px;
            font-size: 28px;
        }}

        .box {{
            margin: 20px auto;
            width: min(95vw, 640px);
            max-width: 98vw;
            border: 4px solid #333;
            border-radius: 12px;
            overflow: hidden;
            background: black;
            min-height: 360px;
            display: flex;
            align-items: center;
            justify-content: center;
        }}

        img {{
            display: block;
            width: 95vw;
            max-width: 640px;
            min-height: 360px;
            height: auto;
            object-fit: contain;
        }}

        .status {{
            margin-top: 8px;
            color: #ffca28;
            font-size: 14px;
        }}

        .info {{
            color: #aaa;
            margin-top: 12px;
            font-size: 15px;
        }}

        .note {{
            color: #ccc;
            margin-top: 8px;
            font-size: 14px;
        }}

        .controls {{
            margin: 18px auto 8px;
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 8px;
            flex-wrap: wrap;
        }}

        button {{
            padding: 9px 16px;
            border: 1px solid #666;
            border-radius: 8px;
            background: #292929;
            color: white;
            cursor: pointer;
            font-size: 15px;
        }}

        button.active {{
            background: #1976d2;
            border-color: #64b5f6;
        }}
    </style>
</head>
<body>
    <h1>Raspberry Pi Camera Live</h1>

    <div class="box">
        <img id="camera" alt="Camera realtime">
    </div>
    <div id="camera-status" class="status">Đang kết nối camera...</div>

    <div class="controls">
        <span>Xoay camera:</span>
        <button data-angle="0">0°</button>
        <button data-angle="90">90°</button>
        <button data-angle="180">180°</button>
        <button data-angle="270">270°</button>
    </div>

    <div class="info">
        Device: {CAMERA_DEVICE} | Resolution: {WIDTH}x{HEIGHT} | FPS: {FPS}
    </div>

    <div class="note">
        Góc hiện tại: <span id="current-angle">90°</span>
    </div>

    <script>
        const camera = document.getElementById("camera");
        const angleLabel = document.getElementById("current-angle");
        const statusLabel = document.getElementById("camera-status");
        const buttons = document.querySelectorAll("button[data-angle]");

        function setRotation(angle) {{
            localStorage.setItem("cameraRotation", angle);
            angleLabel.textContent = angle + "°";
            buttons.forEach((button) => {{
                button.classList.toggle("active", button.dataset.angle === angle);
            }});

            camera.src = "";
            statusLabel.textContent = "Đang mở camera ở góc " + angle + "°...";
            statusLabel.style.color = "#ffca28";
            setTimeout(() => {{
                camera.src = "/stream.mjpg?rotate=" + angle + "&t=" + Date.now();
            }}, 150);
        }}

        camera.addEventListener("load", () => {{
            statusLabel.textContent = "Camera đang hoạt động";
            statusLabel.style.color = "#66bb6a";
        }});

        camera.addEventListener("error", () => {{
            statusLabel.textContent = "Không nhận được hình. Hãy tải lại trang.";
            statusLabel.style.color = "#ef5350";
        }});

        buttons.forEach((button) => {{
            button.addEventListener("click", () => setRotation(button.dataset.angle));
        }});

        setRotation(localStorage.getItem("cameraRotation") || "90");
    </script>
</body>
</html>
"""


def start_ffmpeg(rotation):
    global ACTIVE_PROCESS

    cmd = [
        "ffmpeg",
        "-loglevel", "error",

        # Input camera USB
        "-f", "v4l2",
        "-framerate", str(FPS),
        "-video_size", f"{WIDTH}x{HEIGHT}",
        "-i", CAMERA_DEVICE,
    ]

    filters = {
        90: "transpose=clock",
        180: "hflip,vflip",
        270: "transpose=cclock",
    }
    if rotation in filters:
        cmd.extend(["-vf", filters[rotation]])

    cmd.extend([
        "-q:v", "5",
        "-f", "mjpeg",
        "pipe:1",
    ])

    print("FFmpeg command:")
    print(" ".join(cmd))

    with ACTIVE_PROCESS_LOCK:
        if ACTIVE_PROCESS is not None and ACTIVE_PROCESS.poll() is None:
            ACTIVE_PROCESS.kill()
            try:
                ACTIVE_PROCESS.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass

        ACTIVE_PROCESS = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0,
        )
        return ACTIVE_PROCESS


def release_ffmpeg(process):
    global ACTIVE_PROCESS

    with ACTIVE_PROCESS_LOCK:
        if process.poll() is None:
            process.kill()
        if ACTIVE_PROCESS is process:
            ACTIVE_PROCESS = None


class CameraHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        request = urlparse(self.path)

        if request.path == "/" or request.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML.encode("utf-8"))
            return

        if request.path == "/stream.mjpg":
            try:
                rotation = int(parse_qs(request.query).get("rotate", ["90"])[0])
            except ValueError:
                rotation = 90
            if rotation not in (0, 90, 180, 270):
                rotation = 90

            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header(
                "Content-Type",
                "multipart/x-mixed-replace; boundary=frame"
            )
            self.end_headers()

            process = start_ffmpeg(rotation)
            buffer = b""

            try:
                while True:
                    chunk = process.stdout.read(4096)

                    if not chunk:
                        if process.poll() is not None:
                            break
                        time.sleep(0.05)
                        continue

                    buffer += chunk

                    start = buffer.find(b"\xff\xd8")
                    end = buffer.find(b"\xff\xd9")

                    if start != -1 and end != -1 and end > start:
                        jpg = buffer[start:end + 2]
                        buffer = buffer[end + 2:]

                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n")
                        self.wfile.write(
                            f"Content-Length: {len(jpg)}\r\n\r\n".encode()
                        )
                        self.wfile.write(jpg)
                        self.wfile.write(b"\r\n")

            except BrokenPipeError:
                print("Client disconnected")
            except ConnectionResetError:
                print("Client reset connection")
            finally:
                release_ffmpeg(process)

            return

        self.send_response(404)
        self.end_headers()
        self.wfile.write(b"404 Not Found")


if __name__ == "__main__":
    print("====================================")
    print(" Raspberry Pi Local Camera Web")
    print(" Original View Mode")
    print("====================================")
    print(f"Camera device: {CAMERA_DEVICE}")
    print(f"Resolution   : {WIDTH}x{HEIGHT}")
    print(f"FPS          : {FPS}")
    print("")
    print("Open browser:")
    print(f"http://192.168.10.11:{PORT}")
    print("")
    print("Stop: Ctrl + C")
    print("====================================")

    server = ThreadingHTTPServer((HOST, PORT), CameraHandler)
    server.serve_forever()
