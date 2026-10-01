"""Small MJPEG camera preview server sharing the app camera instance."""

from __future__ import annotations

import json
import logging
import socket
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import cv2
import numpy as np

from hardware.camera_service import CameraService
from .dashboard_html import DASHBOARD_HTML
from .result_store import LatestResultStore


LOGGER = logging.getLogger(__name__)
ASSETS_DIR = Path(__file__).resolve().parents[1] / "assets"


def _preview_display_host(bind_host: str) -> str:
    """Return a browser-reachable host instead of a wildcard bind address."""
    if bind_host not in {"", "0.0.0.0", "::"}:
        return bind_host

    # A UDP connect performs only a local routing lookup; it sends no packet.
    # It reliably selects the Wi-Fi/Ethernet address used for LAN traffic.
    try:
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            probe.connect(("192.0.2.1", 9))
            address = probe.getsockname()[0]
            if address and not address.startswith("127."):
                return address
        finally:
            probe.close()
    except OSError:
        pass

    try:
        for result in socket.getaddrinfo(
            socket.gethostname(),
            None,
            family=socket.AF_INET,
            type=socket.SOCK_STREAM,
        ):
            address = result[4][0]
            if address and not address.startswith("127."):
                return address
    except OSError:
        pass

    return "127.0.0.1"


class CameraPreviewServer:
    """Serve a browser-friendly camera preview without opening /dev/video0 twice."""

    def __init__(
        self,
        camera: CameraService,
        host: str,
        port: int,
        fps: float,
        jpeg_quality: int,
        enabled: bool = True,
        result_store: LatestResultStore | None = None,
    ) -> None:
        self.camera = camera
        self.host = host
        self.port = port
        self.fps = fps
        self.jpeg_quality = jpeg_quality
        self.enabled = enabled
        self.result_store = result_store
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._last_error_log_at = 0.0
        self._display_host: str | None = None

    @property
    def display_host(self) -> str | None:
        """LAN host suitable for showing on the LCD after the server starts."""
        return self._display_host

    def start(self) -> None:
        """Start the preview server in the background."""
        if not self.enabled:
            LOGGER.info("Camera preview disabled by configuration")
            return
        if self._server is not None:
            return

        parent = self

        class PreviewHandler(BaseHTTPRequestHandler):
            def log_message(self, fmt: str, *args: object) -> None:
                LOGGER.debug("Preview client %s - " + fmt, self.client_address[0], *args)

            def do_GET(self) -> None:
                request = urlsplit(self.path)
                path = request.path
                if path in {"/", "/index.html"}:
                    self._send_index()
                    return
                if path == "/stream.mjpg":
                    try:
                        rotation = int(
                            next(
                                (
                                    item.split("=", 1)[1]
                                    for item in request.query.split("&")
                                    if item.startswith("rotate=")
                                ),
                                "90",
                            )
                        )
                    except ValueError:
                        rotation = 90
                    if rotation not in (0, 90, 180, 270):
                        rotation = 90
                    self._send_stream(rotation)
                    return
                if path == "/api/roi":
                    self._send_json(parent.camera.get_roi())
                    return
                if path == "/api/result":
                    payload = (
                        parent.result_store.snapshot()
                        if parent.result_store is not None
                        else {"status": "idle", "message": "Sẵn sàng", "images": []}
                    )
                    self._send_json(payload)
                    return
                if path == "/assets/header-logos.png":
                    self._send_media(ASSETS_DIR / "header-logos.png", "image/png")
                    return
                if path.startswith("/media/image/"):
                    try:
                        index = int(path.rsplit("/", 1)[1])
                    except ValueError:
                        self.send_error(HTTPStatus.NOT_FOUND)
                        return
                    media_path = (
                        parent.result_store.image_path(index)
                        if parent.result_store is not None
                        else None
                    )
                    self._send_media(media_path, "image/jpeg")
                    return
                if path == "/media/audio":
                    media_path = (
                        parent.result_store.audio_path()
                        if parent.result_store is not None
                        else None
                    )
                    self._send_media(media_path, "audio/wav")
                    return
                self.send_error(HTTPStatus.NOT_FOUND)

            def do_POST(self) -> None:
                """Nhận ROI do người dùng kéo trực tiếp trên trang preview."""
                if urlsplit(self.path).path != "/api/roi":
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                try:
                    content_length = int(self.headers.get("Content-Length", "0"))
                    if content_length <= 0 or content_length > 4096:
                        raise ValueError("Invalid ROI request size")
                    payload = json.loads(self.rfile.read(content_length))
                    roi = parent.camera.set_roi(payload)
                except Exception as exc:
                    LOGGER.warning("Camera ROI update rejected: %s", exc)
                    self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
                    return
                self._send_json(roi)

            def _send_json(
                self,
                payload: object,
                status: HTTPStatus = HTTPStatus.OK,
            ) -> None:
                """Gửi phản hồi JSON nhỏ cho giao diện ROI."""
                body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _send_media(self, path: object, content_type: str) -> None:
                """Serve only paths selected by LatestResultStore."""
                if path is None or not hasattr(path, "is_file") or not path.is_file():
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                try:
                    body = path.read_bytes()
                except OSError:
                    LOGGER.exception("Could not read dashboard media: %s", path)
                    self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR)
                    return
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", content_type)
                self.send_header("Cache-Control", "no-store")
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _send_index(self) -> None:
                # ROI thường trực đã được vẽ trên bản sao preview bằng OpenCV.
                # Canvas trong suốt chỉ nhận thao tác và vẽ nét tạm khi kéo chuột.
                body = """<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Durian Camera Preview</title>
  <style>
    *{box-sizing:border-box} body{margin:0;background:#111;color:#eee;font-family:sans-serif;
    min-height:100vh;display:flex;flex-direction:column;align-items:center;padding:12px;gap:10px}
    .toolbar{display:flex;align-items:center;gap:10px;flex-wrap:wrap;justify-content:center}
    button{border:0;border-radius:6px;padding:9px 14px;font-weight:700;cursor:pointer}
    #reset{background:#555;color:#fff} #status{font-size:14px;color:#9ef59e}
    .rotate.active{background:#1976d2;color:#fff}.rotate{background:#444;color:#fff}
    .stage{position:relative;display:inline-block;line-height:0;max-width:100%}
    img,video,canvas{opacity:1!important;filter:none!important;mix-blend-mode:normal!important}
    #preview{display:block;max-width:calc(100vw - 24px);max-height:calc(100vh - 76px);
    width:auto;height:auto;object-fit:contain}
    #roi{position:absolute;inset:0;width:100%;height:100%;cursor:crosshair;
    touch-action:none;background:transparent!important}
  </style>
</head>
<body>
  <div class="toolbar">
    <strong>Drag to select the AI detection area</strong>
    <button id="reset" type="button">Use full frame</button>
    <span id="status">Loading ROI...</span>
  </div>
  <div class="toolbar">
    <strong>Xoay camera:</strong>
    <button class="rotate" data-angle="0" type="button">0°</button>
    <button class="rotate" data-angle="90" type="button">90°</button>
    <button class="rotate" data-angle="180" type="button">180°</button>
    <button class="rotate" data-angle="270" type="button">270°</button>
  </div>
  <div class="stage">
    <img id="preview" src="/stream.mjpg" alt="camera preview">
    <canvas id="roi"></canvas>
  </div>
  <script>
    const image=document.getElementById('preview');
    const canvas=document.getElementById('roi');
    const context=canvas.getContext('2d');
    const status=document.getElementById('status');
    let roi={x:0,y:0,width:1,height:1};
    let start=null;
    let draft=null;
    let rotation=localStorage.getItem('previewRotation')||'90';

    function displayedToRaw(value){
      if(rotation==='90')return{x:value.y,y:1-value.x-value.width,
        width:value.height,height:value.width};
      if(rotation==='180')return{x:1-value.x-value.width,y:1-value.y-value.height,
        width:value.width,height:value.height};
      if(rotation==='270')return{x:1-value.y-value.height,y:value.x,
        width:value.height,height:value.width};
      return value;
    }

    function setRotation(angle){
      rotation=angle;localStorage.setItem('previewRotation',angle);
      document.querySelectorAll('.rotate').forEach(button=>
        button.classList.toggle('active',button.dataset.angle===angle));
      status.textContent='Rotating camera to '+angle+'°...';
      image.src='/stream.mjpg?rotate='+angle+'&t='+Date.now();
    }

    function resizeCanvas(){
      const width=Math.max(1,image.clientWidth),height=Math.max(1,image.clientHeight);
      if(canvas.width!==width||canvas.height!==height){canvas.width=width;canvas.height=height}
      draw(draft);
    }
    function draw(value){
      context.clearRect(0,0,canvas.width,canvas.height);
      if(!value)return;
      const x=value.x*canvas.width,y=value.y*canvas.height;
      const width=value.width*canvas.width,height=value.height*canvas.height;
      context.strokeStyle='#2eff57';context.lineWidth=3;context.setLineDash([10,6]);
      context.strokeRect(x,y,width,height);context.setLineDash([]);
    }
    function point(event){
      const box=canvas.getBoundingClientRect();
      return {x:Math.max(0,Math.min(box.width,event.clientX-box.left)),
              y:Math.max(0,Math.min(box.height,event.clientY-box.top))};
    }
    async function save(value){
      status.textContent='Saving...';
      const response=await fetch('/api/roi',{method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify(value)});
      const payload=await response.json();
      if(!response.ok)throw new Error(payload.error||'Could not save ROI');
      roi=payload;draft=null;draw(null);status.textContent='Detection area saved';
    }
    canvas.addEventListener('pointerdown',event=>{
      start=point(event);draft={x:start.x/canvas.width,y:start.y/canvas.height,width:0,height:0};
      canvas.setPointerCapture(event.pointerId);
    });
    canvas.addEventListener('pointermove',event=>{
      if(!start)return;const current=point(event);
      const left=Math.min(start.x,current.x),top=Math.min(start.y,current.y);
      draft={x:left/canvas.width,y:top/canvas.height,
        width:Math.abs(current.x-start.x)/canvas.width,
        height:Math.abs(current.y-start.y)/canvas.height};draw(draft);
    });
    canvas.addEventListener('pointerup',async event=>{
      if(!start||!draft)return;start=null;canvas.releasePointerCapture(event.pointerId);
      if(draft.width<.02||draft.height<.02){draft=null;draw(null);status.textContent='Area is too small';return}
      try{await save(displayedToRaw(draft))}catch(error){draft=null;draw(null);status.textContent=error.message}
    });
    document.getElementById('reset').addEventListener('click',async()=>{
      try{await save({x:0,y:0,width:1,height:1})}catch(error){status.textContent=error.message}
    });
    document.querySelectorAll('.rotate').forEach(button=>
      button.addEventListener('click',()=>setRotation(button.dataset.angle)));
    image.addEventListener('load',resizeCanvas);window.addEventListener('resize',resizeCanvas);
    fetch('/api/roi').then(response=>response.json()).then(value=>{
      roi=value;status.textContent='Default ROI loaded';resizeCanvas();
    }).catch(()=>{status.textContent='Could not load ROI';resizeCanvas()});
    setRotation(rotation);
  </script>
</body>
</html>""".encode("utf-8")
                body = DASHBOARD_HTML.encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _send_stream(self, rotation: int) -> None:
                self.send_response(HTTPStatus.OK)
                self.send_header("Age", "0")
                self.send_header("Cache-Control", "no-cache, private")
                self.send_header("Pragma", "no-cache")
                self.send_header(
                    "Content-Type",
                    "multipart/x-mixed-replace; boundary=frame",
                )
                self.end_headers()
                delay = 1.0 / parent.fps
                while True:
                    try:
                        frame = parent.camera.read_jpeg(parent.jpeg_quality)
                        if rotation:
                            decoded = cv2.imdecode(
                                np.frombuffer(frame, dtype=np.uint8),
                                cv2.IMREAD_COLOR,
                            )
                            if decoded is None:
                                raise RuntimeError("Could not decode preview frame")
                            rotate_codes = {
                                90: cv2.ROTATE_90_CLOCKWISE,
                                180: cv2.ROTATE_180,
                                270: cv2.ROTATE_90_COUNTERCLOCKWISE,
                            }
                            decoded = cv2.rotate(decoded, rotate_codes[rotation])
                            encoded, buffer = cv2.imencode(
                                ".jpg",
                                decoded,
                                [cv2.IMWRITE_JPEG_QUALITY, parent.jpeg_quality],
                            )
                            if not encoded:
                                raise RuntimeError("Could not encode rotated preview frame")
                            frame = buffer.tobytes()
                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n")
                        self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode("ascii"))
                        self.wfile.write(frame)
                        self.wfile.write(b"\r\n")
                        time.sleep(delay)
                    except (BrokenPipeError, ConnectionResetError):
                        return
                    except Exception:
                        now = time.monotonic()
                        if now - parent._last_error_log_at >= 5.0:
                            parent._last_error_log_at = now
                            LOGGER.warning("Camera preview frame read failed; retrying")
                        time.sleep(0.5)

        self._server = ThreadingHTTPServer((self.host, self.port), PreviewHandler)
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="camera-preview-server",
            daemon=True,
        )
        self._thread.start()
        display_host = _preview_display_host(self.host)
        self._display_host = display_host
        LOGGER.info(
            "Camera preview available at http://%s:%s/ (listening on %s:%s)",
            display_host,
            self.port,
            self.host or "0.0.0.0",
            self.port,
        )

    def stop(self) -> None:
        """Stop the preview server."""
        if self._server is None:
            return
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self._server = None
        self._thread = None
        self._display_host = None
        LOGGER.info("Camera preview stopped")
