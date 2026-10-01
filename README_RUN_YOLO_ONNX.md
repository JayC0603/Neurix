# Raspberry Pi YOLO ONNX Camera Stream

App nay doc USB camera `/dev/video0`, chay YOLO ONNX bang CPU, ve bounding box va stream ra trinh duyet qua MJPEG.

## 1. Tao moi truong Python

```bash
# Chạy từ thư mục gốc của repository.
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Neu `opencv-python` hoac `onnxruntime` khong co wheel phu hop voi Raspberry Pi/Python hien tai, cach thuong on dinh hon la cai OpenCV tu apt va dung Python duoc OS ho tro:

```bash
sudo apt update
sudo apt install -y python3-opencv
```

## 2. Export YOLOv8 `.pt` sang ONNX

Hien tai da co the dat file ONNX tai:

```text
model/yolov8n.onnx
```

Neu muon export lai tu file `.pt` local `code/yolov8n.pt`:

```bash
source .venv/bin/activate
pip install -r requirements-export.txt
python code/export_yolov8_to_onnx.py --weights code/yolov8n.pt --imgsz 320
```

Neu buoc export loi `Illegal instruction` hoac qua nang tren Raspberry Pi, ban nen export tren laptop/PC bang cung lenh tren,
roi copy file `.onnx` vao thu muc `model/` cua Raspberry Pi. App stream chi can ONNX, khong can Torch.

## 3. Chay stream

```bash
source .venv/bin/activate
python code/yolo_onnx_stream.py --model model/yolov8n.onnx --camera /dev/video0 --port 8000
```

Mo tren may laptop/PC cung mang:

```text
http://<IP-cua-Raspberry-Pi>:8000
```

Neu dang SSH va muon tunnel ve may cua ban:

```bash
ssh -L 8000:localhost:8000 raspberrypi@<IP-cua-Raspberry-Pi>
```

Sau do mo:

```text
http://localhost:8000
```

## Tuy chinh nhanh

Giam tai CPU:

```bash
python code/yolo_onnx_stream.py --model model/yolov8n.onnx --imgsz 256 --width 640 --height 480 --fps 10
```

Tang do chinh xac nhung cham hon:

```bash
python code/yolo_onnx_stream.py --model model/yolov8n.onnx --imgsz 416
```
