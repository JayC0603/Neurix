<div align="center">

# Neurix

### Hệ thống AI đa phương thức phân loại độ chín sầu riêng

Kết hợp **hình ảnh**, **âm thanh va đập**, **Raspberry Pi** và **Edge AI**
để phân loại sầu riêng theo thời gian thực.

![Raspberry Pi](https://img.shields.io/badge/Raspberry%20Pi-4-C51A4A?logo=raspberrypi&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-CPU-005CED?logo=onnx&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Audio-EE4C2C?logo=pytorch&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-Camera-5C3EE8?logo=opencv&logoColor=white)

</div>

---

## Giới thiệu

Neurix là hệ thống nhúng chấm độ chín sầu riêng trên Raspberry Pi. Mỗi
lần người dùng nhấn công tắc, hệ thống chụp nhiều ảnh, ghi âm ba cú gõ
từ servo, chạy hai model AI độc lập và fusion xác suất để trả về một
trong ba nhóm:

| Nhãn model | Hiển thị | Ý nghĩa |
|---|---|---|
| `unripe` | Chưa chín | Sầu riêng chưa đạt độ chín |
| `ripe` | Chín | Sầu riêng đã chín |
| `Overripe` | Quá chín | Sầu riêng đã quá chín |

Kết quả và confidence được hiển thị trên LCD1602 và dashboard web.
Confidence được lấy trực tiếp từ xác suất model/fusion, không dùng giá
trị demo hoặc phần trăm ngẫu nhiên.

## Tính năng chính

- Phân loại hình ảnh bằng MobileNetV1 ONNX trên CPU.
- Phân loại âm thanh va đập bằng model PyTorch.
- Chụp ba ảnh và gõ servo ba lần trong hai nhánh chạy song song.
- Fusion vector xác suất ba lớp, mặc định `60% image + 40% audio`.
- Dashboard web thời gian thực: camera, ROI, kết quả, ảnh và audio gần nhất.
- Cho phép kéo ROI trực tiếp trên trình duyệt.
- Tùy chọn phát hiện khay trống trước khi phân loại.
- Chặn nhiều instance cùng tranh camera, GPIO hoặc PWM.
- Hỗ trợ chạy nền và tự khởi động bằng `systemd`.
- Có mock mode để kiểm tra phần mềm khi không có phần cứng.

## Kiến trúc xử lý

```text
Limit switch
     |
     v
Capture/Classify worker
     |
     +------------------------------+
     |                              |
     v                              v
3 camera images              Record microphone
     |                       + Servo 1 taps x3
     v                              |
Image model                         v
     |                       Audio model
     +--------------+---------------+
                    |
                    v
         Probability fusion 60/40
                    |
          +---------+---------+
          |                   |
          v                   v
       LCD1602          Web dashboard :8081
```

## Phần cứng

| Thiết bị | Kết nối mặc định |
|---|---|
| Raspberry Pi 4 | Thiết bị xử lý chính |
| USB camera | `/dev/video0` |
| USB microphone | Thiết bị input mặc định của ALSA |
| LCD1602 I2C | SDA: GPIO2, SCL: GPIO3, địa chỉ `0x27` |
| Công tắc hành trình | NO: GPIO5, COM: GND |
| Servo MG996R | Signal: GPIO18/PWM0 |

> [!CAUTION]
> Servo MG996R phải dùng nguồn ngoài 5–6 V đủ dòng và phải nối chung GND
> với Raspberry Pi. Không cấp nguồn servo trực tiếp từ chân 5 V của Pi.

Sơ đồ chân, calibration servo và quy trình kiểm tra phần cứng đầy đủ:
[durian_classifier_device/README.md](durian_classifier_device/README.md).

## Cấu trúc repository

```text
Neurix/
├── durian_classifier_device/
│   ├── app.py                  # Entry point
│   ├── config.py               # Đọc và validate .env
│   ├── hardware/               # Camera, LCD, switch, servo, microphone
│   ├── inference/              # Image/audio inference workers
│   ├── services/               # Pipeline, dashboard, result store
│   ├── scripts/                # Setup, run, systemd, Docker
│   ├── tests/                  # Hardware/unit/integration tests
│   └── .env.example            # Cấu hình mẫu portable
├── model/
│   ├── onnx/mobilenetv1_image.onnx
│   └── model-audio.pt
├── code/                            # Các công cụ/model demo bổ sung
└── README.md
```

## Yêu cầu

- Raspberry Pi OS 64-bit khuyến nghị.
- Python 3.10 trở lên và có wheel phù hợp cho TensorFlow/PyTorch.
- Camera V4L2, microphone ALSA và I2C đã được bật.
- Khoảng trống đĩa đủ cho virtual environment và model AI.
- Hai file model production:
  - `model/onnx/mobilenetv1_image.onnx`
  - `model/model-audio.pt`

## Cài đặt

### 1. Clone repository

```bash
git clone https://github.com/JayC0603/Neurix.git
cd Neurix
```

Repository có thể được clone hoặc giải nén ở bất kỳ thư mục nào. Các
script tự xác định vị trí project, không phụ thuộc `/home/raspberrypi/...`.

### 2. Cài gói hệ thống

```bash
sudo apt update
sudo apt install -y \
  python3-venv python3-dev build-essential swig \
  liblgpio-dev libportaudio2 libsndfile1 \
  alsa-utils v4l-utils i2c-tools
```

Bật I2C:

```bash
sudo raspi-config
```

Chọn `Interface Options` → `I2C` → `Enable`, sau đó reboot nếu hệ thống yêu cầu.

### 3. Tạo cấu hình

```bash
cd durian_classifier_device
cp .env.example .env
```

Kiểm tra các giá trị quan trọng trong `.env`:

| Biến | Mặc định | Chức năng |
|---|---|---|
| `CAMERA_DEVICE` | `/dev/video0` | Camera USB |
| `MODEL_PATH` | `../model/onnx/mobilenetv1_image.onnx` | Model hình ảnh |
| `AUDIO_MODEL_PATH` | `../model/model-audio.pt` | Model âm thanh |
| `FUSION_IMAGE_WEIGHT` | `0.6` | Trọng số model hình ảnh |
| `FUSION_AUDIO_WEIGHT` | `0.4` | Trọng số model âm thanh |
| `CAMERA_PREVIEW_PORT` | `8081` | Cổng dashboard |
| `DURIAN_PRESENCE_ENABLED` | `false` | Bật kiểm tra khay trống |

Tổng hai trọng số fusion phải bằng `1.0`.

### 4. Cài Python environment

```bash
chmod +x scripts/*.sh
./scripts/setup.sh
```

Script sẽ:

1. Tạo `.venv` trong `durian_classifier_device/`.
2. Cài dependencies từ `requirements.txt`.
3. Kiểm tra file model và framework inference.
4. Tạo thư mục `captures/` và `logs/`.

Không chạy `setup.sh` bằng `sudo`.

### 5. Chạy ứng dụng

```bash
./scripts/run.sh
```

Khi log hiển thị `System ready`, nhấn công tắc hành trình để bắt đầu
một lượt phân loại. Dừng ứng dụng bằng `Ctrl+C`.

## Dashboard web

Lấy IP của Raspberry Pi:

```bash
hostname -I
```

Mở dashboard:

- Trên Raspberry Pi: <http://localhost:8081/>
- Trên thiết bị cùng mạng: `http://<IP-của-Pi>:8081/`
- Ví dụ: <http://192.168.1.4:8081/>

Thiết bị truy cập phải cùng Wi-Fi/LAN với Raspberry Pi. Nếu đổi cổng,
cập nhật `CAMERA_PREVIEW_PORT` trong `.env`.

## Chạy tự động bằng systemd

Chỉ cài service sau khi đã chạy thủ công thành công:

```bash
sudo ./scripts/install-autostart.sh
```

Các lệnh quản lý:

```bash
sudo systemctl status durian-classifier.service
sudo systemctl restart durian-classifier.service
sudo systemctl stop durian-classifier.service
journalctl -u durian-classifier.service -f
```

> [!IMPORTANT]
> Không chạy `./scripts/run.sh` song song với service. Mỗi thời điểm chỉ
> được có một process sở hữu camera, GPIO và PWM.

## Kiểm tra

Kích hoạt virtual environment:

```bash
source .venv/bin/activate
```

Chạy các test không tác động phần cứng:

```bash
python tests/test_fusion_logic.py
python tests/test_camera_preview_server.py
python tests/test_process_lock.py
python tests/test_camera_roi.py
```

Kiểm tra I2C:

```bash
./scripts/check_i2c.sh
```

Danh sách test camera, LCD, servo và công tắc đầy đủ nằm trong
[tài liệu thiết bị](durian_classifier_device/README.md#test).

## Mock mode

Đặt một ảnh test trong `durian_classifier_device/captures/`, sau đó chạy:

```bash
HARDWARE_MOCK=true \
MOCK_IMAGE_PATH=./captures/example.jpg \
./scripts/run.sh
```

Nhấn `Enter` để mô phỏng công tắc. Mock mode không truy cập GPIO, LCD
hoặc camera thật nhưng vẫn nạp và chạy model AI.

## Phát hiện khay trống

Tính năng này mặc định tắt để lần cài đặt đầu không phụ thuộc
vào một ảnh local. Sau khi camera ổn định:

1. Chụp khay trống trong đúng điều kiện ánh sáng vận hành.
2. Lưu ảnh thành `durian_classifier_device/config/empty_scene.jpg`.
3. Cập nhật `.env`:

```dotenv
DURIAN_PRESENCE_ENABLED=true
EMPTY_SCENE_REFERENCE_PATH=config/empty_scene.jpg
```

## Xử lý sự cố

| Hiện tượng | Cách xử lý |
|---|---|
| `Durian classifier is already running` | Dùng `sudo systemctl restart durian-classifier.service`; không chạy thêm instance |
| Không mở được dashboard | Kiểm tra service, cổng `8081` và kết nối cùng mạng |
| `Image/audio model not found` | Kiểm tra `model/` và các biến model trong `.env` |
| `Cannot open camera` | Chạy `fuser -v /dev/video0` và dừng process đang giữ camera |
| Không thấy LCD | Chạy `./scripts/check_i2c.sh`, kiểm tra `0x27`/`0x3F` và dây SDA/SCL |
| Servo rung hoặc hunting | Kiểm tra nguồn ngoài, GND chung, horn, tải cơ khí và calibration HOME |
| Microphone không thu | Chạy `arecord -l`, kiểm tra ALSA và quyền truy cập thiết bị |

Theo dõi log trực tiếp:

```bash
journalctl -u durian-classifier.service -f
```

Log xoay vòng của ứng dụng nằm trong `durian_classifier_device/logs/`; ảnh chụp
nằm trong `durian_classifier_device/captures/`.

## Tài liệu bổ sung

- [Hướng dẫn thiết bị và test chi tiết](durian_classifier_device/README.md)
- [Hướng dẫn YOLO ONNX camera stream](README_RUN_YOLO_ONNX.md)
- [Cấu hình môi trường mẫu](durian_classifier_device/.env.example)

## Tác giả

Phát triển bởi [JayC0603](https://github.com/JayC0603).

---

<div align="center">

**Neurix — Multimodal AI for smarter durian ripeness assessment.**

</div>
