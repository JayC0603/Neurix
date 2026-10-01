# Hệ thống phân loại độ chín sầu riêng

Repository chứa ứng dụng Raspberry Pi kết hợp camera, microphone, servo,
LCD1602 và hai model AI để phân loại sầu riêng thành ba nhóm:
`unripe`, `ripe`, `Overripe`.

Tài liệu đấu dây, cấu hình, test phần cứng và khắc phục lỗi chi tiết nằm
tại [durian_classifier_device/README.md](durian_classifier_device/README.md).

## Cấu trúc cần thiết

```text
.
├── durian_classifier_device/       # ứng dụng chạy trên Raspberry Pi
├── model/
│   ├── onnx/
│   │   └── mobilenetv1_image.onnx   # model hình ảnh
│   └── model-audio.pt               # model âm thanh
└── README.md
```

Hai file model phải tồn tại trước khi chạy `setup.sh`. Nếu lưu model ở
vị trí khác, cập nhật `MODEL_PATH` và `AUDIO_MODEL_PATH` trong
`durian_classifier_device/.env`.

## Cài đặt trên Raspberry Pi

### 1. Tải repository

```bash
git clone https://github.com/JayC0603/Neurix.git
cd Neurix
```

Cũng có thể tải file ZIP và giải nén, sau đó mở terminal tại thư mục
vừa giải nén. Không cần đặt repository ở một đường dẫn cố định.

### 2. Cài gói hệ thống

```bash
sudo apt update
sudo apt install -y python3-venv python3-dev build-essential swig \
  liblgpio-dev libportaudio2 libsndfile1 alsa-utils v4l-utils i2c-tools
```

Bật I2C bằng `sudo raspi-config`, chọn
`Interface Options -> I2C -> Enable`, sau đó khởi động lại nếu được yêu cầu.

### 3. Tạo cấu hình và môi trường Python

```bash
cd durian_classifier_device
cp .env.example .env
chmod +x scripts/*.sh
./scripts/setup.sh
```

`.env.example` chỉ dùng đường dẫn tương đối. Script cài đặt tự xác định
thư mục project nên có thể chạy repository ở bất kỳ vị trí nào.

### 4. Chạy thủ công

```bash
./scripts/run.sh
```

Giữ terminal này mở. Dừng ứng dụng bằng `Ctrl+C`.

### 5. Mở dashboard

Lấy IP của Raspberry Pi:

```bash
hostname -I
```

Mở một trong hai địa chỉ:

- Trên chính Raspberry Pi: `http://localhost:8081/`
- Trên thiết bị cùng mạng: `http://<IP-của-Pi>:8081/`

Ví dụ: `http://192.168.1.4:8081/`.

## Tự chạy khi khởi động

Sau khi chạy thủ công thành công, cài service:

```bash
sudo ./scripts/install-autostart.sh
```

Quản lý service:

```bash
sudo systemctl status durian-classifier.service
sudo systemctl restart durian-classifier.service
journalctl -u durian-classifier.service -f
```

Không chạy `./scripts/run.sh` song song với service. Camera, GPIO và file
khóa chỉ cho phép một instance hoạt động.

## Chạy thử không có phần cứng

Đặt một ảnh test trong `durian_classifier_device/captures/`, sau đó chạy:

```bash
cd durian_classifier_device
HARDWARE_MOCK=true MOCK_IMAGE_PATH=./captures/example.jpg ./scripts/run.sh
```

Mock mode không truy cập LCD, GPIO hoặc camera thật nhưng vẫn nạp model AI.

## Lỗi thường gặp

- `Durian classifier is already running`: dùng
  `sudo systemctl restart durian-classifier.service`, không chạy thêm instance.
- Không mở được cổng `8081`: kiểm tra
  `journalctl -u durian-classifier.service -f` và bảo đảm hai thiết bị cùng mạng.
- `image/audio model not found`: kiểm tra cấu trúc `model/` hoặc sửa hai
  đường dẫn model trong `.env`.
- `Cannot open camera`: dùng `fuser -v /dev/video0` để tìm process đang giữ camera.
- Không thấy LCD: chạy `./scripts/check_i2c.sh` và kiểm tra địa chỉ
  `0x27`/`0x3F`.

Log runtime nằm trong `durian_classifier_device/logs/`; ảnh chụp nằm trong
`durian_classifier_device/captures/`.
