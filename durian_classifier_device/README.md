# Durian Classifier Device

Ứng dụng độc lập cho Raspberry Pi 4: nhấn công tắc hành trình để chụp ảnh USB camera,
phân loại độ chín sầu riêng bằng model ONNX/Keras/PyTorch và hiển thị kết quả
trên LCD1602 I2C.
Kết quả được giữ trên LCD trong 15 giây trước khi trở về màn hình tên trường và đội.

Tất cả lệnh bên dưới dùng đường dẫn tương đối. Repository có thể
được clone hoặc giải nén ở bất kỳ thư mục nào; `scripts/run.sh` luôn tự chuyển
về đúng thư mục ứng dụng trước khi chạy.

## Chạy nhanh sau khi tải source

Cài các gói hệ thống trên Raspberry Pi OS:

```bash
sudo apt update
sudo apt install -y python3-venv python3-dev build-essential swig \
  liblgpio-dev libportaudio2 libsndfile1 alsa-utils v4l-utils i2c-tools
```

Tải source từ GitHub:

```bash
git clone https://github.com/JayC0603/Neurix.git
cd Neurix
```

Cấu trúc tối thiểu mong đợi:

```text
<repo>/
├── durian_classifier_device/
│   ├── app.py
│   ├── .env.example
│   └── scripts/
└── model/
    ├── onnx/mobilenetv1_image.onnx
    └── model-audio.pt
```

Từ thư mục gốc của repository:

```bash
cd durian_classifier_device
cp .env.example .env
chmod +x scripts/*.sh
./scripts/setup.sh
./scripts/run.sh
```

Lần cài đặt đầu có thể mất vài phút do PyTorch, TensorFlow và
ONNX Runtime có dung lượng lớn. Không chạy `setup.sh` bằng `sudo`.

Nếu tên hoặc vị trí model khác, chỉnh `MODEL_PATH` và
`AUDIO_MODEL_PATH` trong `.env`. Đường dẫn tương đối trong `.env` được tính
từ thư mục `durian_classifier_device/`.

`.env.example` tắt kiểm tra khay trống ở lần chạy đầu. Sau khi camera hoạt
động, chụp ảnh khay trống, lưu thành `config/empty_scene.jpg`, rồi đổi:

```dotenv
DURIAN_PRESENCE_ENABLED=true
EMPTY_SCENE_REFERENCE_PATH=config/empty_scene.jpg
```

Khi log báo `Camera preview available`, mở `http://localhost:8081/` trên
Pi hoặc `http://<IP-của-Pi>:8081/` trên thiết bị cùng mạng. Lấy IP bằng:

```bash
hostname -I
```

## Luồng hoạt động

```text
IDLE -> SWITCH PRESSED -> CAPTURING -> ANALYZING -> SHOW RESULT -> READY
```

Khi Raspberry Pi khởi động, `durian-classifier.service` gọi
`ServoController.init_servos()` để sở hữu PWM0 nhưng chưa phát lệnh góc. Phần
mềm không khởi tạo, không mở và không ghi kênh PWM nào cho servo thứ hai.

Callback GPIO chỉ đặt một job vào `Queue(maxsize=1)`. Một worker duy nhất sở hữu luồng
xử lý camera/model; lần nhấn mới bị bỏ qua trong khi hệ thống bận.

### Chuỗi Servo 1 MG996R

Chỉ PWM0/BCM18 phát xung điều khiển Servo 1. Khi micro đã bắt đầu ghi, Servo 1
gõ đúng ba nhịp ở góc 120° trong 3 giây, luôn quay về HOME 90°
giữa các nhịp và không cộng/trừ từ vị trí cũ. Kết thúc, Servo 1 trở về HOME
rồi mới detach PWM. Hệ thống chỉ đưa kết quả phân loại cuối lên LCD sau khi
đủ ba nhịp.
Sau lệnh HOME cuối, code chờ đủ `RETURN_HOME_TIME + HOME_SETTLE_TIME` rồi luôn
detach PWM để dừng hunting/buzz. Nếu
thanh bị nghiêng khi mất torque, cơ cấu cần điểm tựa, lò xo hoặc đối trọng tại
HOME; không giữ PWM vô thời hạn để chống tải vì MG996R có thể hunting liên tục.

Mỗi lần nhấn công tắc, nhánh camera chụp 3 ảnh theo các mốc cách nhau 3 giây và
chạy model ảnh. Đồng thời, nhánh audio mở microphone, chạy ba nhịp servo
và chạy model âm thanh. Hai kết quả được chờ đủ rồi
fusion theo trọng số cấu hình.

MG996R phải dùng nguồn ngoài 5–6 V đủ dòng và bắt buộc nối chung GND nguồn servo
với GND Raspberry Pi. Nếu tay gõ tự
trôi sau detach, đó là tải cơ khí; cần chốt chặn/lò xo/cân bằng tay đòn, không bù
bằng lệnh góc ngẫu nhiên.

## Cấu trúc

```text
app.py                         entry point và lifecycle
config.py                      cấu hình biến môi trường
config/classes.txt             thứ tự class bắt buộc
hardware/                      LCD, công tắc và camera
inference/                     preprocessing, model và kết quả
services/                      worker capture/classify
utils/logging_config.py        console + rotating logs/app.log
scripts/                       setup, kiểm tra, chạy và cài systemd
tests/                         test độc lập từng thành phần
captures/                      ảnh đã chụp
logs/                          log runtime
```

## Đấu dây

### LCD1602 I2C

| LCD | Raspberry Pi 4 |
|---|---|
| VCC | Pin vật lý 2 hoặc 4, 5V |
| GND | Pin vật lý 6, GND |
| SDA | Pin vật lý 3, GPIO2/SDA1 |
| SCL | Pin vật lý 5, GPIO3/SCL1 |

### Công tắc hành trình KW11-3Z

| Switch | Raspberry Pi 4 |
|---|---|
| COM | GND, pin vật lý 9 hoặc 6 |
| NO | Pin vật lý 29, BCM GPIO5 |
| NC | Không dùng |

### Servo MG996R

| Kết nối | Raspberry Pi / nguồn ngoài |
|---|---|
| Signal Servo 1 | Physical pin 12 / BCM18 / PWM0 |
| VCC Servo 1 | Cực dương nguồn 6.01 V riêng số 1 |
| GND Servo 1 | Nối chung với physical pin 39 (GND) của Pi |

Không cấp nguồn servo từ chân 5 V của Pi. GPIO18 chỉ mang tín hiệu. Không cần
nối servo thứ hai vào GPIO19; ứng dụng chỉ sở hữu PWM0. Backend dùng PWM phần cứng
để tránh jitter do scheduler Python; file lock chặn hai controller của project
cùng sở hữu PWM.

Để phân biệt lỗi riêng GPIO18/PWM0 với dây, tải và cơ khí Servo 1, dừng service,
chuyển duy nhất dây signal Servo 1 sang physical pin 11/BCM17 rồi chạy test có
xác nhận thủ công:

```bash
sudo systemctl stop durian-classifier.service
cd durian_classifier_device
source .venv/bin/activate
python tests/test_servo_1_gpio17.py
```

Test gửi đúng một chuỗi tuyệt đối `90 -> 150 -> 90`, dùng một `LGPIOFactory`,
dải xung 1–2 ms và cùng timing trong `.env`. BCM17 không có kênh hardware PWM
như BCM18, nên phép thử A/B đồng thời thay đường GPIO/PWM. Nếu BCM17 hết rung,
ưu tiên điều tra pinmux/PWM0/dây signal BCM18; nếu vẫn rung, ưu tiên horn,
backlash, tải thanh gõ, wiring và calibration cơ khí. Chuyển dây về physical
pin 12/BCM18 trước khi bật lại service.

Để phân biệt hunting do giữ torque với rung khi không còn PWM, giữ nguyên dây
production BCM18 và chạy test HOME không gõ:

```bash
sudo systemctl stop durian-classifier.service
cd durian_classifier_device
source .venv/bin/activate
python tests/test_servo_home_hunting.py
```

Test giữ Servo 1 tại HOME với PWM trong 15 giây, detach riêng Servo 1 trong 10
giây và yêu cầu nhập `YEN`, `RUNG` hoặc `YEN-NGHIENG` cho từng giai đoạn. Cleanup
chỉ command HOME nếu lệnh gần nhất chưa phải HOME, chờ đủ rồi mới detach. Chỉ đặt
cơ cấu bị `YEN-NGHIENG` sau detach thì cần điểm tựa, lò xo hoặc đối trọng tại HOME.

Trước khi gắn tải, calibration HOME theo thứ tự: tháo horn/thanh gõ nếu cần, chạy
lệnh 90 độ, chờ servo ổn định, rồi lắp horn đúng vị trí HOME cơ khí. Sau đó mới
thử `90 -> 150 -> 90`. Nếu code đã về
90 nhưng thanh gõ còn lệch, kiểm tra riêng pulse calibration từng servo, cách lắp
horn, backlash, preload, khối lượng tay gõ và mô-men servo. Không bù sai lệch bằng
góc cộng dồn.

GPIO chỉ chịu logic 3.3V. Không nối 5V vào GPIO5 hoặc công tắc. Phần mềm bật pull-up
nội bộ: chưa nhấn là HIGH, nhấn nối xuống GND là LOW.

## Bật và kiểm tra I2C

Project không tự sửa `/boot/config.txt`. Bật thủ công:

```bash
sudo raspi-config
```

Chọn `Interface Options -> I2C -> Enable`, khởi động lại nếu được yêu cầu, sau đó:

```bash
sudo apt install i2c-tools
./scripts/check_i2c.sh
```

LCD thường xuất hiện tại `0x27` hoặc `0x3F`. Ứng dụng ưu tiên hai địa chỉ này và tiếp
tục chạy camera/model nếu LCD không được tìm thấy.

## Môi trường Python và cài đặt

Kiểm tra trước:

```bash
uname -m
python3 --version
python3 -m venv --help >/dev/null
```

Sau khi tải đủ hai file model theo cấu trúc ở phần "Chạy nhanh":

```bash
cd durian_classifier_device
chmod +x scripts/*.sh
cp .env.example .env
./scripts/setup.sh
```

`setup.sh` tạo `.venv`, cài dependencies, kiểm tra file model và tạo
`captures/`, `logs/`. Không chạy script này bằng `sudo`; chỉ các bước cài
gói hệ thống hoặc service mới cần `sudo`.

## Class và preprocessing

Thứ tự output không được thay đổi:

```text
0 -> unripe
1 -> ripe
2 -> Overripe
```

Model phải có đúng ba output. Ứng dụng sẽ dừng với `ModelConfigurationError` nếu
mapping hoặc output shape không đúng.

`MODEL_NORMALIZATION` hỗ trợ `zero_one`, `minus_one_one`, và `none`.
Normalization khi inference **phải giống normalization lúc training**. Giá trị mặc
định `zero_one` không phải bằng chứng về preprocessing đã dùng khi huấn luyện; hãy
xác nhận pipeline training trước khi vận hành thật.

`MODEL_OUTPUT_TYPE` hỗ trợ `auto`, `softmax`, `sigmoid`, và `logits`. Với `auto`, output
có tổng gần 1 và nằm trong `[0,1]` được xem là xác suất; trường hợp khác được xem là
logits và áp dụng softmax.

## Test

Kích hoạt virtualenv rồi chạy từ thư mục project:

```bash
source .venv/bin/activate

python tests/test_lcd.py
python tests/test_limit_switch.py
python tests/test_camera.py
python tests/test_model.py --image ./captures/example.jpg
python tests/test_pipeline.py
```

Test LCD giả lập:

```bash
python tests/test_lcd.py --mock --delay 0.2
```

Test camera/model/pipeline bằng ảnh có sẵn, không dùng GPIO hoặc camera thật:

```bash
python tests/test_camera.py --mock-image ./captures/example.jpg
HARDWARE_MOCK=true MOCK_IMAGE_PATH=./captures/example.jpg \
  python tests/test_pipeline.py
```

Trong pipeline mock, nhấn Enter để tạo đúng một job.

## Chạy ứng dụng

Đảm bảo không còn chương trình FFmpeg/OpenCV khác giữ `/dev/video0`:

```bash
fuser -v /dev/video0
```

Sau đó:

```bash
cd durian_classifier_device
source .venv/bin/activate
cp .env.example .env   # chỉ cần lần đầu
python app.py
```

Hoặc:

```bash
./scripts/run.sh
```

Khi ứng dụng đang chạy, camera preview được phát từ chính process phân loại tại:

```text
http://<raspberry-pi-ip>:8081/
```

Ví dụ trên chính Raspberry Pi:

```text
http://localhost:8081/
```

Không chạy `camera_web_ffmpeg.py` song song với app này vì hai process sẽ tranh
`/dev/video0`. Nếu muốn đổi port/FPS preview, chỉnh `CAMERA_PREVIEW_PORT` và
`CAMERA_PREVIEW_FPS` trong `.env`.

Mock mode vẫn dùng model thật:

```bash
HARDWARE_MOCK=true MOCK_IMAGE_PATH=./captures/example.jpg python app.py
```

Nhấn Enter để trigger. `Ctrl+C`, `SIGINT` và `SIGTERM` đều thực hiện cleanup.

## Tự chạy khi cấp nguồn

Ứng dụng có thể chạy như một dịch vụ `systemd`; không cần SSH vào Pi để chạy
`app.py`. Cài một lần:

```bash
cd durian_classifier_device
chmod +x scripts/*.sh
sudo ./scripts/install-autostart.sh
```

Service dùng đúng `.venv` và `.env` của project, tự chạy sau khi boot và tự thử lại
sau 10 giây nếu camera, GPIO hoặc thiết bị USB chưa sẵn sàng. Khi LCD hiện
`System ready`, dòng thứ hai tự hiển thị địa chỉ IP do DHCP cấp. Nhấn công tắc
GPIO5 để chạy pipeline; không cần SSH để tìm IP khi chuyển sang mạng đã lưu khác.

Raspberry Pi tự nhận IP trên từng mạng bằng DHCP. Mật khẩu của một Wi-Fi hoàn toàn
mới không thể tự suy ra; cần lưu mạng đó một lần bằng NetworkManager. Sau khi đã
lưu, Pi sẽ tự kết nối lại ở các lần khởi động sau. Có thể mở preview bằng IP trên
LCD và cổng `8081`, ví dụ `http://192.168.1.5:8081/`.

Các lệnh kiểm tra và quản lý:

```bash
sudo systemctl status durian-classifier.service
journalctl -u durian-classifier.service -f
sudo systemctl restart durian-classifier.service
sudo systemctl disable --now durian-classifier.service
```

Không chạy `scripts/run.sh`, `python app.py` hoặc container Docker song song với
service vì chúng sẽ tranh camera và GPIO. Ứng dụng dùng `logs/app.lock` để phát hiện
instance thứ hai và dừng sớm với hướng dẫn restart service, thay vì báo `GPIO busy`.

## Lỗi thường gặp

- `No such file /dev/i2c-1`: I2C chưa bật; dùng `raspi-config`, sau đó reboot.
- Không thấy `0x27`/`0x3F`: kiểm tra VCC, GND, SDA, SCL; thử chỉnh biến trở tương phản.
- `Cannot open camera`: `/dev/video0` đang bị process khác giữ; kiểm tra bằng `fuser`.
- Camera không đạt 1280x720: chọn kích thước UVC hỗ trợ trong `.env`.
- `No module named tensorflow`: venv chưa có TensorFlow tương thích; không cài bừa vào
  Python hệ thống.
- Model output không phải 3: model không khớp mapping bắt buộc, inference sẽ bị chặn.
- Confidence/class sai: kiểm tra lại normalization và output type so với lúc training.
- `PinFactoryFallback` hoặc lỗi GPIO: kiểm tra `gpiozero`, `lgpio`, quyền GPIO và bảo
  đảm đang chạy trên Raspberry Pi.
- Nhiều callback: kiểm tra chân NO/COM và giữ debounce `0.05`; không nối chân NC.

## Logging và ảnh

Log được ghi đồng thời ra console và `logs/app.log`, xoay vòng ở 2 MB với năm bản sao.
Ảnh được lưu trong `captures/`. Chỉ bật `DELETE_IMAGE_AFTER_INFERENCE=true` nếu muốn
xóa ảnh sau inference thành công; ảnh lỗi được giữ để chẩn đoán.
