#!/bin/bash

echo "======================================"
echo " RASPBERRY PI 4 - TEST TẤT CẢ CỔNG"
echo "======================================"

echo ""
echo "===== 1. THÔNG TIN HỆ THỐNG ====="
echo "Hostname:"
hostname

echo ""
echo "IP:"
hostname -I

echo ""
echo "OS:"
cat /etc/os-release | grep PRETTY_NAME

echo ""
echo "Kernel:"
uname -a

echo ""
echo "CPU:"
lscpu | grep "Model name" || cat /proc/cpuinfo | grep "Model" | head -1

echo ""
echo "RAM:"
free -h

echo ""
echo "Disk:"
df -h

echo ""
echo "===== 2. KIỂM TRA NGUỒN / NHIỆT ĐỘ ====="
echo "Nhiệt độ:"
vcgencmd measure_temp 2>/dev/null || echo "Không chạy được vcgencmd"

echo ""
echo "Trạng thái nguồn:"
vcgencmd get_throttled 2>/dev/null || echo "Không chạy được vcgencmd"

echo ""
echo "Ghi chú:"
echo "throttled=0x0 là bình thường"
echo "Nếu khác 0x0 có thể bị yếu nguồn hoặc quá nhiệt"

echo ""
echo "===== 3. KIỂM TRA USB ====="
echo "Thiết bị USB:"
lsusb

echo ""
echo "Log USB gần nhất:"
dmesg | grep -i usb | tail -20

echo ""
echo "===== 4. KIỂM TRA CAMERA ====="
echo "Danh sách /dev/video:"
ls /dev/video* 2>/dev/null || echo "Không thấy camera /dev/video"

echo ""
echo "Danh sách camera bằng v4l2:"
v4l2-ctl --list-devices 2>/dev/null || echo "Chưa cài v4l-utils hoặc không có camera"

echo ""
echo "Thông tin /dev/video0:"
v4l2-ctl -d /dev/video0 --all 2>/dev/null || echo "Không đọc được /dev/video0"

echo ""
echo "===== 5. KIỂM TRA GPIO ====="
echo "Sơ đồ chân Raspberry Pi:"
pinout 2>/dev/null || echo "Không có lệnh pinout"

echo ""
echo "Danh sách GPIO:"
gpioinfo 2>/dev/null || echo "Chưa cài gpiod hoặc không đọc được GPIO"

echo ""
echo "===== 6. KIỂM TRA I2C ====="
echo "Thiết bị I2C bus 1:"
i2cdetect -y 1 2>/dev/null || echo "I2C chưa bật hoặc chưa cài i2c-tools"

echo ""
echo "===== 7. KIỂM TRA SPI ====="
echo "Thiết bị SPI:"
ls /dev/spidev* 2>/dev/null || echo "Không thấy SPI. Có thể SPI chưa bật"

echo ""
echo "===== 8. KIỂM TRA UART / SERIAL ====="
echo "Cổng ttyUSB:"
ls /dev/ttyUSB* 2>/dev/null || echo "Không thấy /dev/ttyUSB"

echo ""
echo "Cổng ttyACM:"
ls /dev/ttyACM* 2>/dev/null || echo "Không thấy /dev/ttyACM"

echo ""
echo "Cổng serial Raspberry Pi:"
ls /dev/ttyAMA* /dev/ttyS* 2>/dev/null || echo "Không thấy ttyAMA/ttyS"

echo ""
echo "Log serial:"
dmesg | grep -i tty | tail -20

echo ""
echo "===== 9. KIỂM TRA MẠNG ====="
echo "Card mạng:"
ip addr

echo ""
echo "Route:"
ip route

echo ""
echo "WiFi:"
iwconfig 2>/dev/null || echo "Không có iwconfig hoặc WiFi không khả dụng"

echo ""
echo "Ping Internet:"
ping -c 4 google.com 2>/dev/null || echo "Không ping được Internet"

echo ""
echo "===== 10. KIỂM TRA BLUETOOTH ====="
bluetoothctl show 2>/dev/null || echo "Bluetooth chưa bật hoặc chưa cài bluetoothctl"

echo ""
echo "===== 11. KIỂM TRA ÂM THANH ====="
echo "Thiết bị audio:"
aplay -l 2>/dev/null || echo "Không thấy thiết bị audio output"

echo ""
echo "Thiết bị micro:"
arecord -l 2>/dev/null || echo "Không thấy thiết bị audio input"

echo ""
echo "===== 12. TÓM TẮT NHANH ====="

echo ""
if ls /dev/video* >/dev/null 2>&1; then
    echo "[OK] Camera có thiết bị /dev/video"
else
    echo "[FAIL] Không thấy camera"
fi

if i2cdetect -y 1 >/dev/null 2>&1; then
    echo "[OK] I2C chạy được"
else
    echo "[FAIL] I2C chưa chạy"
fi

if ls /dev/spidev* >/dev/null 2>&1; then
    echo "[OK] SPI đã bật"
else
    echo "[FAIL] SPI chưa bật hoặc không có thiết bị"
fi

if gpioinfo >/dev/null 2>&1; then
    echo "[OK] GPIO đọc được"
else
    echo "[FAIL] GPIO chưa đọc được"
fi

if ls /dev/ttyUSB* >/dev/null 2>&1 || ls /dev/ttyACM* >/dev/null 2>&1; then
    echo "[OK] Có thiết bị Serial USB"
else
    echo "[INFO] Không thấy thiết bị Serial USB"
fi

if ping -c 1 google.com >/dev/null 2>&1; then
    echo "[OK] Có Internet"
else
    echo "[FAIL] Không có Internet"
fi

THROTTLED=$(vcgencmd get_throttled 2>/dev/null)
if echo "$THROTTLED" | grep -q "0x0"; then
    echo "[OK] Nguồn ổn"
else
    echo "[WARN] Có thể yếu nguồn hoặc quá nhiệt: $THROTTLED"
fi

echo ""
echo "======================================"
echo " TEST XONG"
echo "======================================"