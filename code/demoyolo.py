import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Không mở được camera.")
    exit()

ret, frame = cap.read()

if not ret:
    print("Không đọc được ảnh từ camera.")
    cap.release()
    exit()

results = model.predict(frame, imgsz=320, conf=0.25, save=True)

print("Đã chạy YOLO xong.")
print(results[0].boxes)

cap.release()