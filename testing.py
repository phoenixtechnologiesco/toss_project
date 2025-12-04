import cv2
from ultralytics import YOLO
import time

model = YOLO("best.pt")

stream_url = "rtsp://admin:phoenix0332@192.168.1.108:554/cam/realmonitor?channel=1&subtype=0"
cap = cv2.VideoCapture(stream_url)

if not cap.isOpened():
    print("ERROR: Cannot open RTSP stream")
    exit()

print("✓ Running Real-Time Detection...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame read error.")
        continue

    
    results = model.predict(
        frame,
        conf=0.3,        
        iou=0.45,
        device=0,         
        verbose=False
    )

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls)
        conf = float(box.conf)

        label = f"Bullet Hole {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Real-Time Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()