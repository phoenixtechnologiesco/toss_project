import cv2
from ultralytics import YOLO
import numpy as np
from datetime import datetime
import os
from itertools import combinations
import math

# ===== CONFIGURATION =====
model = YOLO("best.pt")
stream_url = "rtsp://admin:phoenix0332@192.168.1.108:554/cam/realmonitor?channel=1&subtype=0"

PIXELS_PER_INCH = 186
PROXIMITY_THRESHOLD_PX = 25  # Increased to handle slight movement
CONFIDENCE_THRESHOLD = 0.3  # Higher than 0.2 to reduce noise
IOU_THRESHOLD = 0.5         # For NMS
BOX_SCALE_FACTOR = 1     # Scale box size by 1.5x for better visibility

SAVE_DIR = "shooting_sessions"
os.makedirs(SAVE_DIR, exist_ok=True)

# ===== SETUP CAMERA =====
cap = cv2.VideoCapture(stream_url)
if not cap.isOpened():
    print("ERROR: Cannot open RTSP stream")
    exit()

print("✓ System Ready.")
print("Press 's' to START a new session")
print("Press 'e' to END session (save + analyze)")
print("Press 'q' to QUIT")

all_bullet_holes = []  # All confirmed holes (from all past sessions)
session_new_holes = []  # ONLY truly new holes in current session
current_session = False

while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame read error.")
        continue

    key = cv2.waitKey(1) & 0xFF

    # === START NEW SESSION ===
    if key == ord('s'):
        current_session = True
        session_new_holes.clear()
        print("▶ New session started!")

    # === END CURRENT SESSION ===
    elif key == ord('e'):
        if len(session_new_holes) < 2:
            print("⚠ Not enough *new* bullet holes to compute distance.")
        else:
            max_dist_px = 0
            farthest_pair = None
            for (x1, y1), (x2, y2) in combinations(session_new_holes, 2):
                dist = math.hypot(x2 - x1, y2 - y1)
                if dist > max_dist_px:
                    max_dist_px = dist
                    farthest_pair = ((x1, y1), (x2, y2))

            max_dist_inch = max_dist_px / PIXELS_PER_INCH

            if farthest_pair:
                cv2.line(frame, farthest_pair[0], farthest_pair[1], (255, 0, 0), 2)
                cv2.putText(
                    frame,
                    f"Max Distance: {max_dist_inch:.2f} in",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                )

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(SAVE_DIR, f"session_{timestamp}.jpg")
            cv2.imwrite(save_path, frame)
            print(f"✅ Session saved: {save_path}")
            print(f"📏 Longest distance (NEW holes only): {max_dist_inch:.2f} inches")

        all_bullet_holes.extend(session_new_holes)
        current_session = False
        session_new_holes.clear()

    # === QUIT ===
    elif key == ord('q'):
        break

    # === OBJECT DETECTION WITH FILTERING ===
    results = model.predict(
        frame,
        conf=CONFIDENCE_THRESHOLD,   # Higher confidence
        iou=IOU_THRESHOLD,           # Apply NMS
        device=0,
        verbose=False
    )

    # Extract detections with center points
    current_detections_with_box = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        current_detections_with_box.append(((x1, y1, x2, y2), (cx, cy)))

    # === If session active: identify TRULY NEW holes ===
    if current_session:
        for (box_coords, (cx, cy)) in current_detections_with_box:
            is_new = True
            for (ox, oy) in (all_bullet_holes + session_new_holes):
                if math.hypot(cx - ox, cy - oy) < PROXIMITY_THRESHOLD_PX:
                    is_new = False
                    break
            if is_new:
                already_added = False
                for (nx, ny) in session_new_holes:
                    if math.hypot(cx - nx, cy - ny) < PROXIMITY_THRESHOLD_PX:
                        already_added = True
                        break
                if not already_added:
                    session_new_holes.append((cx, cy))

        cv2.putText(frame, "SESSION ACTIVE", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # === DRAW ALL HOLES WITH SCALED BOXES AND COLORS ===
    for (box_coords, (cx, cy)) in current_detections_with_box:
        x1, y1, x2, y2 = box_coords

        # Scale the box for better visibility
        w = x2 - x1
        h = y2 - y1
        cx_box = (x1 + x2) // 2
        cy_box = (y1 + y2) // 2
        new_w = int(w * BOX_SCALE_FACTOR)
        new_h = int(h * BOX_SCALE_FACTOR)
        scaled_x1 = max(0, cx_box - new_w // 2)
        scaled_y1 = max(0, cy_box - new_h // 2)
        scaled_x2 = min(frame.shape[1], cx_box + new_w // 2)
        scaled_y2 = min(frame.shape[0], cy_box + new_h // 2)

        # Check if old
        is_old = False
        for (ox, oy) in all_bullet_holes:
            if math.hypot(cx - ox, cy - oy) < PROXIMITY_THRESHOLD_PX:
                is_old = True
                break

        if is_old:
            cv2.rectangle(frame, (scaled_x1, scaled_y1), (scaled_x2, scaled_y2), (0, 0, 255), 3)  # Red, thicker line
        else:
            if current_session:
                is_new_in_session = False
                for (nx, ny) in session_new_holes:
                    if math.hypot(cx - nx, cy - ny) < PROXIMITY_THRESHOLD_PX:
                        is_new_in_session = True
                        break
                if is_new_in_session:
                    cv2.rectangle(frame, (scaled_x1, scaled_y1), (scaled_x2, scaled_y2), (0, 255, 0), 3)  # Green, thicker line

    cv2.imshow("Shooting Analysis", frame)

# === CLEANUP ===
cap.release()
cv2.destroyAllWindows()