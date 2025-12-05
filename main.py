# # this code detects real time bullets only

# import cv2
# from ultralytics import YOLO

# model = YOLO("best.pt")

# stream_url = "rtsp://admin:phoenix0332@192.168.1.108:554/cam/realmonitor?channel=1&subtype=0"
# cap = cv2.VideoCapture(stream_url)

# if not cap.isOpened():
#     print("ERROR: Cannot open RTSP stream")
#     exit()

# print("✓ Running Real-Time Detection...")

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("Frame read error.")
#         continue

    
#     results = model.predict(
#         frame,
#         conf=0.3,        
#         iou=0.45,
#         device=0,         
#         verbose=False
#     )

#     for box in results[0].boxes:
#         x1, y1, x2, y2 = map(int, box.xyxy[0])
#         cls = int(box.cls)
#         conf = float(box.conf)

#         label = f"Bullet Hole {conf:.2f}"
#         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
#         cv2.putText(frame, label, (x1, y1 - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

#     cv2.imshow("Real-Time Detection", frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()







# this code detects real time bullet holes also find longest distance between any 2 points

# import cv2
# from ultralytics import YOLO
# import numpy as np
# from datetime import datetime
# import os
# from itertools import combinations
# import math

# # ===== CONFIGURATION =====
# model = YOLO("best.pt")
# stream_url = "rtsp://admin:phoenix0332@192.168.1.108:554/cam/realmonitor?channel=1&subtype=0"

# # CALIBRATION: Adjust based on your setup!
# # Example: 100 pixels = 1 inch (measure this physically!)
# PIXELS_PER_INCH = 186  # <-- YOU MUST SET THIS!

# SAVE_DIR = "shooting_sessions"
# os.makedirs(SAVE_DIR, exist_ok=True)

# # ===== SETUP CAMERA =====
# cap = cv2.VideoCapture(stream_url)
# if not cap.isOpened():
#     print("ERROR: Cannot open RTSP stream")
#     exit()

# print("✓ System Ready.")
# print("Press 's' to START a new session")
# print("Press 'e' to END session (save + analyze)")
# print("Press 'q' to QUIT")

# current_session = False
# bullet_holes = []  # List of (cx, cy) during current session

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("Frame read error.")
#         continue

#     key = cv2.waitKey(1) & 0xFF

#     # === START NEW SESSION ===
#     if key == ord('s'):
#         current_session = True
#         bullet_holes.clear()
#         print("▶ New session started!")

#     # === END CURRENT SESSION ===
#     elif key == ord('e'):
#         if len(bullet_holes) < 2:
#             print("⚠ Not enough bullet holes to compute distance.")
#         else:
#             # Compute max distance between any two holes
#             max_dist_px = 0
#             pair = None
#             for (x1, y1), (x2, y2) in combinations(bullet_holes, 2):
#                 dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
#                 if dist > max_dist_px:
#                     max_dist_px = dist
#                     pair = ((x1, y1), (x2, y2))

#             max_dist_inch = max_dist_px / PIXELS_PER_INCH

#             # Draw line between farthest holes
#             if pair:
#                 cv2.line(frame, pair[0], pair[1], (255, 0, 0), 2)
#                 label = f"Max Distance: {max_dist_inch:.2f} in"
#                 cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

#             # Save frame
#             timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#             save_path = os.path.join(SAVE_DIR, f"session_{timestamp}.jpg")
#             cv2.imwrite(save_path, frame)
#             print(f"✅ Session saved: {save_path}")
#             print(f"📏 Longest distance: {max_dist_inch:.2f} inches")

#         current_session = False
#         bullet_holes.clear()

#     # === QUIT ===
#     elif key == ord('q'):
#         break

#     # === DETECTION (only during active session) ===
#     if current_session:
#         results = model.predict(
#             frame,
#             conf=0.2,
#             iou=0.45,
#             device=0,
#             verbose=False
#         )

#         current_frame_holes = []
#         for box in results[0].boxes:
#             x1, y1, x2, y2 = map(int, box.xyxy[0])
#             cls = int(box.cls)
#             conf = float(box.conf)

#             # Compute center (more stable for distance)
#             cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
#             current_frame_holes.append((cx, cy))

#             label = f"Bullet Hole {conf:.2f}"
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
#             cv2.putText(frame, label, (x1, y1 - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

#         # Update bullet_holes (optional: add only new holes)
#         # For simplicity, we replace per-frame (YOLO may flicker)
#         # Alternatively, use tracking (e.g., SORT) for robustness
#         bullet_holes = current_frame_holes.copy()

#         cv2.putText(frame, "SESSION ACTIVE", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

#     cv2.imshow("Shooting Analysis", frame)

# # === CLEANUP ===
# cap.release()
# cv2.destroyAllWindows()