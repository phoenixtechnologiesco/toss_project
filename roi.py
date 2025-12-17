# import cv2
# from ultralytics import YOLO
# from datetime import datetime
# import os
# from itertools import combinations
# import math

# # ===== CONFIGURATION =====
# model = YOLO("best.pt")
# stream_url = "rtsp://admin:phoenix0332@192.168.1.108:554/cam/realmonitor?channel=1&subtype=0"

# # Calibrate this using a ruler on your target within the ROI!
# PIXELS_PER_INCH = 186  # Adjust after ROI selection
# PROXIMITY_THRESHOLD_PX = 15

# # Detection confidence — increased to reduce false positives
# CONFIDENCE_THRESHOLD = 0.2  # was 0.2 → now 0.5
# IOU_THRESHOLD = 0.6        # tighter NMS

# # Box scaling for visibility
# BOX_SCALE_FACTOR = 1

# # Area filtering (in PIXELS²) — critical for noise reduction
# MIN_AREA = 60   # Adjust based on real bullet hole size in ROI
# MAX_AREA = 800  # Prevent large false positives

# # Display size for full-screen ROI view
# DISPLAY_WIDTH = 1500
# DISPLAY_HEIGHT = 775

# SAVE_DIR = "shooting_sessions"
# os.makedirs(SAVE_DIR, exist_ok=True)

# # ===== COLOR PALETTE (12 distinct colors) =====
# COLOR_PALETTE = [
#     (0, 255, 0),    # Green
#     (255, 0, 0),    # Blue
#     (0, 0, 255),    # Red
#     (255, 255, 0),  # Cyan
#     (255, 0, 255),  # Magenta
#     (0, 255, 255),  # Yellow
#     (128, 0, 128),  # Purple
#     (0, 128, 128),  # Teal
#     (128, 128, 0),  # Olive
#     (255, 165, 0),  # Orange
#     (255, 192, 203),# Pink
#     (165, 42, 42),  # Brown
# ]

# # ===== SESSION MANAGEMENT =====
# all_sessions = []  # Each: {"timestamp": str, "holes": [(cx,cy,x1,y1,x2,y2)], "color": (B,G,R)}
# current_session_index = -1

# # ===== CAMERA SETUP =====
# cap = cv2.VideoCapture(stream_url)
# if not cap.isOpened():
#     print("❌ ERROR: Cannot open RTSP stream")
#     exit()

# print("📸 Initializing camera...")

# # Read several frames to stabilize RTSP buffer
# for _ in range(30):
#     ret, temp = cap.read()
#     if not ret:
#         continue

# # Get a clean frame for ROI selection
# ret, roi_frame = cap.read()
# if not ret or roi_frame is None:
#     print("❌ Failed to capture initial frame for ROI selection!")
#     cap.release()
#     exit()

# # ===== ROI SELECTION =====
# print("\n🎯 Please select your TARGET AREA:")
# print("   - Click and drag a rectangle around the paper target")
# print("   - Press ENTER to confirm")
# print("   - Press ESC to cancel (will use full frame)")

# cv2.namedWindow("Select Target Area", cv2.WINDOW_AUTOSIZE)
# roi = cv2.selectROI("Select Target Area", roi_frame, fromCenter=False)
# cv2.destroyWindow("Select Target Area")

# if roi[2] <= 10 or roi[3] <= 10:  # too small
#     print("⚠ No valid ROI selected. Using FULL FRAME.")
#     USE_ROI = False
#     roi = None
# else:
#     USE_ROI = True
#     roi_x, roi_y, roi_w, roi_h = roi
#     print(f"✅ ROI selected: x={roi_x}, y={roi_y}, w={roi_w}, h={roi_h}")
#     # Recalibrate PIXELS_PER_INCH if needed (optional: add ruler measurement here)

# print("\n🚀 System Ready!")
# print("Press 's' → Start new shooter session")
# print("Press 'e' → End session & save analysis")
# print("Press 'q' → Quit & save final summary")

# def get_all_existing_centers():
#     centers = []
#     for s in all_sessions:
#         for entry in s["holes"]:
#             centers.append((entry[0], entry[1]))  # (cx, cy)
#     return centers

# # ===== MAIN LOOP =====
# while True:
#     ret, full_frame = cap.read()
#     if not ret or full_frame is None or full_frame.size == 0:
#         print("⚠ Frame read error. Skipping...")
#         continue

#     # Apply ROI crop
#     if USE_ROI:
#         # Ensure ROI is within bounds
#         roi_x = max(0, min(roi_x, full_frame.shape[1] - 1))
#         roi_y = max(0, min(roi_y, full_frame.shape[0] - 1))
#         roi_w = min(roi_w, full_frame.shape[1] - roi_x)
#         roi_h = min(roi_h, full_frame.shape[0] - roi_y)
        
#         if roi_w <= 0 or roi_h <= 0:
#             frame_for_yolo = full_frame.copy()
#             display_frame = full_frame.copy()
#         else:
#             frame_for_yolo = full_frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
#             display_frame = frame_for_yolo.copy()
#     else:
#         frame_for_yolo = full_frame.copy()
#         display_frame = full_frame.copy()

#     # Safety check
#     if frame_for_yolo.size == 0:
#         print("⚠ Empty processing frame!")
#         continue

#     key = cv2.waitKey(1) & 0xFF

#     # === START SESSION ===
#     if key == ord('s'):
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         color = COLOR_PALETTE[len(all_sessions) % len(COLOR_PALETTE)]
#         all_sessions.append({
#             "timestamp": timestamp,
#             "holes": [],
#             "color": color
#         })
#         current_session_index = len(all_sessions) - 1
#         print(f"▶ New shooter session started! ID: {len(all_sessions)} | Color: {color[::-1]} (RGB)")

#     # === END SESSION ===
#     elif key == ord('e'):
#         if current_session_index == -1:
#             print("⚠ No active session.")
#         else:
#             session = all_sessions[current_session_index]
#             holes = session["holes"]
#             if len(holes) < 2:
#                 print("⚠ Not enough hits to compute group size.")
#                 max_dist_inch = 0.0
#             else:
#                 centers = [(cx, cy) for (cx, cy, _, _, _, _) in holes]
#                 max_dist_px = 0
#                 farthest_pair = None
#                 for (x1, y1), (x2, y2) in combinations(centers, 2):
#                     dist = math.hypot(x2 - x1, y2 - y1)
#                     if dist > max_dist_px:
#                         max_dist_px = dist
#                         farthest_pair = ((x1, y1), (x2, y2))
#                 max_dist_inch = max_dist_px / PIXELS_PER_INCH

#                 if farthest_pair:
#                     cv2.line(display_frame, farthest_pair[0], farthest_pair[1], (255, 255, 255), 2)
#                     cv2.putText(
#                         display_frame,
#                         f"Group Size: {max_dist_inch:.2f} in",
#                         (10, 30),
#                         cv2.FONT_HERSHEY_SIMPLEX,
#                         0.8,
#                         (255, 255, 255),
#                         2,
#                     )

#             save_path = os.path.join(SAVE_DIR, f"session_{session['timestamp']}.jpg")
#             cv2.imwrite(save_path, display_frame)
#             print(f"✅ Session saved: {save_path}")
#             if len(holes) >= 2:
#                 print(f"🎯 Group size (this shooter): {max_dist_inch:.2f} inches")
#             current_session_index = -1

#     # === QUIT ===
#     elif key == ord('q'):
#         final_frame = display_frame.copy()
#         for sess in all_sessions:
#             color = sess["color"]
#             for (cx, cy, x1, y1, x2, y2) in sess["holes"]:
#                 cv2.rectangle(final_frame, (x1, y1), (x2, y2), color, 3)
#         final_save = os.path.join(SAVE_DIR, f"final_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
#         cv2.imwrite(final_save, final_frame)
#         print(f"📸 Final summary saved: {final_save}")
#         break

#     # === YOLO DETECTION (on cropped frame) ===
#     try:
#         results = model.predict(
#             frame_for_yolo,
#             conf=CONFIDENCE_THRESHOLD,
#             iou=IOU_THRESHOLD,
#             device=0,
#             verbose=False
#         )
#     except Exception as e:
#         print(f"⚠ YOLO error: {e}")
#         continue

#     current_detections = []
#     for box in results[0].boxes:
#         x1, y1, x2, y2 = map(int, box.xyxy[0])
#         if x2 <= x1 or y2 <= y1:
#             continue

#         w = x2 - x1
#         h = y2 - y1
#         area = w * h

#         # 🔥 CRITICAL: Filter by area to remove noise
#         if area < MIN_AREA or area > MAX_AREA:
#             continue

#         cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

#         # Scale box for better visibility
#         cx_box, cy_box = (x1 + x2) // 2, (y1 + y2) // 2
#         new_w = int(w * BOX_SCALE_FACTOR)
#         new_h = int(h * BOX_SCALE_FACTOR)
#         sx1 = max(0, cx_box - new_w // 2)
#         sy1 = max(0, cy_box - new_h // 2)
#         sx2 = min(frame_for_yolo.shape[1], cx_box + new_w // 2)
#         sy2 = min(frame_for_yolo.shape[0], cy_box + new_h // 2)

#         current_detections.append((cx, cy, sx1, sy1, sx2, sy2))

#     # === ADD NEW HOLES TO CURRENT SESSION ===
#     if current_session_index != -1:
#         session = all_sessions[current_session_index]
#         all_existing = get_all_existing_centers()

#         for (cx, cy, sx1, sy1, sx2, sy2) in current_detections:
#             # Skip if too close to any existing hole (any session)
#             if any(math.hypot(cx - ox, cy - oy) < PROXIMITY_THRESHOLD_PX for (ox, oy) in all_existing):
#                 continue
#             # Skip if already in current session
#             if any(math.hypot(cx - scx, cy - scy) < PROXIMITY_THRESHOLD_PX for (scx, scy, *_ ) in session["holes"]):
#                 continue
#             session["holes"].append((cx, cy, sx1, sy1, sx2, sy2))
#             print(f"  ➕ New hit for Shooter #{current_session_index + 1}")

#         cv2.putText(display_frame, f"SHOOTER {current_session_index + 1} ACTIVE", (10, 60),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

#     # === DRAW ALL BOXES ===
#     for sess in all_sessions:
#         color = sess["color"]
#         for (cx, cy, x1, y1, x2, y2) in sess["holes"]:
#             cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 1)

#     # === RESIZE FOR FULL-SCREEN DISPLAY ===
#     display_resized = cv2.resize(display_frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT), interpolation=cv2.INTER_LINEAR)
    
#     # Add title
#     cv2.putText(display_resized, "SHOOTING ANALYSIS", (10, 30),
#                 cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

#     cv2.imshow("Shooting Analysis - Target View", display_resized)

# # ===== CLEANUP =====
# cap.release()
# cv2.destroyAllWindows()






import cv2
from ultralytics import YOLO
from datetime import datetime
import os
from itertools import combinations
import math

# ===== CONFIGURATION =====
model = YOLO("best.pt")
stream_url = "rtsp://admin:phoenix0332@192.168.1.108:554/cam/realmonitor?channel=1&subtype=0"

# Calibrate this using a ruler on your target within the ROI!
PIXELS_PER_INCH = 186  # Adjust after ROI selection

# Detection confidence — increased to reduce false positives
CONFIDENCE_THRESHOLD = 0.4  # Slightly more permissive
IOU_THRESHOLD = 0.6        # Permissive NMS → allows close holes

# Area filtering (in PIXELS²)
MIN_AREA = 30   # For tiny holes
MAX_AREA = 5000  # Prevent large false positives

# Display size for full-screen ROI view
DISPLAY_WIDTH = 1500
DISPLAY_HEIGHT = 775

SAVE_DIR = "shooting_sessions"
os.makedirs(SAVE_DIR, exist_ok=True)

# ===== COLOR PALETTE (12 distinct colors) =====
COLOR_PALETTE = [
    (0, 255, 0),    # Green
    (255, 0, 0),    # Blue
    (0, 0, 255),    # Red
    (255, 255, 0),  # Cyan
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Yellow
    (128, 0, 128),  # Purple
    (0, 128, 128),  # Teal
    (128, 128, 0),  # Olive
    (255, 165, 0),  # Orange
    (255, 192, 203),# Pink
    (165, 42, 42),  # Brown
]

# ===== SESSION MANAGEMENT =====
all_sessions = []  # Each: {"timestamp": str, "holes": [(cx,cy,x1,y1,x2,y2)], "color": (B,G,R)}
current_session_index = -1

# ===== CAMERA SETUP =====
cap = cv2.VideoCapture(stream_url)
if not cap.isOpened():
    print("❌ ERROR: Cannot open RTSP stream")
    exit()

print("📸 Initializing camera...")

# Read several frames to stabilize RTSP buffer
for _ in range(30):
    ret, temp = cap.read()
    if not ret:
        continue

# Get a clean frame for ROI selection
ret, roi_frame = cap.read()
if not ret or roi_frame is None:
    print("❌ Failed to capture initial frame for ROI selection!")
    cap.release()
    exit()

# ===== ROI SELECTION =====
print("\n🎯 Please select your TARGET AREA:")
print("   - Click and drag a rectangle around the paper target")
print("   - Press ENTER to confirm")
print("   - Press ESC to cancel (will use full frame)")

cv2.namedWindow("Select Target Area", cv2.WINDOW_AUTOSIZE)
roi = cv2.selectROI("Select Target Area", roi_frame, fromCenter=False)
cv2.destroyWindow("Select Target Area")

if roi[2] <= 10 or roi[3] <= 10:  # too small
    print("⚠ No valid ROI selected. Using FULL FRAME.")
    USE_ROI = False
    roi = None
else:
    USE_ROI = True
    roi_x, roi_y, roi_w, roi_h = roi
    print(f"✅ ROI selected: x={roi_x}, y={roi_y}, w={roi_w}, h={roi_h}")

print("\n🚀 System Ready!")
print("Press 's' → Start new shooter session")
print("Press 'e' → End session & save analysis")
print("Press 'q' → Quit & save final summary")

# ===== HELPER FUNCTIONS =====
def calculate_iou(box1, box2):
    """Calculate IoU between two boxes (x1, y1, x2, y2)."""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    inter_x1 = max(x1_1, x1_2)
    inter_y1 = max(y1_1, y1_2)
    inter_x2 = min(x2_1, x2_2)
    inter_y2 = min(y2_1, y2_2)

    if inter_x1 >= inter_x2 or inter_y1 >= inter_y2:
        return 0.0

    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = area1 + area2 - inter_area

    return inter_area / union_area

def get_all_existing_holes():
    """Returns list of all holes across all sessions as (cx, cy, x1, y1, x2, y2)"""
    holes = []
    for s in all_sessions:
        for entry in s["holes"]:
            holes.append(entry)
    return holes

# ===== MAIN LOOP =====
while True:
    ret, full_frame = cap.read()
    if not ret or full_frame is None or full_frame.size == 0:
        print("⚠ Frame read error. Skipping...")
        continue

    # Apply ROI crop
    if USE_ROI:
        roi_x = max(0, min(roi_x, full_frame.shape[1] - 1))
        roi_y = max(0, min(roi_y, full_frame.shape[0] - 1))
        roi_w = min(roi_w, full_frame.shape[1] - roi_x)
        roi_h = min(roi_h, full_frame.shape[0] - roi_y)
        
        if roi_w <= 0 or roi_h <= 0:
            frame_for_yolo = full_frame.copy()
            display_frame = full_frame.copy()
        else:
            frame_for_yolo = full_frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
            display_frame = frame_for_yolo.copy()
    else:
        frame_for_yolo = full_frame.copy()
        display_frame = full_frame.copy()

    if frame_for_yolo.size == 0:
        print("⚠ Empty processing frame!")
        continue

    key = cv2.waitKey(1) & 0xFF

    # === START SESSION ===
    if key == ord('s'):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        color = COLOR_PALETTE[len(all_sessions) % len(COLOR_PALETTE)]
        all_sessions.append({
            "timestamp": timestamp,
            "holes": [],
            "color": color
        })
        current_session_index = len(all_sessions) - 1
        print(f"▶ New shooter session started! ID: {len(all_sessions)} | Color: {color[::-1]} (RGB)")

    # === END SESSION ===
    elif key == ord('e'):
        if current_session_index == -1:
            print("⚠ No active session.")
        else:
            session = all_sessions[current_session_index]
            holes = session["holes"]
            total_hits = len(holes)
            max_dist_inch = 0.0

            if total_hits == 0:
                print("⚠ No hits detected in this session.")
            else:
                print(f"🎯 Session ended: {total_hits} hit(s) detected.")

                if total_hits >= 2:
                    centers = [(cx, cy) for (cx, cy, _, _, _, _) in holes]
                    max_dist_px = 0
                    farthest_pair = None
                    for (x1, y1), (x2, y2) in combinations(centers, 2):
                        dist = math.hypot(x2 - x1, y2 - y1)
                        if dist > max_dist_px:
                            max_dist_px = dist
                            farthest_pair = ((x1, y1), (x2, y2))
                    max_dist_inch = max_dist_px / PIXELS_PER_INCH

                    if farthest_pair:
                        cv2.line(display_frame, farthest_pair[0], farthest_pair[1], (255, 255, 255), 1)
                
                # === NEW: Display total hits + group size ===
                info_text = f"Hits: {total_hits}"
                if total_hits >= 2:
                    info_text += f" | Group: {max_dist_inch:.2f} in"
                
                cv2.putText(
                    display_frame,
                    info_text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                )
                print(f"🎯 Final stats → Hits: {total_hits}, Group size: {max_dist_inch:.2f} inches")

            save_path = os.path.join(SAVE_DIR, f"session_{session['timestamp']}.jpg")
            cv2.imwrite(save_path, display_frame)
            print(f"✅ Session saved: {save_path}")
            current_session_index = -1

    # === QUIT ===
    elif key == ord('q'):
        final_frame = display_frame.copy()
        for sess in all_sessions:
            color = sess["color"]
            for (cx, cy, x1, y1, x2, y2) in sess["holes"]:
                cv2.rectangle(final_frame, (x1, y1), (x2, y2), color, 1)
                cv2.circle(final_frame, (cx, cy), 2, color, -1)
        final_save = os.path.join(SAVE_DIR, f"final_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
        cv2.imwrite(final_save, final_frame)
        print(f"📸 Final summary saved: {final_save}")
        break

    # === YOLO DETECTION (on cropped frame) ===
    try:
        results = model.predict(
            frame_for_yolo,
            conf=CONFIDENCE_THRESHOLD,
            iou=IOU_THRESHOLD,
            device=0,
            verbose=False
        )
    except Exception as e:
        print(f"⚠ YOLO error: {e}")
        continue

    current_detections = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        if x2 <= x1 or y2 <= y1:
            continue

        w = x2 - x1
        h = y2 - y1
        area = w * h

        # 🔥 AREA FILTER
        if area < MIN_AREA or area > MAX_AREA:
            continue

        # 🔍 SHAPE FILTER: Bullet holes are roughly circular
        aspect_ratio = max(w, h) / min(w, h)
        if aspect_ratio > 2.0:  # Too rectangular → likely not a bullet hole
            continue

        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        # ✅ USE ORIGINAL BOX (NO SCALING) FOR ACCURACY
        sx1, sy1, sx2, sy2 = x1, y1, x2, y2

        current_detections.append((cx, cy, sx1, sy1, sx2, sy2))

    # === ADD NEW HOLES TO CURRENT SESSION ===
    if current_session_index != -1:
        session = all_sessions[current_session_index]
        all_existing_holes_list = get_all_existing_holes()

        for (cx, cy, sx1, sy1, sx2, sy2) in current_detections:
            # Check against ALL existing holes (any session) using IoU + Center Proximity
            is_duplicate = False
            for (ex_cx, ex_cy, ex_x1, ex_y1, ex_x2, ex_y2) in all_existing_holes_list:
                curr_box = (sx1, sy1, sx2, sy2)
                existing_box = (ex_x1, ex_y1, ex_x2, ex_y2)
                iou = calculate_iou(curr_box, existing_box)
                center_dist = math.hypot(cx - ex_cx, cy - ex_cy)
                if iou > 0.9 or center_dist < 10:  # 90% overlap OR <10px apart
                    is_duplicate = True
                    break

            if is_duplicate:
                continue

            # Check against holes in CURRENT session only
            already_in_session = False
            for (scx, scy, sc_x1, sc_y1, sc_x2, sc_y2) in session["holes"]:
                curr_box = (sx1, sy1, sx2, sy2)
                sess_box = (sc_x1, sc_y1, sc_x2, sc_y2)
                iou = calculate_iou(curr_box, sess_box)
                center_dist = math.hypot(cx - scx, cy - scy)
                if iou > 0.9 or center_dist < 10:
                    already_in_session = True
                    break

            if already_in_session:
                continue

            session["holes"].append((cx, cy, sx1, sy1, sx2, sy2))
            print(f"  ➕ New hit for Shooter #{current_session_index + 1}")

        cv2.putText(display_frame, f"SHOOTER {current_session_index + 1} ACTIVE", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # === DRAW ALL BOXES ===
    for sess in all_sessions:
        color = sess["color"]
        for (cx, cy, x1, y1, x2, y2) in sess["holes"]:
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 1)  # Thin border
            cv2.circle(display_frame, (cx, cy), 2, color, -1)           # Center dot

    # === RESIZE FOR FULL-SCREEN DISPLAY ===
    display_resized = cv2.resize(display_frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT), interpolation=cv2.INTER_LINEAR)
    cv2.putText(display_resized, "SHOOTING ANALYSIS", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Shooting Analysis - Target View", display_resized)

# ===== CLEANUP =====
cap.release()
cv2.destroyAllWindows()