# backend.py
import cv2
import numpy as np
import os
import math
from datetime import datetime
from itertools import combinations
from ultralytics import YOLO

class ShootingAnalyzer:
    def __init__(self, model_path="best.pt", stream_url="rtsp://admin:phoenix0332@192.168.1.108:554/cam/realmonitor?channel=1&subtype=0"):
        # ===== CONFIGURATION =====
        self.model_path = model_path
        self.stream_url = stream_url
        self.model = YOLO(self.model_path) if os.path.exists(self.model_path) else None

        self.PIXELS_PER_INCH = 186
        self.CONFIDENCE_THRESHOLD = 0.2
        self.IOU_THRESHOLD = 0.6
        self.MIN_AREA = 20
        self.MAX_AREA = 800
        self.SAVE_DIR = "shooting_sessions"
        os.makedirs(self.SAVE_DIR, exist_ok=True)

        # ===== COLOR PALETTE =====
        self.COLOR_PALETTE = [
            (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 128), (0, 128, 128),
            (128, 128, 0), (255, 165, 0), (255, 192, 203), (165, 42, 42),
        ]

        # ===== STATE =====
        self.cap = cv2.VideoCapture(self.stream_url)
        if not self.cap.isOpened():
            raise RuntimeError("❌ Cannot open RTSP stream")

        self.use_roi = False
        self.roi_x = self.roi_y = self.roi_w = self.roi_h = 0
        self.all_sessions = []
        self.current_session_index = -1
        self.last_full_frame = None
        self.last_display_frame = None

        # Stabilize stream
        for _ in range(30):
            self.cap.read()

    def select_roi(self):
        """Select ROI and return success status"""
        ret, roi_frame = self.cap.read()
        if not ret or roi_frame is None:
            return False, "❌ Failed to capture frame for ROI selection!"

        cv2.namedWindow("Select Target Area", cv2.WINDOW_AUTOSIZE)
        roi = cv2.selectROI("Select Target Area", roi_frame, fromCenter=False)
        cv2.destroyWindow("Select Target Area")

        if roi[2] <= 10 or roi[3] <= 10:
            self.use_roi = False
            return True, "⚠ No valid ROI selected. Using FULL FRAME."
        else:
            self.use_roi = True
            self.roi_x, self.roi_y, self.roi_w, self.roi_h = roi
            return True, f"✅ ROI selected: x={self.roi_x}, y={self.roi_y}, w={self.roi_w}, h={self.roi_h}"

    def start_session(self, name="Shooter", service_no="N/A", service_year="N/A"):
        """Start a new session"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        color = self.COLOR_PALETTE[len(self.all_sessions) % len(self.COLOR_PALETTE)]
        self.all_sessions.append({
            "name": name,
            "service_no": service_no,        # ← Add this
            "service_year": service_year,    # ← Add this
            "timestamp": timestamp,
            "holes": [],
            "color": color
        })
        self.current_session_index = len(self.all_sessions) - 1
        return True

    def stop_session(self):
        """End current session, return stats, save image with group line"""
        if self.current_session_index == -1:
            return {"success": False, "message": "⚠ No active session."}

        session = self.all_sessions[self.current_session_index]
        holes = session["holes"]
        total_hits = len(holes)
        max_dist_inch = 0.0
        # Get the display frame (already has boxes, line, text)
        display_frame = self.get_display_frame()

        if total_hits == 0:
            message = "⚠ No hits detected."
        else:
            if total_hits >= 2:
                centers = [(h[0], h[1]) for h in holes]
                max_dist_px = max(math.hypot(x2 - x1, y2 - y1) for (x1, y1), (x2, y2) in combinations(centers, 2))
                max_dist_inch = max_dist_px / self.PIXELS_PER_INCH
                # Draw line on display frame
                farthest = max(combinations(centers, 2), key=lambda p: math.hypot(p[1][0]-p[0][0], p[1][1]-p[0][1]))
                cv2.line(display_frame, farthest[0], farthest[1], (255, 255, 255), 1)

            # Add text
            info_text = f"Hits: {total_hits} | Group: {max_dist_inch:.2f} in"
            cv2.putText(display_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # Save image
            # ✅ Use SN_NAME_SY format
            sn = session.get('service_no', 'N/A')
            name = session.get('name', 'N/A')
            sy = session.get('service_year', 'N/A')
            save_path = os.path.join(self.SAVE_DIR, f"{sn}_{name}_{sy}.jpg")
            cv2.imwrite(save_path, display_frame)
            session["LastImagePath"] = save_path

            message = f"✅ Session saved. Hits: {total_hits}, Group: {max_dist_inch:.2f} inches"

        session["group_size"] = max_dist_inch
        self.current_session_index = -1
        return {"success": True, "hits": total_hits, "group_size": max_dist_inch, "message": message}

    def get_frame(self):
        """Get raw full frame from camera"""
        ret, frame = self.cap.read()
        if ret:
            self.last_full_frame = frame.copy()
            return frame
        return self.last_full_frame

    def get_display_frame(self):
        """Get ROI-cropped + annotated frame for display"""
        if self.last_full_frame is None:
            return np.zeros((480, 640, 3), dtype=np.uint8)

        frame = self.last_full_frame.copy()
        if self.use_roi:
            frame = frame[self.roi_y:self.roi_y+self.roi_h, self.roi_x:self.roi_x+self.roi_w]

        # Draw all holes
        for sess in self.all_sessions:
            for (cx, cy, x1, y1, x2, y2) in sess["holes"]:
                cv2.rectangle(frame, (x1, y1), (x2, y2), sess["color"], 1)
                cv2.circle(frame, (cx, cy), 2, sess["color"], -1)

        if self.current_session_index != -1:
            name = self.all_sessions[self.current_session_index]["name"]
            cv2.putText(frame, f"{name} ACTIVE", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        return frame

    def process_frame(self):
        """Run YOLO detection and update current session"""
        if not self.model or self.current_session_index == -1:
            return

        frame = self.get_frame()
        if frame is None:
            return

        if self.use_roi:
            frame = frame[self.roi_y:self.roi_y+self.roi_h, self.roi_x:self.roi_x+self.roi_w]

        try:
            results = self.model.predict(frame, conf=self.CONFIDENCE_THRESHOLD, iou=self.IOU_THRESHOLD, verbose=False)
            current_detections = []
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                if x2 <= x1 or y2 <= y1:
                    continue
                w, h = x2 - x1, y2 - y1
                area = w * h
                if area < self.MIN_AREA or area > self.MAX_AREA:
                    continue
                aspect_ratio = max(w, h) / min(w, h)
                if aspect_ratio > 2.0:
                    continue
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                current_detections.append((cx, cy, x1, y1, x2, y2))

            session = self.all_sessions[self.current_session_index]
            all_existing = self.get_all_existing_holes()
            for (cx, cy, x1, y1, x2, y2) in current_detections:
                # Check against all existing holes
                is_dup = False
                for (ex_cx, ex_cy, ex_x1, ex_y1, ex_x2, ex_y2) in all_existing:
                    iou = self.calculate_iou((x1, y1, x2, y2), (ex_x1, ex_y1, ex_x2, ex_y2))
                    center_dist = math.hypot(cx - ex_cx, cy - ex_cy)
                    if iou > 0.9 or center_dist < 10:
                        is_dup = True
                        break
                if is_dup:
                    continue
                # Check against current session
                already_in = False
                for (scx, scy, sx1, sy1, sx2, sy2) in session["holes"]:
                    iou = self.calculate_iou((x1, y1, x2, y2), (sx1, sy1, sx2, sy2))
                    center_dist = math.hypot(cx - scx, cy - scy)
                    if iou > 0.9 or center_dist < 10:
                        already_in = True
                        break
                if not already_in:
                    session["holes"].append((cx, cy, x1, y1, x2, y2))
        except Exception as e:
            print(f"⚠ YOLO error: {e}")

    def get_all_existing_holes(self):
        holes = []
        for s in self.all_sessions:
            for entry in s["holes"]:
                holes.append(entry)
        return holes

    def calculate_iou(self, box1, box2):
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

    def save_final_summary(self):
        """Save final image with all sessions in full frame"""
        if self.last_full_frame is None:
            return None
        final_frame = self.last_full_frame.copy()
        for sess in self.all_sessions:
            color = sess["color"]
            for (cx, cy, x1, y1, x2, y2) in sess["holes"]:
                fx1 = x1 + self.roi_x if self.use_roi else x1
                fy1 = y1 + self.roi_y if self.use_roi else y1
                fx2 = x2 + self.roi_x if self.use_roi else x2
                fy2 = y2 + self.roi_y if self.use_roi else y2
                fcx = cx + self.roi_x if self.use_roi else cx
                fcy = cy + self.roi_y if self.use_roi else cy
                cv2.rectangle(final_frame, (fx1, fy1), (fx2, fy2), color, 1)
                cv2.circle(final_frame, (fcx, fcy), 2, color, -1)
        path = os.path.join(self.SAVE_DIR, f"final_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
        cv2.imwrite(path, final_frame)
        return path

    def release(self):
        if self.cap:
            self.cap.release()