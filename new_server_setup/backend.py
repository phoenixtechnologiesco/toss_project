# # backend.py
# import sys
# import os
# import cv2
# import math
# import ctypes
# import numpy as np
# from datetime import datetime
# from itertools import combinations
# from ultralytics import YOLO

# sys.path.append(
#     r"C:\Program Files (x86)\MVS\Development\Samples\Python\MvImport"
# )
# from MvCameraControl_class import *

# class ShootingAnalyzer:
#     def __init__(self):
#         BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#         self.model_path = os.path.normpath(
#             os.path.join(BASE_DIR, "..", "models", "new_best.pt")
#         )
#         self.model = YOLO(self.model_path)

#         self.PIXELS_PER_INCH = 186
#         self.CONFIDENCE_THRESHOLD = 0.3
#         self.IOU_THRESHOLD = 0.5
#         self.MIN_AREA = 80
#         self.MAX_AREA = 2000

#         self.SAVE_DIR = "shooting_sessions"
#         os.makedirs(self.SAVE_DIR, exist_ok=True)

#         self.COLOR_PALETTE = [
#             (0, 255, 0),      # Bright Green
#             (255, 0, 0),      # Bright Red
#             (0, 0, 255),      # Bright Blue
#             (255, 255, 0),    # Yellow (high visibility)
#             (255, 0, 255),    # Magenta
#             (0, 255, 255),    # Cyan
#             (128, 0, 128),    # Purple
#             (255, 165, 0),    # Orange (great for sunlight)
#         ]

#         self.all_sessions = []
#         self.current_session_index = -1
#         self.last_full_frame = None

#         self._init_camera()

#     # -------------------------------------------------
#     # CAMERA
#     # -------------------------------------------------
#     def _init_camera(self):
#         device_list = MV_CC_DEVICE_INFO_LIST()
#         MvCamera.MV_CC_EnumDevices(MV_GIGE_DEVICE, device_list)
#         self.cam = MvCamera()

#         st_info = ctypes.cast(
#             device_list.pDeviceInfo[0],
#             ctypes.POINTER(MV_CC_DEVICE_INFO)
#         ).contents

#         self.cam.MV_CC_CreateHandle(st_info)
#         self.cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
#         self.cam.MV_CC_StartGrabbing()

#         st_param = MVCC_INTVALUE()
#         self.cam.MV_CC_GetIntValue("PayloadSize", st_param)
#         self.payload_size = st_param.nCurValue
#         self.data_buf = (ctypes.c_ubyte * self.payload_size)()

#     def _grab_frame(self):
#         frame_info = MV_FRAME_OUT_INFO_EX()
#         ret = self.cam.MV_CC_GetOneFrameTimeout(
#             self.data_buf, self.payload_size, frame_info, 1000
#         )
#         if ret != 0:
#             return None

#         img = np.frombuffer(self.data_buf, dtype=np.uint8)

#         if frame_info.enPixelType == PixelType_Gvsp_Mono8:
#             img = img.reshape((frame_info.nHeight, frame_info.nWidth))
#             img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
#         elif frame_info.enPixelType == PixelType_Gvsp_BGR8_Packed:
#             img = img.reshape((frame_info.nHeight, frame_info.nWidth, 3))
#         else:
#             return None

#         return img.copy()

#     # -------------------------------------------------
#     # SESSION
#     # -------------------------------------------------
#     def start_session(self, name, service_no, service_year):
#         self.all_sessions.append({
#             "name": name,
#             "service_no": service_no,
#             "service_year": service_year,
#             "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
#             "holes": [],
#             "color": self.COLOR_PALETTE[len(self.all_sessions) % len(self.COLOR_PALETTE)],
#             "max_pair": None,
#             "group_size": 0.0
#         })
#         self.current_session_index = len(self.all_sessions) - 1

#     def stop_session(self):
#         session = self.all_sessions[self.current_session_index]
#         holes = session["holes"]

#         max_px = 0
#         max_pair = None

#         if len(holes) >= 2:
#             for (x1,y1,*_), (x2,y2,*_) in combinations(holes, 2):
#                 d = math.hypot(x2-x1, y2-y1)
#                 if d > max_px:
#                     max_px = d
#                     max_pair = ((x1,y1),(x2,y2))

#         session["group_size"] = max_px / self.PIXELS_PER_INCH
#         session["max_pair"] = max_pair

#         frame = self.get_display_frame()
#         path = os.path.join(
#             self.SAVE_DIR,
#             f"{session['service_no']}_{session['name']}.jpg"
#         )
#         cv2.imwrite(path, frame)
#         session["LastImagePath"] = path

#         self.current_session_index = -1
#         return {
#             "success": True,
#             "hits": len(holes),
#             "group_size": session["group_size"]
#         }   

#     # -------------------------------------------------
#     # PROCESS FRAME - IMPROVED DUPLICATE DETECTION
#     # -------------------------------------------------
#     def process_frame(self):
#         frame = self._grab_frame()
#         if frame is None:
#             return

#         self.last_full_frame = frame.copy()

#         if self.current_session_index == -1:
#             return

#         session = self.all_sessions[self.current_session_index]
#         results = self.model.predict(frame, conf=self.CONFIDENCE_THRESHOLD, verbose=False)

#         for box in results[0].boxes:
#             x1, y1, x2, y2 = map(int, box.xyxy[0])
#             cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            
#             # Calculate width and height of the detected bullet hole
#             box_width = x2 - x1
#             box_height = y2 - y1
#             # Use the larger dimension as the threshold for duplicate detection
#             detection_size = max(box_width, box_height)
#             # Set threshold as 1.5 times the detection size to ensure no overlap
#             overlap_threshold = int(detection_size * 1.5)

#             # Check for duplicates across ALL sessions to prevent cross-session duplicates
#             is_duplicate = False
#             for existing_session in self.all_sessions:
#                 for (ex, ey, *_) in existing_session["holes"]:
#                     distance = math.hypot(cx - ex, cy - ey)
#                     if distance < overlap_threshold:
#                         is_duplicate = True
#                         break
#                 if is_duplicate:
#                     break

#             if not is_duplicate:
#                 # Add the new hole to the current session with its specific color
#                 session["holes"].append((cx, cy, x1, y1, x2, y2))

#     # -------------------------------------------------
#     # DISPLAY
#     # -------------------------------------------------
#     def get_display_frame(self):
#         if self.last_full_frame is None:
#             return np.zeros((480,640,3), dtype=np.uint8)

#         frame = self.last_full_frame.copy()

#         for session in self.all_sessions:
#             color = session["color"]

#             for (cx,cy,x1,y1,x2,y2) in session["holes"]:
#                 cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
#                 cv2.circle(frame,(cx,cy),2,color,-1)

#             if session["max_pair"]:
#                 (p1,p2) = session["max_pair"]
#                 cv2.line(frame, p1, p2, color, 2)

#         return frame

#     # -------------------------------------------------
#     # CLEANUP
#     # -------------------------------------------------
#     def save_final_summary(self):
#         frame = self.get_display_frame()
#         path = os.path.join(self.SAVE_DIR, "FINAL_SUMMARY.jpg")
#         cv2.imwrite(path, frame)

#     def release(self):
#         self.cam.MV_CC_StopGrabbing()
#         self.cam.MV_CC_CloseDevice()
#         self.cam.MV_CC_DestroyHandle()





# backend.py
import sys
import os
import cv2
import math
import ctypes
import numpy as np
from datetime import datetime
from itertools import combinations
from ultralytics import YOLO

sys.path.append(
    r"C:\Program Files (x86)\MVS\Development\Samples\Python\MvImport"
)
from MvCameraControl_class import *
# from MvCameraControl_class import * 

class ShootingAnalyzer:
    def __init__(self):
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.normpath(
            os.path.join(BASE_DIR, "..", "models", "new_best.pt")
        )
        self.model = YOLO(self.model_path)

        self.PIXELS_PER_INCH = 186
        self.CONFIDENCE_THRESHOLD = 0.3
        self.IOU_THRESHOLD = 0.5
        self.MIN_AREA = 30
        self.MAX_AREA = 2000

        self.SAVE_DIR = "shooting_sessions"
        os.makedirs(self.SAVE_DIR, exist_ok=True)

        self.COLOR_PALETTE = [
            (0, 255, 0),      # Bright Green
            (255, 0, 0),      # Bright Red
            (0, 0, 255),      # Bright Blue
            (255, 255, 0),    # Yellow (high visibility)
            (255, 0, 255),    # Magenta
            (0, 255, 255),    # Cyan
            (128, 0, 128),    # Purple
            (255, 165, 0),    # Orange (great for sunlight)
        ]

        self.all_sessions = []
        self.current_session_index = -1
        self.last_full_frame = None
        
        # Lazy camera initialization
        self.cam = None
        self.payload_size = 0
        self.data_buf = None
        self.camera_initialized = False

    # -------------------------------------------------
    # CAMERA - LAZY INITIALIZATION (NO SETTINGS)
    # -------------------------------------------------
    def _init_camera(self):
        if self.camera_initialized:
            return True
            
        try:
            device_list = MV_CC_DEVICE_INFO_LIST()
            ret = MvCamera.MV_CC_EnumDevices(MV_GIGE_DEVICE, device_list)
            if ret != 0 or device_list.nDeviceNum == 0:
                print("No camera found!")
                return False

            self.cam = MvCamera()

            st_info = ctypes.cast(
                device_list.pDeviceInfo[0],
                ctypes.POINTER(MV_CC_DEVICE_INFO)
            ).contents

            ret = self.cam.MV_CC_CreateHandle(st_info)
            if ret != 0:
                print("Failed to create camera handle")
                return False

            ret = self.cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
            if ret != 0:
                print("Failed to open camera device")
                return False

            self.cam.MV_CC_StartGrabbing()

            st_param = MVCC_INTVALUE()
            ret = self.cam.MV_CC_GetIntValue("PayloadSize", st_param)
            if ret != 0:
                print("Failed to get payload size")
                return False

            self.payload_size = st_param.nCurValue
            self.data_buf = (ctypes.c_ubyte * self.payload_size)()

            self.camera_initialized = True
            print("Camera initialized successfully!")
            return True
            
        except Exception as e:
            print(f"Camera initialization error: {e}")
            return False

    def _grab_frame(self):
        if not self.camera_initialized:
            if not self._init_camera():
                return None

        try:
            frame_info = MV_FRAME_OUT_INFO_EX()
            ret = self.cam.MV_CC_GetOneFrameTimeout(
                self.data_buf, self.payload_size, frame_info, 1000
            )
            if ret != 0:
                return None

            img = np.frombuffer(self.data_buf, dtype=np.uint8)

            if frame_info.enPixelType == PixelType_Gvsp_Mono8:
                img = img.reshape((frame_info.nHeight, frame_info.nWidth))
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif frame_info.enPixelType == PixelType_Gvsp_BGR8_Packed:
                img = img.reshape((frame_info.nHeight, frame_info.nWidth, 3))
            else:
                return None

            return img.copy()
        except Exception as e:
            print(f"Frame grab error: {e}")
            return None

    # -------------------------------------------------
    # SESSION - START/STOP
    # -------------------------------------------------
    def start_session(self, name, service_no, service_year):
        self.all_sessions.append({
            "name": name,
            "service_no": service_no,
            "service_year": service_year,
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "holes": [],
            "color": self.COLOR_PALETTE[len(self.all_sessions) % len(self.COLOR_PALETTE)],
            "max_pair": None,
            "group_size": 0.0
        })
        self.current_session_index = len(self.all_sessions) - 1

    def stop_session(self):
        if self.current_session_index == -1 or self.current_session_index >= len(self.all_sessions):
            return {"success": False, "message": "No active session"}

        session = self.all_sessions[self.current_session_index]
        holes = session["holes"]

        max_px = 0
        max_pair = None

        if len(holes) >= 2:
            for (x1,y1,*_), (x2,y2,*_) in combinations(holes, 2):
                d = math.hypot(x2-x1, y2-y1)
                if d > max_px:
                    max_px = d
                    max_pair = ((x1,y1),(x2,y2))

        session["group_size"] = max_px / self.PIXELS_PER_INCH
        session["max_pair"] = max_pair

        frame = self.get_display_frame()
        # Create filename with service year
        filename = f"{session['service_no']}_{session['name']}_{session['service_year']}.jpg"
        path = os.path.join(self.SAVE_DIR, filename)
        cv2.imwrite(path, frame)
        session["LastImagePath"] = path

        self.current_session_index = -1
        
        return {
            "success": True,
            "hits": len(holes),
            "group_size": session["group_size"]
        }

    # -------------------------------------------------
    # PROCESS FRAME - IMPROVED DUPLICATE DETECTION
    # -------------------------------------------------
    def process_frame(self):
        frame = self._grab_frame()
        if frame is None:
            return

        self.last_full_frame = frame.copy()

        if self.current_session_index == -1:
            return

        session = self.all_sessions[self.current_session_index]
        results = self.model.predict(frame, conf=self.CONFIDENCE_THRESHOLD, verbose=False)

        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            
            # Calculate width and height of the detected bullet hole
            box_width = x2 - x1
            box_height = y2 - y1
            # Use the larger dimension as the threshold for duplicate detection
            detection_size = max(box_width, box_height)
            # Set threshold as 1.5 times the detection size to ensure no overlap
            overlap_threshold = int(detection_size * 1.5)

            # Check for duplicates across ALL sessions to prevent cross-session duplicates
            is_duplicate = False
            for existing_session in self.all_sessions:
                for (ex, ey, *_) in existing_session["holes"]:
                    distance = math.hypot(cx - ex, cy - ey)
                    if distance < overlap_threshold:
                        is_duplicate = True
                        break
                if is_duplicate:
                    break

            if not is_duplicate:
                # Add the new hole to the current session with its specific color
                session["holes"].append((cx, cy, x1, y1, x2, y2))

    # -------------------------------------------------
    # DISPLAY
    # -------------------------------------------------
    def get_display_frame(self):
        if self.last_full_frame is None:
            # Return a blank frame if no camera feed yet
            return np.zeros((720, 1280, 3), dtype=np.uint8)

        frame = self.last_full_frame.copy()

        for session in self.all_sessions:
            color = session["color"]

            for (cx,cy,x1,y1,x2,y2) in session["holes"]:
                cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
                cv2.circle(frame,(cx,cy),2,color,-1)

            if session["max_pair"]:
                (p1,p2) = session["max_pair"]
                cv2.line(frame, p1, p2, color, 2)

        return frame

    # -------------------------------------------------
    # SESSION DATA FOR WEB
    # -------------------------------------------------
    def get_session_data(self):
        """Return current session information for web interface"""
        return {
            'current_session_index': self.current_session_index,
            'total_sessions': len(self.all_sessions),
            'active_hits': len(self.all_sessions[self.current_session_index]['holes']) if self.current_session_index != -1 else 0,
            'all_sessions': [
                {
                    'name': session.get('name', 'N/A'),
                    'service_no': session.get('service_no', 'N/A'),
                    'service_year': session.get('service_year', 'N/A'),
                    'hits': len(session.get('holes', [])),
                    'group_size': session.get('group_size', 0.0),
                    'timestamp': session.get('timestamp', 'N/A')
                }
                for session in self.all_sessions
            ]
        }

    # -------------------------------------------------
    # CLEANUP
    # -------------------------------------------------
    def save_final_summary(self):
        frame = self.get_display_frame()
        path = os.path.join(self.SAVE_DIR, "FINAL_SUMMARY.jpg")
        cv2.imwrite(path, frame)

    def release(self):
        if self.cam and self.camera_initialized:
            self.cam.MV_CC_StopGrabbing()
            self.cam.MV_CC_CloseDevice()
            self.cam.MV_CC_DestroyHandle()
            self.camera_initialized = False