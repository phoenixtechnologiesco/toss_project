import sys
import cv2
import numpy as np
import time
import math
import os
import re # Added for safe filename creation
from datetime import datetime
from itertools import combinations

# --- PyQt Imports ---
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, 
    QHBoxLayout, QPushButton, QLabel, QStackedWidget,
    QGridLayout, QLineEdit, QMessageBox, QSizePolicy, 
    QTableWidget, QTableWidgetItem, QHeaderView, QDialog, # QDialog added for report view
    QScrollArea
)

# 1. PyQt5.QtCore se QCloseEvent hata diya gaya hai
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer 

# 2. PyQt5.QtGui mein QCloseEvent ko add kar diya gaya hai (Yeh correct location hai)
from PyQt5.QtGui import QImage, QPixmap, QCloseEvent, QBrush, QColor # QBrush/QColor for table

# --- YOLO/OpenCV Imports ---
try:
    from ultralytics import YOLO
    import torch
    
    # Check PyTorch compatibility - helps diagnose WinError 1114 issues
    if torch.cuda.is_available():
        print("PyTorch CUDA available. Running on GPU (if specified).")
    else:
        print("PyTorch running on CPU mode.")
        
except ImportError as e:
    print(f"🛑 ERROR: Required library not found: {e}")
    sys.exit(1)
except Exception as e:
    # Captures the critical OSError: [WinError 1114] DLL error
    if "[WinError 1114]" in str(e) or "c10.dll" in str(e):
        print(f"🛑 CRITICAL PYTORCH ERROR: {e}")
        print("This is likely a missing system dependency (Microsoft Visual C++ Redistributable) or an incompatible GPU installation.")
        print("Please re-install PyTorch using the CPU-only command: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu")
    else:
        print(f"🛑 UNKNOWN ERROR during PyTorch/YOLO load: {e}")
    sys.exit(1)


# **********************************************
## 1. CONFIGURATION & CONSTANTS
# **********************************************

MODEL_PATH = "best.pt"
model = None
try:
    if os.path.exists(MODEL_PATH):
        model = YOLO(MODEL_PATH)
        print(f"YOLO Model loaded from: {MODEL_PATH}")
    else:
        print(f"⚠️ WARNING: YOLO model file '{MODEL_PATH}' not found. Live detection disabled.")
except Exception as e:
    print(f"🛑 ERROR: Failed to load YOLO model: {e}")
    model = None 

# Camera Stream URL
stream_url = "rtsp://admin:phoenix0332@192.168.1.108:554/cam/realmonitor?channel=1&subtype=0"

# --- SECURITY CONSTANT ---
DEFAULT_PASSWORD = "123" 


# Constants
PIXELS_PER_INCH = 186 
PROXIMITY_THRESHOLD_PX = 25 
CONFIDENCE_THRESHOLD = 0.2
IOU_THRESHOLD = 0.5
BOX_SCALE_FACTOR = 1.5 
SAVE_DIR = "shooting_sessions"
IMAGE_SAVE_DIR = os.path.join(SAVE_DIR, "session_images") # New directory for images
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(IMAGE_SAVE_DIR, exist_ok=True) # Ensure image directory exists

# Color Palette (BGR format)
COLOR_PALETTE = [
    (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 0, 128), (0, 128, 128),
    (128, 128, 0), (255, 165, 0), (255, 192, 203), (165, 42, 42),
]

# Color for loaded (inactive) sessions
LOADED_SESSION_COLOR = (150, 150, 150) 


# ************************************************************
## 2. LOGIN WINDOW CLASS (STYLED)
# ************************************************************

class LoginWindow(QWidget):
    # Signal to tell the main window that login was successful
    login_successful = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TOS Login")
        self.setFixedSize(350, 250)
        
        # --- DARK THEME STYLE APPLIED ---
        self.setStyleSheet("""
            QWidget {
                background-color: #2e3b4e; /* Dark blue/grey background */
                color: #ffffff; /* White text */
                font-size: 14px;
            }
            QLabel {
                font-weight: bold;
                padding-top: 10px;
            }
            QLineEdit {
                padding: 8px;
                border: 1px solid #5d7591;
                border-radius: 5px;
                background-color: #1e2833;
                color: #ffffff;
            }
            QPushButton {
                padding: 10px;
                border: none;
                border-radius: 5px;
                background-color: #4CAF50; /* Green button */
                color: white;
                font-weight: bold;
                margin-top: 15px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        # -------------------------------------

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)

        # Title/Heading
        title_label = QLabel("<h2>Admin Login</h2>")
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        
        self.label = QLabel("Enter Password:")
        self.password_input = QLineEdit()
        self.password_input.setPlaceholderText("Password")
        self.password_input.setEchoMode(QLineEdit.Password) 
        self.password_input.returnPressed.connect(self.check_login) # Enter press support
        
        self.login_button = QPushButton("Login & Go to Dashboard")
        self.login_button.clicked.connect(self.check_login)
        
        layout.addWidget(self.label)
        layout.addWidget(self.password_input)
        layout.addWidget(self.login_button)
        
    def check_login(self):
        password = self.password_input.text()
        
        if password == DEFAULT_PASSWORD:
            QMessageBox.information(self, "Success", "Login Successful!")
            self.login_successful.emit() 
            self.close()
        else:
            QMessageBox.critical(self, "Error", "Invalid Password. Please try again.")
            self.password_input.clear()


# ************************************************************
## 3. SHARED DATA HELPER FUNCTION
# ************************************************************

def get_all_existing_centers(all_sessions_list):
    """Returns all confirmed hole centers (cx, cy) from all sessions."""
    centers = []
    for s in all_sessions_list:
        # Ignore the currently active session's holes from the 'existing' list
        # to allow new detections in the active session.
        # Check if the session's color is the 'LOADED_SESSION_COLOR' (means it's not active)
        if s["color"] == LOADED_SESSION_COLOR:
            for entry in s["holes"]:
                if len(entry) >= 2: 
                    centers.append((entry[0], entry[1])) 
    return centers


# ************************************************************
## 4. REPORT DIALOG (NEW CLASS)
# ************************************************************

class SessionReportDialog(QDialog):
    """Displays detailed report and the saved target image for a session."""
    def __init__(self, session_data, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Report: {session_data['name']} ({session_data['service_no']})")
        self.setFixedSize(800, 600)
        self.setStyleSheet("""
            QDialog { background-color: #1e2833; color: #e0e0e0; }
            QLabel { color: #e0e0e0; font-size: 14px; }
            QLabel#header { color: #4CAF50; font-size: 18px; font-weight: bold; }
        """)
        
        main_layout = QVBoxLayout(self)
        
        # --- Details Section ---
        details_label = QLabel()
        details_label.setObjectName("header")
        details_text = f"Shooter: {session_data.get('name', 'N/A')} (SN: {session_data.get('service_no', 'N/A')}, SY: {session_data.get('service_year', 'N/A')})\n"
        details_text += f"Total Hits: {len(session_data['holes'])}\n"
        details_text += f"Group Size (Extreme Spread): {session_data.get('group_size_inches', 0.0):.2f} inches\n"
        details_text += f"Start Time: {session_data.get('start_time', 'N/A').replace('_', ' ')}"
        details_label.setText(details_text)
        main_layout.addWidget(details_label)
        
        # --- Image Display Section ---
        self.image_label = QLabel("No Target Image Saved.")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.setStyleSheet("border: 1px dashed #5d7591;")
        
        image_path = session_data.get('LastImagePath')
        if image_path and os.path.exists(image_path):
            pixmap = QPixmap(image_path)
            # Scale image to fit within the dialog while maintaining aspect ratio
            if not pixmap.isNull():
                scaled_pixmap = pixmap.scaled(750, 500, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.image_label.setPixmap(scaled_pixmap)
                self.image_label.setText("") 
        
        main_layout.addWidget(self.image_label)
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        main_layout.addWidget(close_btn)


# **********************************************
## 5. DETECTION WORKER THREAD (QThread)
# **********************************************

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray) 
    hit_count_signal = pyqtSignal(int) 
    status_signal = pyqtSignal(str) 

    def __init__(self, shared_data, parent=None):
        super().__init__(parent)
        self.cap = None
        self.running = False 
        self.shared_data = shared_data
        self.get_all_existing_centers = get_all_existing_centers
        self.model = model # Use the global model instance
        self.last_frame = None # To store the last frame before session ends

    def run(self):
        self.cap = cv2.VideoCapture(stream_url)
        if not self.cap.isOpened():
            self.status_signal.emit("ERROR: Cannot open RTSP stream. Check URL/Connection.")
            self.running = False
            
            w, h = 640, 480
            frame = np.zeros((h, w, 3), dtype=np.uint8)
            cv2.putText(frame, "CAMERA OFFLINE. NO LIVE DATA.", (w//2 - 200, h//2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            self.change_pixmap_signal.emit(frame)
            return

        self.running = True
        self.status_signal.emit("System Ready. Detection Loop Started.")

        while self.running:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                self.status_signal.emit("Frame read error. Retrying...")
                time.sleep(0.1)
                continue
            
            # Save the raw frame for analysis/saving before drawing annotations
            self.last_frame = frame.copy() 

            # --- 1. YOLO DETECTION ---
            current_detections = [] 
            if self.model and self.shared_data.get('index', -1) != -1: # Only run model if a session is active
                try:
                    results = self.model.predict(
                        frame,
                        conf=CONFIDENCE_THRESHOLD,
                        iou=IOU_THRESHOLD,
                        device='cpu', 
                        verbose=False
                    )
                    
                    for box in results[0].boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                        # Calculate scaled bounding box for drawing (better visualization)
                        w = x2 - x1
                        h = y2 - y1
                        new_w = int(w * BOX_SCALE_FACTOR)
                        new_h = int(h * BOX_SCALE_FACTOR)
                        scaled_x1 = max(0, cx - new_w // 2)
                        scaled_y1 = max(0, cy - new_h // 2)
                        scaled_x2 = min(frame.shape[1], cx + new_w // 2)
                        scaled_y2 = min(frame.shape[0], cy + new_h // 2)

                        current_detections.append((cx, cy, scaled_x1, scaled_y1, scaled_x2, scaled_y2))
                
                except Exception as e:
                    self.status_signal.emit(f"Detection Error: {e}")
                    QThread.msleep(500) # Wait longer on error
                    continue
            
            # --- 2. ADD NEW HOLES TO CURRENT SESSION ---
            current_index = self.shared_data.get('index', -1)
            
            if current_index != -1 and current_index < len(self.shared_data['all_sessions']):
                session = self.shared_data['all_sessions'][current_index]
                # Get existing holes from all INACTIVE sessions + current active session's existing holes
                all_existing = self.get_all_existing_centers(self.shared_data['all_sessions'])
                
                new_holes_found = 0
                for (cx, cy, sx1, sy1, sx2, sy2) in current_detections:
                    is_duplicate = False
                    for (ox, oy) in all_existing:
                        # Check proximity against ALL confirmed holes
                        if math.hypot(cx - ox, cy - oy) < PROXIMITY_THRESHOLD_PX:
                            is_duplicate = True
                            break

                    if not is_duplicate:
                        # Add as a NEW hole
                        session["holes"].append((cx, cy, sx1, sy1, sx2, sy2))
                        new_holes_found += 1
                        
                if new_holes_found > 0:
                    current_holes_count = len(session["holes"])
                    self.hit_count_signal.emit(current_holes_count)
            
            # --- 3. DRAW ALL BOXES/CIRCLES FROM ALL SESSIONS ---
            for sess in self.shared_data['all_sessions']:
                color = sess["color"]
                shooter_label = f"{sess['name']} ({len(sess['holes'])})"
                
                # Draw label only for active session or if a hit is visible
                if sess["color"] != LOADED_SESSION_COLOR or len(sess["holes"]) > 0:
                    cv2.putText(frame, shooter_label, (10, 30 + self.shared_data['all_sessions'].index(sess) * 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                            
                for hole in sess["holes"]:
                    if len(hole) == 6:
                        (cx, cy, x1, y1, x2, y2) = hole
                        # Draw bounding box only for the active session (to avoid clutter from old ones)
                        if sess["color"] != LOADED_SESSION_COLOR:
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3) 
                        
                        # Always draw the center circle for confirmed hits
                        cv2.circle(frame, (cx, cy), 5, color, -1)
                    elif len(hole) >= 2: 
                        # Handle loaded sessions which might only have (cx, cy)
                        cx, cy = hole[0], hole[1]
                        cv2.circle(frame, (cx, cy), 5, color, -1)

            # --- 4. EMIT PROCESSED FRAME ---
            self.change_pixmap_signal.emit(frame)
            
            QThread.msleep(10) # Process at around 100 FPS (if possible)

        # Cleanup
        if self.cap:
            self.cap.release()
        self.status_signal.emit("Detection Loop Stopped. Camera Released.")

    def start(self):
        # Prevent starting if already running
        if not self.isRunning():
            self.running = True
            super().start()

    def stop(self):
        self.running = False
        # Wait for thread to finish gracefully (up to 2 seconds)
        if self.isRunning():
            self.wait(2000)


# **********************************************
## 6. MAIN WINDOW (UI) STRUCTURE (STYLED)
# **********************************************

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Target Observation System (TOS) - Live Analysis")
        self.setGeometry(100, 100, 1200, 800)
        self.current_frame_pixmap = None # NEW: To store the last frame as QPixmap

        # --- APPLY GLOBAL DARK THEME TO MAIN WINDOW ---
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #1e2833; /* Dark background */
                color: #e0e0e0; /* Light text */
            }
            QLabel {
                color: #e0e0e0;
            }
            h2 {
                color: #4CAF50; /* Green heading color */
            }
            QLineEdit {
                background-color: #2e3b4e;
                border: 1px solid #5d7591;
                color: #ffffff;
                padding: 5px;
            }
            QTableWidget {
                background-color: #2e3b4e;
                gridline-color: #3f4c5e;
                color: #ffffff;
                selection-background-color: #5d7591; /* Highlight selected row */
            }
            QHeaderView::section {
                background-color: #2e3b4e;
                color: #4CAF50;
                padding: 4px;
                border: 1px solid #3f4c5e;
            }
            QPushButton {
                padding: 10px;
                margin: 5px; 
                border-radius: 5px;
                font-weight: bold;
                background-color: #3f4c5e; 
                color: white;
            }
            QPushButton:hover {
                background-color: #5d7591;
            }
            #DeleteButton {
                background-color: #E74C3C; /* Red color for Delete */
                color: white;
            }
            #DeleteButton:hover {
                background-color: #C0392B; 
            }
        """)
        # ---------------------------------------------
        
        self.shared_data = {
            'all_sessions': [], 
            'index': -1, 
            'session_details': {'studentName': "N/A", 'serviceNo': "N/A", 'serviceYear': "N/A"} 
        }

        # Initialize thread here so closeEvent can access it safely
        self.thread = self.create_video_thread() 
        
        self.hide() 
        
        self.login_window = LoginWindow()
        self.login_window.login_successful.connect(self.initialize_main_ui)
        self.login_window.show()


    def initialize_main_ui(self):
        """Initializes the main UI components after successful login."""
        self.load_all_sessions() 

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)
        
        self.setup_sidebar()
        self.setup_content_area()
        
        # Ensure thread is stopped when app closes (Important for clean shutdown)
        QApplication.instance().aboutToQuit.connect(self.close_cleanup) 
        
        self.showMaximized() 


    def load_all_sessions(self):
        """Loads session history from saved .txt files into shared_data."""
        loaded_sessions = []
        
        for filename in os.listdir(SAVE_DIR):
            if filename.endswith(".txt"):
                file_path = os.path.join(SAVE_DIR, filename)
                
                try:
                    with open(file_path, 'r') as f:
                        content = f.read()
                    
                    data = {}
                    holes = []
                    
                    # Parsing the text file content
                    for line in content.splitlines():
                        if line.startswith("Shooter Name:"):
                            data['name'] = line.split(': ')[1].strip()
                        elif line.startswith("Service No:"):
                            data['service_no'] = line.split(': ')[1].strip()
                        elif line.startswith("Service Year:"):
                            data['service_year'] = line.split(': ')[1].strip()
                        elif line.startswith("Start Time:"):
                            data['start_time'] = line.split(': ')[1].strip()
                        elif line.startswith("Extreme Spread (Group Size):"):
                            try:
                                group_size_str = line.split(': ')[1].strip().split(' ')[0]
                                data['group_size_inches'] = float(group_size_str)
                            except ValueError:
                                data['group_size_inches'] = 0.0 
                        elif line.startswith("Last Image Path:"): # NEW: Load image path
                            data['LastImagePath'] = line.split(': ')[1].strip()
                        
                        # Hole coordinates are expected in the format (cx, cy)
                        elif line.strip().startswith('(') and line.strip().endswith(')'):
                            parts = line.strip('()').split(', ')
                            if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
                                cx = int(parts[0])
                                cy = int(parts[1])
                                # Load all 6 components (cx, cy, x1, y1, x2, y2) with placeholders
                                holes.append((cx, cy, 0, 0, 0, 0)) 

                    if 'name' in data:
                        session_data = {
                            "name": data.get('name', 'N/A'),
                            "service_no": data.get('service_no', 'N/A'),
                            "service_year": data.get('service_year', 'N/A'),
                            "start_time": data.get('start_time', 'N/A'),
                            "holes": holes, 
                            "color": LOADED_SESSION_COLOR, # All loaded sessions start as inactive
                            "group_size_inches": data.get('group_size_inches', 0.0),
                            "LastImagePath": data.get('LastImagePath', None) # NEW
                        }
                        loaded_sessions.append(session_data)
                        
                except Exception as e:
                    print(f"Error reading or parsing session file {filename}: {e}")

        self.shared_data['all_sessions'].extend(loaded_sessions)
        print(f"Loaded {len(loaded_sessions)} previous sessions from disk.")

    def setup_sidebar(self):
        self.sidebar = QWidget()
        self.sidebar.setFixedWidth(200)
        # --- SIDEBAR STYLING ---
        self.sidebar.setStyleSheet("""
            QWidget {
                background-color: #2e3b4e; /* Darker than main window */
                color: #ffffff; 
            }
            QPushButton {
                text-align: left; /* Align text to match web look */
            } 
            QPushButton#stop_btn { /* Specific style for Stop button */
                background-color: #C0392B; 
            }
            QPushButton#stop_btn:hover {
                background-color: #E74C3C; 
            }
        """)
        # -----------------------
        self.sidebar_layout = QVBoxLayout(self.sidebar)
        
        self.btn_dashboard = QPushButton("🏠 Dashboard")
        self.btn_new_session = QPushButton("📝 New Session")
        self.btn_live_detection = QPushButton("📹 Live Detection")
        self.btn_stop_session = QPushButton("🛑 Stop Session & Analyze")
        
        self.btn_stop_session.setObjectName("stop_btn")

        self.btn_dashboard.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(2))
        self.btn_new_session.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(0)) 
        self.btn_live_detection.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(1))
        
        self.btn_stop_session.clicked.connect(self.stop_current_session) 

        self.sidebar_layout.addWidget(QLabel("TOS"))
        self.sidebar_layout.addWidget(QLabel("admin"))
        self.sidebar_layout.addSpacing(10)
        self.sidebar_layout.addWidget(self.btn_dashboard)
        self.sidebar_layout.addWidget(self.btn_new_session)
        self.sidebar_layout.addWidget(self.btn_live_detection)
        self.sidebar_layout.addStretch()
        self.sidebar_layout.addWidget(self.btn_stop_session)
        
        self.main_layout.addWidget(self.sidebar)

    def setup_content_area(self):
        self.stacked_widget = QStackedWidget()
        
        self.new_session_page = QWidget()
        self.setup_new_session_page() 
        self.stacked_widget.addWidget(self.new_session_page)
        
        self.live_detection_page = QWidget()
        self.setup_live_detection_page() 
        self.stacked_widget.addWidget(self.live_detection_page)
        
        self.dashboard_page = QWidget()
        self.setup_dashboard_page()
        self.stacked_widget.addWidget(self.dashboard_page)
        
        self.main_layout.addWidget(self.stacked_widget)
        
    def setup_new_session_page(self):
        layout = QGridLayout(self.new_session_page)
        
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Enter Shooter Name")
        
        self.service_input = QLineEdit()
        self.service_input.setPlaceholderText("Enter Service Number (SN)")

        self.service_year_input = QLineEdit()
        self.service_year_input.setPlaceholderText("Enter Service Year (SY)")
        
        self.start_btn = QPushButton("🚀 Start Session & Activate Camera")
        self.start_btn.setStyleSheet("padding: 15px; background-color: #4CAF50; color: white; font-weight: bold; border-radius: 5px;")
        
        layout.addWidget(QLabel("<h2>Start New Session Details</h2>"), 0, 0, 1, 2)
        layout.addWidget(QLabel("Service No (SN):"), 1, 0)
        layout.addWidget(self.service_input, 1, 1)
        layout.addWidget(QLabel("Service Year (SY):"), 2, 0)
        layout.addWidget(self.service_year_input, 2, 1)
        layout.addWidget(QLabel("Shooter Name:"), 3, 0)
        layout.addWidget(self.name_input, 3, 1)
        
        layout.addWidget(self.start_btn, 4, 1, 1, 1) 
        layout.setRowStretch(5, 1) 

        self.start_btn.clicked.connect(self.start_new_session)
        
    def setup_live_detection_page(self):
        layout = QVBoxLayout(self.live_detection_page)
        
        self.video_label = QLabel("Camera Feed will appear here...")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding) 
        self.video_label.setStyleSheet("border: 2px solid #4CAF50; background-color: black; color: white;")
        layout.addWidget(self.video_label)
        
        self.status_label = QLabel("Status: Ready. Please start a new session.")
        self.status_label.setStyleSheet("padding: 5px; background-color: #3f4c5e; color: #4CAF50; font-weight: bold; border-radius: 5px;")
        layout.addWidget(self.status_label)
        
    def setup_dashboard_page(self):
        layout = QVBoxLayout(self.dashboard_page)
        
        header_layout = QHBoxLayout()
        header_layout.addWidget(QLabel("<h2>🎯 Session History & Analysis</h2>"))
        
        # --- DELETE BUTTON ---
        self.delete_all_btn = QPushButton("🗑️ Delete All Old Sessions")
        self.delete_all_btn.setObjectName("DeleteButton")
        self.delete_all_btn.clicked.connect(self.delete_all_old_sessions)
        header_layout.addWidget(self.delete_all_btn)
        
        layout.addLayout(header_layout)
        # ---------------------
        
        self.session_table = QTableWidget()
        # --- UPDATED COLUMN COUNT (6 -> 5) ---
        self.session_table.setColumnCount(5) 
        self.session_table.setHorizontalHeaderLabels(
            # Start Time removed, Report added
            ["Name", "Service No", "Service Year", "Total Hits", "Report"] 
        )
        self.session_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.session_table)
        
        # --- NEW: Connect cellClicked signal ---
        self.session_table.cellClicked.connect(self.handle_table_click)

        self.update_dashboard()

    def handle_table_click(self, row, column):
        """Handle clicks on the table. Only clicking the 'Report' column does anything."""
        # Check if the clicked column is the "Report" column (index 4)
        if column == 4:
            # Get the session data corresponding to the clicked row
            # Since we display active session first, the index 'i' in update_dashboard
            # corresponds to 'row' here.
            
            # Find the original session dictionary in all_sessions
            try:
                # We need a unique key. Let's use Name, SN, and SY combination as a temporary key
                name = self.session_table.item(row, 0).text()
                sn = self.session_table.item(row, 1).text()
                sy = self.session_table.item(row, 2).text()

                session_to_view = None
                for session in self.shared_data['all_sessions']:
                    if session['name'] == name and session['service_no'] == sn and session['service_year'] == sy:
                        session_to_view = session
                        break
                
                if session_to_view:
                    # Open the report dialog
                    report_dialog = SessionReportDialog(session_to_view, self)
                    report_dialog.exec_()
                else:
                    QMessageBox.warning(self, "Error", "Session data not found in memory.")

            except Exception as e:
                print(f"Error handling table click: {e}")
                QMessageBox.critical(self, "Error", "Error loading session details.")
        
    def update_dashboard(self):
        """Refreshes the session table on the Dashboard page."""
        self.session_table.setRowCount(0)
        
        # Filter and process sessions: active session should be displayed first
        active_index = self.shared_data.get('index', -1)
        active_session = None
        inactive_sessions = []
        
        for i, session in enumerate(self.shared_data['all_sessions']):
            if i == active_index:
                active_session = session
            else:
                inactive_sessions.append(session)

        # Prepare list for display (Active first, then Inactive)
        display_sessions = ([active_session] if active_session else []) + inactive_sessions
        
        for i, session in enumerate(display_sessions):
            self.session_table.insertRow(i)
            
            name_item = QTableWidgetItem(session['name'])
            service_item = QTableWidgetItem(session['service_no'])
            service_year_item = QTableWidgetItem(session.get('service_year', 'N/A'))
            
            hits_item = QTableWidgetItem(str(len(session['holes'])))
            
            group_size = session.get('group_size_inches', 'N/A')
            
            # --- NEW Report Item ---
            report_item = QTableWidgetItem("View Report")
            report_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable) # Make it selectable
            if session.get('LastImagePath') and os.path.exists(session['LastImagePath']):
                report_item.setForeground(QBrush(QColor("#4CAF50"))) # Green if image is saved
            else:
                report_item.setForeground(QBrush(QColor("yellow"))) # Yellow if no image saved (or active)

            if session == active_session:
                # Active session display
                group_size_item = QTableWidgetItem("ACTIVE")
                name_item.setBackground(QApplication.palette().highlight()) # Highlight color
            else:
                # Inactive session display
                if group_size != 'N/A':
                    group_size = f"{group_size:.2f} in"
                
                group_size_item = QTableWidgetItem(group_size)
                name_item.setBackground(QColor("#2e3b4e")) # Default dark background


            self.session_table.setItem(i, 0, name_item)
            self.session_table.setItem(i, 1, service_item)
            self.session_table.setItem(i, 2, service_year_item) 
            # self.session_table.setItem(i, 3, time_item) # START TIME REMOVED
            self.session_table.setItem(i, 3, hits_item)
            self.session_table.setItem(i, 4, report_item) # Report Column
            
    def create_video_thread(self):
        thread = VideoThread(self.shared_data) 
        
        thread.change_pixmap_signal.connect(self.update_image)
        thread.status_signal.connect(self.update_status)
        thread.hit_count_signal.connect(self.update_hit_count)
        
        return thread
    
    def update_image(self, cv_img):
        """Receives a frame from the thread and displays it in the video_label, also stores as pixmap."""
        try:
            h, w, ch = cv_img.shape
            bytes_per_line = ch * w
            
            # Convert BGR (OpenCV) to RGB for Qt
            rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            c_pixmap = QPixmap.fromImage(convert_to_Qt_format)
            
            # --- NEW: Store the last frame as QPixmap for saving ---
            self.current_frame_pixmap = c_pixmap.copy() 
            # ------------------------------------------------------
            
            # Scale to fit the label dynamically
            scaled_pixmap = c_pixmap.scaled(
                self.video_label.size(), 
                Qt.KeepAspectRatio, 
                Qt.SmoothTransformation
            )
            
            self.video_label.setPixmap(scaled_pixmap)
            
        except Exception as e:
            print(f"Error in image conversion/display: {e}")

    def update_status(self, message):
        self.status_label.setText(f"Status: {message}")
        
    def update_hit_count(self, count):
        current_shooter = self.shared_data['session_details'].get('studentName', "N/A")
        
        if self.shared_data.get('index', -1) != -1:
            self.status_label.setText(f"Status: ACTIVE | Shooter: {current_shooter} | Hits: {count}")
        
        self.update_dashboard()

    def calculate_group_size(self, centers):
        """Calculates the Extreme Spread (Group Size) in inches."""
        valid_centers = [(c[0], c[1]) for c in centers if len(c) >= 2]
        
        if len(valid_centers) < 2:
            return 0.0

        max_distance = 0
        
        # Compare every unique pair of points
        for (cx1, cy1), (cx2, cy2) in combinations(valid_centers, 2):
            distance_px = math.hypot(cx1 - cx2, cy1 - cy2)
            if distance_px > max_distance:
                max_distance = distance_px
        
        # Convert max pixel distance to inches
        group_size_inches = max_distance / PIXELS_PER_INCH 
        return group_size_inches

    def start_new_session(self):
        shooter_name = self.name_input.text().strip()
        service_no = self.service_input.text().strip()
        service_year = self.service_year_input.text().strip() 

        if not shooter_name or not service_no or not service_year:
            QMessageBox.warning(self, "Input Error", "Shooter Name, Service Number, and Service Year are required to start a session.")
            return

        if self.shared_data['index'] != -1:
            QMessageBox.information(self, "Session Active", "Please stop the current session before starting a new one.")
            return

        session_index = len(self.shared_data['all_sessions'])
        
        # --- NEW FILE NAMING LOGIC ---
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Safe name: remove special characters from name
        safe_name = re.sub(r'[^A-Za-z0-9_]+', '', shooter_name.lower())
        
        # File ID: SN_SY_NAME_TIMESTAMP
        session_id = f"{service_no}_{service_year}_{safe_name}_{timestamp}"
        # -----------------------------

        new_session = {
            "name": shooter_name,
            "service_no": service_no,
            "service_year": service_year, 
            "start_time": timestamp, # Using full timestamp for internal consistency
            "holes": [], 
            "color": COLOR_PALETTE[session_index % len(COLOR_PALETTE)], # Assign a unique color
            "group_size_inches": 0.0,
            "filename": f"{session_id}.txt", # NEW: Store filename
            "LastImagePath": None # NEW: Initialize image path
        }
        
        self.shared_data['all_sessions'].append(new_session)
        self.shared_data['index'] = session_index
        self.shared_data['session_details'] = {
            'studentName': shooter_name, 
            'serviceNo': service_no,
            'serviceYear': service_year 
        }
        
        # Start the thread if it's not running
        if not self.thread.isRunning():
            self.thread.start()
        else:
            self.thread.status_signal.emit(f"New session for {shooter_name} started.")

        self.stacked_widget.setCurrentIndex(1)
        self.status_label.setText(f"Status: ACTIVE | Session started for {shooter_name}.")
        self.update_dashboard() 

    def save_session_data(self, session):
        """Saves the final session data (with group size and image path) to a .txt file."""
        file_path = os.path.join(SAVE_DIR, session['filename'])
        
        try:
            with open(file_path, 'w') as f:
                f.write(f"Shooter Name: {session['name']}\n")
                f.write(f"Service No: {session['service_no']}\n")
                f.write(f"Service Year: {session['service_year']}\n")
                f.write(f"Start Time: {session['start_time']}\n")
                f.write(f"Total Hits: {len(session['holes'])}\n")
                f.write(f"Extreme Spread (Group Size): {session['group_size_inches']:.2f} inches\n")
                f.write(f"Last Image Path: {session.get('LastImagePath', 'N/A')}\n\n") # NEW: Save image path
                
                f.write("--- Hit Coordinates (cx, cy) ---\n")
                for hole in session['holes']:
                    f.write(f"({hole[0]}, {hole[1]})\n")
            print(f"Session saved: {file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save session data to disk: {e}")

    def save_last_image(self, session_id, pixmap):
        """Saves the last QPixmap frame as PNG."""
        if pixmap is None or pixmap.isNull():
            print("Warning: Last frame pixmap is null, cannot save image.")
            return None
        
        image_filename = f"{session_id}_target_result.png"
        image_path = os.path.join(IMAGE_SAVE_DIR, image_filename)
        
        try:
            pixmap.save(image_path, "PNG")
            print(f"Last image saved to: {image_path}")
            return image_path
        except Exception as e:
            print(f"Error saving image: {e}")
            return None

    def stop_current_session(self):
        current_index = self.shared_data['index']
        
        if current_index == -1:
            QMessageBox.information(self, "No Session Active", "There is no active session to stop.")
            return
            
        session = self.shared_data['all_sessions'][current_index]

        # --- 1. Calculate Group Size ---
        centers = session["holes"] 
        group_size = self.calculate_group_size(centers)
        session["group_size_inches"] = group_size

        # --- 2. Save Last Image ---
        if self.current_frame_pixmap:
            # Create a unique file ID from session details for the image name
            session_file_id = session['filename'].replace('.txt', '')
            image_path = self.save_last_image(session_file_id, self.current_frame_pixmap)
            session["LastImagePath"] = image_path
            
        # --- 3. Save Session Data to Disk ---
        self.save_session_data(session)

        # --- 4. Reset Shared Data / UI ---
        session["color"] = LOADED_SESSION_COLOR # Mark as inactive (grey on detection feed)
        self.shared_data['index'] = -1 
        self.shared_data['session_details'] = {'studentName': "N/A", 'serviceNo': "N/A", 'serviceYear': "N/A"}
        self.current_frame_pixmap = None # Clear saved image
        
        self.update_dashboard()
        self.status_label.setText(f"Status: Session ended for {session['name']}. Analysis saved.")
        
        # Go to Dashboard to see results
        self.stacked_widget.setCurrentIndex(2)
        
        QMessageBox.information(self, "Session Ended", f"Session for {session['name']} ended successfully.\nTotal Hits: {len(centers)}\nGroup Size: {group_size:.2f} inches.")

    def delete_all_old_sessions(self):
        # ... (Existing logic for deleting sessions remains here) ...
        if self.shared_data['index'] != -1:
            QMessageBox.warning(self, "Action Denied", "Cannot delete sessions while a session is actively running.")
            return
            
        reply = QMessageBox.question(
            self,
            "Confirm Deletion",
            "Are you sure you want to delete ALL old sessions? This action is permanent.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            deleted_count = 0
            
            # 1. Delete .txt files from the directory
            for filename in os.listdir(SAVE_DIR):
                if filename.endswith(".txt"):
                    file_path = os.path.join(SAVE_DIR, filename)
                    try:
                        os.remove(file_path)
                        deleted_count += 1
                    except Exception as e:
                        print(f"Error deleting file {file_path}: {e}")
                        
            # 2. Delete .png image files from the image directory
            image_deleted_count = 0
            for filename in os.listdir(IMAGE_SAVE_DIR):
                if filename.endswith(".png"):
                    file_path = os.path.join(IMAGE_SAVE_DIR, filename)
                    try:
                        os.remove(file_path)
                        image_deleted_count += 1
                    except Exception as e:
                        print(f"Error deleting image file {file_path}: {e}")
                        
            # 3. Clear in-memory list
            self.shared_data['all_sessions'].clear()
            self.update_dashboard()
            
            QMessageBox.information(self, "Success", f"Successfully deleted {deleted_count} session files and {image_deleted_count} image files.")


    def close_cleanup(self, event=None):
        """Called when the application is about to quit."""
        if self.thread and self.thread.isRunning():
            self.thread.stop()
        if event:
            # If called via closeEvent (user clicks X)
            event.accept()
        else:
            # If called via aboutToQuit signal (clean up)
            pass

    def closeEvent(self, event: QCloseEvent):
        """Intercepts the window close event to ensure thread shutdown."""
        # Use the cleanup function
        self.close_cleanup(event)


# **********************************************
## 7. MAIN EXECUTION BLOCK
# **********************************************

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Initialize the main window, which handles the login flow
    main_window = MainWindow() 
    
    # main_window.show() # Show is handled after successful login
    
    sys.exit(app.exec_())