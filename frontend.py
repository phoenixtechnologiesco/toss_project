# frontend.py
import sys
import os
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QStackedWidget, QGridLayout, QLineEdit,
    QMessageBox, QTableWidget, QTableWidgetItem, QHeaderView, QDialog, QSizePolicy
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap, QCloseEvent, QBrush, QColor, QPalette

from backend import ShootingAnalyzer  

DEFAULT_PASSWORD = "123"
SAVE_DIR = "shooting_sessions"
IMAGE_SAVE_DIR = os.path.join(SAVE_DIR, "session_images")
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(IMAGE_SAVE_DIR, exist_ok=True)

class LoginWindow(QMainWindow):
    login_successful = pyqtSignal()
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TOS Login Required")
        self.setObjectName("LoginWindow")
        self.showMaximized()

        central_container = QWidget()
        central_container.setObjectName("CentralContainer")
        self.setCentralWidget(central_container)
        main_layout = QVBoxLayout(central_container)
        main_layout.setAlignment(Qt.AlignCenter)

        self.login_form_widget = QWidget()
        self.login_form_widget.setObjectName("LoginFormContainer")
        self.login_form_widget.setFixedSize(350, 300)
        form_layout = QVBoxLayout(self.login_form_widget)
        form_layout.setAlignment(Qt.AlignCenter)
        form_layout.setContentsMargins(30, 30, 30, 30)

        title_label = QLabel("<h2>Admin Login</h2>")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("color: #4CAF50; padding-bottom: 10px;")
        form_layout.addWidget(title_label)
        self.label = QLabel("Enter Password:")
        self.password_input = QLineEdit()
        self.password_input.setPlaceholderText("Password")
        self.password_input.setEchoMode(QLineEdit.Password) 
        self.password_input.returnPressed.connect(self.check_login)
        self.login_button = QPushButton("Login & Go to Dashboard")
        self.login_button.clicked.connect(self.check_login)
        form_layout.addWidget(self.label)
        form_layout.addWidget(self.password_input)
        form_layout.addWidget(self.login_button)
        main_layout.addWidget(self.login_form_widget, alignment=Qt.AlignCenter)

        image_path = r"bg_img.jpeg"
        self.bg_pixmap = QPixmap(image_path)
        if self.bg_pixmap.isNull():
            palette = self.palette()
            palette.setColor(QPalette.Window, QColor(0, 0, 0))
            self.setPalette(palette)
        else:
            scaled_pixmap = self.bg_pixmap.scaled(
                self.size(), 
                Qt.KeepAspectRatioByExpanding, 
                Qt.SmoothTransformation
            )
            brush = QBrush(scaled_pixmap)
            brush.setStyle(Qt.TexturePattern)
            palette = self.palette()
            palette.setBrush(QPalette.Window, brush)
            self.setPalette(palette)
            self.setAutoFillBackground(True)

        self.setStyleSheet("""
            QWidget { color: #ffffff; font-size: 14px; }
            QLineEdit {
                padding: 10px; border: 1px solid #5d7591; border-radius: 5px;
                background-color: #1e2833; color: #ffffff;
            }
            QPushButton {
                padding: 10px; border: none; border-radius: 5px;
                background-color: #4CAF50; color: white; font-weight: bold; margin-top: 15px;
            }
            QPushButton:hover { background-color: #45a049; }
            #LoginFormContainer {
                background-color: #283038D9; border: 1px solid #4a545e; border-radius: 10px;
            }
        """)

    def check_login(self):
        password = self.password_input.text()
        if password == DEFAULT_PASSWORD:
            QMessageBox.information(self, "Success", "Login Successful!")
            self.login_successful.emit() 
            self.close()
        else:
            QMessageBox.critical(self, "Error", "Invalid Password. Please try again.")
            self.password_input.clear()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, 'bg_pixmap') and not self.bg_pixmap.isNull():
            scaled_pixmap = self.bg_pixmap.scaled(
                self.size(), 
                Qt.KeepAspectRatioByExpanding, 
                Qt.SmoothTransformation
            )
            brush = QBrush(scaled_pixmap)
            brush.setStyle(Qt.TexturePattern)
            palette = self.palette()
            palette.setBrush(QPalette.Window, brush)
            self.setPalette(palette)
            self.update()

class SessionReportDialog(QDialog):
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
        details_label = QLabel()
        details_label.setObjectName("header")
        details_text = f"Shooter: {session_data.get('name', 'N/A')} (SN: {session_data.get('service_no', 'N/A')}, SY: {session_data.get('service_year', 'N/A')})\n"
        details_text += f"Total Hits: {len(session_data['holes'])}\n"
        details_text += f"Group Size (Extreme Spread): {session_data.get('group_size', 0.0):.2f} inches\n"
        details_text += f"Start Time: {session_data.get('timestamp', 'N/A').replace('_', ' ')}"
        details_label.setText(details_text)
        main_layout.addWidget(details_label)

        self.image_label = QLabel("No Target Image Saved.")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.setStyleSheet("border: 1px dashed #5d7591;")
        image_path = session_data.get('LastImagePath')
        if image_path and os.path.exists(image_path):
            pixmap = QPixmap(image_path)
            if not pixmap.isNull():
                scaled_pixmap = pixmap.scaled(750, 500, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.image_label.setPixmap(scaled_pixmap)
                self.image_label.setText("") 
            else:
                self.image_label.setText("❌ Image could not be loaded.")
        else:
            self.image_label.setText("📷 No target image saved for this session.")

        main_layout.addWidget(self.image_label)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        main_layout.addWidget(close_btn)

class VideoThread(QThread):
    frame_ready = pyqtSignal(np.ndarray)
    def __init__(self, analyzer):
        super().__init__()
        self.analyzer = analyzer
        self.running = True

    def run(self):
        while self.running:
            self.analyzer.process_frame()
            frame = self.analyzer.get_display_frame()
            if frame is not None:
                self.frame_ready.emit(frame)
            self.msleep(30)

    def stop(self):
        self.running = False
        self.wait()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Target Observation System (TOS) - Live Analysis")
        self.setGeometry(100, 100, 1200, 800)
        self.analyzer = None
        self.thread = None
        self.roi_selected = False

        self.login_window = LoginWindow()
        self.login_window.login_successful.connect(self.initialize_main_ui)
        self.login_window.show()

    def initialize_main_ui(self):
        try:
            self.analyzer = ShootingAnalyzer()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to initialize backend: {e}")
            sys.exit(1)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)

        self.setup_sidebar()
        self.setup_content_area()

        # # ROI selection
        # success, msg = self.analyzer.select_roi()
        # if not success:
        #     QMessageBox.critical(self, "ROI Error", msg)
        #     sys.exit(1)
        # QMessageBox.information(self, "ROI Selected", msg)
        # self.roi_selected = True

        # Start video thread
        self.thread = VideoThread(self.analyzer)
        self.thread.frame_ready.connect(self.update_image)
        self.thread.start()

        self.showMaximized()

    def setup_sidebar(self):
        self.sidebar = QWidget()
        self.sidebar.setFixedWidth(200)
        self.sidebar.setStyleSheet("""
            QWidget { background-color: #2e3b4e; color: #ffffff; }
            QPushButton { text-align: left; } 
            QPushButton#stop_btn { background-color: #C0392B; }
            QPushButton#stop_btn:hover { background-color: #E74C3C; }
        """)
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
        self.video_label.setFixedSize(1280, 720)
        self.video_label.setScaledContents(True)
        self.video_label.setStyleSheet("border: 2px solid #4CAF50; background-color: black; color: white;")
        layout.addWidget(self.video_label)
        self.status_label = QLabel("Status: Ready. Please start a new session.")
        self.status_label.setStyleSheet("padding: 5px; background-color: #3f4c5e; color: #4CAF50; font-weight: bold; border-radius: 5px;")
        layout.addWidget(self.status_label)

    def setup_dashboard_page(self):
        layout = QVBoxLayout(self.dashboard_page)
        header_layout = QHBoxLayout()
        header_layout.addWidget(QLabel("<h2>🎯 Session History & Analysis</h2>"))
        self.delete_all_btn = QPushButton("🗑️ Delete All Old Sessions")
        self.delete_all_btn.setObjectName("DeleteButton")
        header_layout.addWidget(self.delete_all_btn)
        layout.addLayout(header_layout)

        self.session_table = QTableWidget()
        self.session_table.setColumnCount(5) 
        self.session_table.setHorizontalHeaderLabels(
            ["Name", "Service No", "Service Year", "Total Hits", "Report"] 
        )
        self.session_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.session_table)

    def update_dashboard(self):
        """Update the session history table with all sessions"""
        self.session_table.setRowCount(0)
        for i, session in enumerate(self.analyzer.all_sessions):
            row = i
            self.session_table.insertRow(row)
            self.session_table.setItem(row, 0, QTableWidgetItem(session.get('name', 'N/A')))
            self.session_table.setItem(row, 1, QTableWidgetItem(session.get('service_no', 'N/A')))
            self.session_table.setItem(row, 2, QTableWidgetItem(session.get('service_year', 'N/A')))
            self.session_table.setItem(row, 3, QTableWidgetItem(str(len(session['holes']))))

            # Report button
            report_btn = QPushButton("View Report")
            report_btn.setObjectName("AnalyzeButton")
            
            # Connect the button's clicked signal directly to the new method
            report_btn.clicked.connect(lambda checked, sess=session: self.open_report_dialog(sess))

            btn_widget = QWidget()
            btn_layout = QHBoxLayout(btn_widget)
            btn_layout.addWidget(report_btn)
            btn_layout.setAlignment(Qt.AlignCenter)
            btn_layout.setContentsMargins(0, 0, 0, 0)
            self.session_table.setCellWidget(row, 4, btn_widget)

            # If this is the active session, highlight it
            if i == self.analyzer.current_session_index:
                for col in range(5):
                    item = self.session_table.item(row, col)
                    if item:
                        item.setBackground(QBrush(QColor(50, 100, 50, 150)))

    def open_report_dialog(self, session_data):
        """Open the SessionReportDialog for the given session data"""
        report_dialog = SessionReportDialog(session_data, self)
        report_dialog.exec_()

    def start_new_session(self):
        name = self.name_input.text().strip()
        sn = self.service_input.text().strip()
        sy = self.service_year_input.text().strip()
        if not name or not sn or not sy:
            QMessageBox.warning(self, "Input Error", "Please fill all fields.")
            return
        if self.analyzer:
            self.analyzer.start_session(name=name, service_no=sn, service_year=sy)  # ← Pass them
            self.status_label.setText(f"Session '{name}' started.")
            self.stacked_widget.setCurrentIndex(1)

    def stop_current_session(self):
        if not self.analyzer:
            return
        result = self.analyzer.stop_session()
        if not result["success"]:
            QMessageBox.information(self, "Info", result["message"])
            return

        self.status_label.setText(f"Hits: {result['hits']}, Group: {result['group_size']:.2f} in")

        current_session = self.analyzer.all_sessions[-1] if self.analyzer.all_sessions else None
        if current_session:
            report_dialog = SessionReportDialog(current_session, self)
            report_dialog.exec_()

        self.update_dashboard()

        self.stacked_widget.setCurrentIndex(2)

    @pyqtSlot(np.ndarray)
    def update_image(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        self.video_label.setPixmap(pixmap)  

    def closeEvent(self, event: QCloseEvent):
        if self.thread:
            self.thread.stop()
        if self.analyzer:
            self.analyzer.save_final_summary()
            self.analyzer.release()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow() 
    sys.exit(app.exec_())