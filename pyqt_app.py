import sys
import pickle
import threading
import time
import cv2
import face_recognition
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QTextEdit)
from PyQt6.QtGui import QImage, QPixmap, QFont
from PyQt6.QtCore import Qt, QThread, pyqtSignal

# --- Constants and Configuration ---
APP_TITLE = "C.C.E: Counter Criminal Evasion System (PyQt)"
# --- BGR Tuples for OpenCV ---
ACCENT_COLOR = (0, 255, 0)   # Green
ALERT_COLOR = (0, 0, 255)    # Red

# --- Video Processing Thread ---
class VideoThread(QThread):
    """
    A dedicated thread to handle video capture and face recognition
    to prevent the GUI from freezing.
    """
    # Signals to send data back to the main GUI thread
    change_pixmap_signal = pyqtSignal(np.ndarray)
    log_signal = pyqtSignal(str, str)
    status_signal = pyqtSignal(str, str)

    def __init__(self, known_data, criminal_list):
        super().__init__()
        self._run_flag = True
        self.known_data = known_data
        self.criminal_list = criminal_list
        
        # Optimization variables
        self.process_every_n_frames = 3
        self.frame_counter = 0
        self.last_known_locations = []
        self.last_known_names = []

    def run(self):
        """The main loop of the video thread."""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.log_signal.emit("Failed to open video source.", "ERROR")
            return

        self.log_signal.emit("Camera initialized successfully.", "INFO")
        
        while self._run_flag:
            ret, frame = cap.read()
            if not ret:
                self.log_signal.emit("Failed to grab frame from camera.", "WARNING")
                time.sleep(0.5)
                continue

            self.frame_counter += 1
            
            # Process only every Nth frame for performance
            if self.frame_counter % self.process_every_n_frames == 0:
                self.process_frame(frame)

            # Draw results on every frame for a smooth display
            processed_frame = self.draw_on_frame(frame)
            
            # Emit the processed frame to the GUI
            self.change_pixmap_signal.emit(processed_frame)
            
            # Check for criminals and update status
            is_criminal_detected = any(name.lower() in self.criminal_list for name in self.last_known_names)
            if is_criminal_detected:
                self.status_signal.emit("ALERT!", "#F44336") # Red
                detected_criminals = [n for n in self.last_known_names if n.lower() in self.criminal_list]
                if self.frame_counter % 15 == 0: # Log less frequently
                    self.log_signal.emit(f"Criminal detected: {detected_criminals}", "ALERT")
            else:
                self.status_signal.emit("RUNNING", "#4CAF50") # Green

        cap.release()
        self.log_signal.emit("Video thread stopped.", "INFO")

    def stop(self):
        """Sets the flag to stop the thread."""
        self._run_flag = False
        self.wait()

    def process_frame(self, frame):
        """Processes a single frame for face recognition."""
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        self.last_known_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, self.last_known_locations)

        current_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(self.known_data["encodings"], face_encoding, tolerance=0.6)
            name = "Unknown"

            if True in matches:
                first_match_index = matches.index(True)
                name = self.known_data["names"][first_match_index]
            
            current_names.append(name)
        self.last_known_names = current_names

    def draw_on_frame(self, frame):
        """Draws the last known face locations and names on the frame."""
        for (top, right, bottom, left), name in zip(self.last_known_locations, self.last_known_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            box_color = ACCENT_COLOR
            if name.lower() in self.criminal_list:
                box_color = ALERT_COLOR
            elif name == "Unknown":
                box_color = (255, 165, 0)

            cv2.rectangle(frame, (left, top), (right, bottom), box_color, 2)
            
            label_font = cv2.FONT_HERSHEY_DUPLEX
            (text_width, text_height), baseline = cv2.getTextSize(name, label_font, 0.8, 1)
            cv2.rectangle(frame, (left, bottom - text_height - 10), (left + text_width + 6, bottom), box_color, cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 6), label_font, 0.8, (255, 255, 255), 1)
        
        return frame

# --- Main Application Window ---
class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.setGeometry(100, 100, 1000, 700)
        self.setStyleSheet("background-color: #212121; color: #EAEAEA;")
        
        self.known_data = self.load_encodings()
        self.criminal_list = {"kshitij"}
        
        self.initUI()
        self.thread = None

    def load_encodings(self):
        """Loads face encodings. Exits if not found."""
        try:
            with open("encodings.pickle", "rb") as f:
                return pickle.load(f)
        except FileNotFoundError:
            # A proper GUI app would use a QMessageBox here.
            print("[ERROR] encodings.pickle not found. Please run encode_faces.py first.")
            sys.exit()

    def initUI(self):
        """Sets up the user interface."""
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout()

        # --- Left Control Panel ---
        control_panel = QWidget()
        control_panel.setFixedWidth(250)
        control_panel.setStyleSheet("background-color: #2C2C2C;")
        control_layout = QVBoxLayout()

        title_font = QFont("Helvetica", 12, QFont.Weight.Bold)
        
        lbl_title = QLabel("CONTROL PANEL")
        lbl_title.setFont(title_font)
        lbl_title.setStyleSheet("color: #4CAF50;")
        control_layout.addWidget(lbl_title, alignment=Qt.AlignmentFlag.AlignHCenter)

        self.btn_start = QPushButton("Start System")
        self.btn_start.setStyleSheet("background-color: #4CAF50; color: white; padding: 5px;")
        self.btn_start.clicked.connect(self.start_system)
        control_layout.addWidget(self.btn_start)

        self.btn_stop = QPushButton("Stop System")
        self.btn_stop.setStyleSheet("background-color: #F44336; color: white; padding: 5px;")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop_system)
        control_layout.addWidget(self.btn_stop)

        lbl_status_title = QLabel("SYSTEM STATUS")
        lbl_status_title.setFont(title_font)
        control_layout.addWidget(lbl_status_title, alignment=Qt.AlignmentFlag.AlignHCenter)
        
        self.status_label = QLabel("OFFLINE")
        status_font = QFont("Helvetica", 14, QFont.Weight.Bold)
        self.status_label.setFont(status_font)
        self.status_label.setStyleSheet("color: gray;")
        control_layout.addWidget(self.status_label, alignment=Qt.AlignmentFlag.AlignHCenter)

        lbl_log_title = QLabel("EVENT LOG")
        lbl_log_title.setFont(title_font)
        control_layout.addWidget(lbl_log_title, alignment=Qt.AlignmentFlag.AlignHCenter)
        
        self.event_log = QTextEdit()
        self.event_log.setReadOnly(True)
        self.event_log.setStyleSheet("background-color: #1A1A1A; color: #EAEAEA; font-family: Consolas;")
        control_layout.addWidget(self.event_log)
        
        control_panel.setLayout(control_layout)

        # --- Right Video Panel ---
        video_panel = QWidget()
        video_layout = QVBoxLayout()
        self.video_label = QLabel("System Offline")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("background-color: black;")
        video_layout.addWidget(self.video_label)
        video_panel.setLayout(video_layout)
        
        main_layout.addWidget(control_panel)
        main_layout.addWidget(video_panel)
        central_widget.setLayout(main_layout)

    def update_image(self, cv_img):
        """Updates the video_label with a new opencv image."""
        qt_img = self.convert_cv_qt(cv_img)
        self.video_label.setPixmap(qt_img)

    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap."""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.video_label.width(), self.video_label.height(), Qt.AspectRatioMode.KeepAspectRatio)
        return QPixmap.fromImage(p)

    def log_event(self, message, level="INFO"):
        """Appends a message to the event log."""
        timestamp = time.strftime("%H:%M:%S")
        self.event_log.append(f"[{timestamp}] [{level}] {message}")

    def update_status(self, text, color):
        """Updates the system status label."""
        self.status_label.setText(text)
        self.status_label.setStyleSheet(f"color: {color};")

    def start_system(self):
        """Starts the video thread."""
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.log_event("System started. Initializing camera...", "INFO")
        
        self.thread = VideoThread(self.known_data, self.criminal_list)
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.log_signal.connect(self.log_event)
        self.thread.status_signal.connect(self.update_status)
        self.thread.start()

    def stop_system(self):
        """Stops the video thread."""
        if self.thread:
            self.thread.stop()
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.video_label.setText("System Offline")
        self.video_label.setStyleSheet("background-color: black; color: white;")
        self.update_status("OFFLINE", "gray")

    def closeEvent(self, event):
        """Handles the window closing event."""
        self.stop_system()
        event.accept()

# --- Main Execution ---
if __name__=="__main__":
    app = QApplication(sys.argv)
    a = App()
    a.show()
    sys.exit(app.exec())
