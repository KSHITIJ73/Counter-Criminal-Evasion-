
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

    def _init_(self, known_data, criminal_list):
        super()._init_()
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
    def _init_(self):
        super()._init_()
        self.setWindowTitle(APP_TITLE)
        self.setGeometry(100, 100, 1000, 700)
        self.setStyleSheet("background-color: #212121; color: #EAEAEA;")
        
        self.known_data = self.load_encodings()
        self.criminal_list = {"kshitij"}
        
        self.initUI()
        self.thread = None

        