import tkinter as tk
from tkinter import font as tkfont
from tkinter import messagebox, scrolledtext
from PIL import Image, ImageTk
import cv2
import pickle
import face_recognition
import threading
import time

# --- Constants and Configuration ---
APP_TITLE = "C.C.E: Counter Criminal Evasion System"
BG_COLOR = "#212121"
TEXT_COLOR = "#EAEAEA"
ACCENT_COLOR = "#4CAF50"  # Green
ALERT_COLOR = "#F44336"   # Red
FONT_BOLD = ("Helvetica", 12, "bold")
FONT_NORMAL = ("Helvetica", 10)

# --- Main Application Class ---
class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.configure(bg=BG_COLOR)
        
        # --- State Variables ---
        self.video_source = 0 # 0 for webcam, or use an RTMP URL string
        self.is_running = False
        self.video_thread = None
        
        # --- Load Face Encodings ---
        self.known_data = self.load_encodings("encodings.pickle")
        self.criminal_list = {"arjun"} # Example criminal list

        # --- UI Setup ---
        self.create_widgets()
        
    def load_encodings(self, path):
        """Loads face encodings from a pickle file."""
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except FileNotFoundError:
            messagebox.showerror("Error", f"Encoding file not found at '{path}'. Please run encode_faces.py first.")
            self.root.quit()
            return None

    def create_widgets(self):
        """Creates and arranges all the UI widgets."""
        # Main frame
        main_frame = tk.Frame(self.root, bg=BG_COLOR)
        main_frame.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

        # --- Left Control Panel ---
        control_panel = tk.Frame(main_frame, bg="#2C2C2C", width=250)
        control_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        control_panel.pack_propagate(False)

        tk.Label(control_panel, text="CONTROL PANEL", fg=ACCENT_COLOR, bg="#2C2C2C", font=FONT_BOLD).pack(pady=10)

        # Buttons
        self.btn_start = tk.Button(control_panel, text="Start System", command=self.start_system, font=FONT_NORMAL, bg=ACCENT_COLOR, fg="white")
        self.btn_start.pack(fill=tk.X, padx=10, pady=5)

        self.btn_stop = tk.Button(control_panel, text="Stop System", command=self.stop_system, font=FONT_NORMAL, bg=ALERT_COLOR, fg="white", state=tk.DISABLED)
        self.btn_stop.pack(fill=tk.X, padx=10, pady=5)
        
        # Separator
        tk.Frame(control_panel, height=2, bg=BG_COLOR).pack(fill=tk.X, pady=10)

        # Status Indicators
        tk.Label(control_panel, text="SYSTEM STATUS", fg=TEXT_COLOR, bg="#2C2C2C", font=FONT_BOLD).pack(pady=(10,0))
        self.status_label = tk.Label(control_panel, text="OFFLINE", fg="gray", bg="#2C2C2C", font=("Helvetica", 14, "bold"))
        self.status_label.pack(pady=5)

        # Event Log
        tk.Label(control_panel, text="EVENT LOG", fg=TEXT_COLOR, bg="#2C2C2C", font=FONT_BOLD).pack(pady=(20,0))
        self.event_log = scrolledtext.ScrolledText(control_panel, height=15, bg="#1A1A1A", fg=TEXT_COLOR, font=("Consolas", 9), state=tk.DISABLED)
        self.event_log.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # --- Right Video Panel ---
        video_panel = tk.Frame(main_frame, bg="black")
        video_panel.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)

        self.video_label = tk.Label(video_panel, text="System Offline", bg="black", fg="white", font=("Helvetica", 20))
        self.video_label.pack(expand=True, fill=tk.BOTH)

    def log_event(self, message, level="INFO"):
        """Logs a message to the event log with a timestamp."""
        timestamp = time.strftime("%H:%M:%S")
        full_message = f"[{timestamp}] [{level}] {message}\n"
        
        self.event_log.config(state=tk.NORMAL)
        self.event_log.insert(tk.END, full_message)
        self.event_log.see(tk.END) # Auto-scroll
        self.event_log.config(state=tk.DISABLED)

    def start_system(self):
        """Starts the video processing thread."""
        if self.is_running:
            return
            
        self.is_running = True
        self.btn_start.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        self.status_label.config(text="RUNNING", fg=ACCENT_COLOR)
        
        self.log_event("System started. Initializing camera...")
        
        # Run the video loop in a separate thread to not freeze the UI
        self.video_thread = threading.Thread(target=self.video_loop)
        self.video_thread.daemon = True
        self.video_thread.start()

    def stop_system(self):
        """Stops the video processing thread."""
        self.is_running = False
        self.btn_start.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)
        self.status_label.config(text="OFFLINE", fg="gray")
        self.video_label.config(image='', text="System Offline") # Clear image
        self.log_event("System stopped by user.")

    def video_loop(self):
        """The main video processing loop."""
        cap = cv2.VideoCapture(self.video_source)
        if not cap.isOpened():
            self.log_event("Failed to open video source.", "ERROR")
            self.stop_system()
            return

        self.log_event("Camera initialized successfully.")
        
        while self.is_running:
            ret, frame = cap.read()
            if not ret:
                self.log_event("Failed to grab frame from camera.", "WARNING")
                time.sleep(0.5)
                continue

            # --- INSERT YOUR FACE RECOGNITION LOGIC HERE ---
            # This is where you process the frame
            processed_frame, names = self.process_frame(frame)
            
            # Check for criminals and update status
            is_criminal_detected = any(name in self.criminal_list for name in names)
            if is_criminal_detected:
                self.status_label.config(text="ALERT!", fg=ALERT_COLOR)
                self.log_event(f"Criminal detected: {[n for n in names if n in self.criminal_list]}", "ALERT")
            else:
                 self.status_label.config(text="RUNNING", fg=ACCENT_COLOR)

            # --- Update UI with the new frame ---
            # Convert the OpenCV frame (BGR) to a PIL Image (RGB)
            img = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            
            # Convert the PIL Image to a Tkinter PhotoImage
            imgtk = ImageTk.PhotoImage(image=img)
            
            # Update the video label with the new image
            self.video_label.imgtk = imgtk
            self.video_label.config(image=imgtk)
            
        cap.release()

    def process_frame(self, frame):
        """
        Processes a single frame for face recognition.
        This is where you integrate the logic from your original main.py.
        """
        # Resize frame for faster processing (optional)
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Find all the faces and face encodings in the current frame
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        detected_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(self.known_data["encodings"], face_encoding, tolerance=0.6)
            name = "Unknown"

            if True in matches:
                first_match_index = matches.index(True)
                name = self.known_data["names"][first_match_index]
            
            detected_names.append(name)

        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, detected_names):
            # Scale back up face locations since the frame we detected in was scaled
            top *= 2
            right *= 2
            bottom *= 2
            left *= 2

            # Set box color
            box_color = ACCENT_COLOR # Green for known
            if name in self.criminal_list:
                box_color = ALERT_COLOR # Red for criminal
            elif name == "Unknown":
                box_color = (255, 165, 0) # Orange for unknown

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), box_color, 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), box_color, cv2.FILLED)
            label_font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), label_font, 1.0, (255, 255, 255), 1)

        return frame, detected_names

# --- Main Execution ---
if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()
