# Counter-Criminal-Evasion-
Counter Criminal Evasion (C.C.E) System

The Counter Criminal Evasion System is an application developed in Python, OpenCV, and PyQt6 that provides real-time facial recognition and notification functionalities. It identifies faces from live video streams or CCTV footage and triggers alerts when it detects a known criminal or unauthorized individual. This system is intended for implementation in colleges, offices, or organizations to facilitate automated identity monitoring via CCTV surveillance.

 # Features
  1.Utilizes real-time facial detection and identification via webcam or CCTV camera. 
  2.Constructed with a graphical user interface (GUI) using PyQt6. 
  3.Displays a live feed from the camera with bounding boxes around detected faces and their names. 
  4.Triggers an alarm sound upon recognizing a criminal. 
  5.Presents a popup window featuring the individual’s name, photograph, and the location of the camera. 
  6.Includes control buttons for starting, stopping, and silencing the alarm. 
  7.Facilitates easy addition of new faces to the database. 
  8.Operates entirely offline after the initial setup.

# Project Structure
Counter-Criminal-Evasion/
│
├── pyqt_app.py          # Main application (GUI + detection logic).
├── encode_face.py       # Script to create facial encodings from dataset.
├── encodings.pickle     # Encoded facial data (generated automatically).
├── alarm.wav            # Alarm sound file.
│
├── dataset/             # Folder containing known faces.
│   ├── kshitij/         #This example dataset you can make it accourding to your choice.
│   │   ├── 1.jpg        #make sure your dataset folder name should be in small case.
│   │   ├── 2.jpg
│   ├── rutvik/
│       ├── 1.jpg
│       ├── 2.jpg
│
└── README.md            # Documentation file

# System Requirements
Software Requirements.
Python 3.10 or later.
Windows 10/11 (recommended).
OpenCV, face_recognition, PyQt6.
OS Windows 10 or 11.

# Hardware Requirements
Minimum 8 GB RAM  Recommended 16GB RAM or above.
i5 11Generation /Ryzen 5 or higher processor -recommended i5 12/13Gen.
Minimum NVIDIA graphic card  GTX/RTX 1650 or above for better prefomance use high prefomance Graphic Card.
HD webcam or IP CCTV camera.
SSD for faster processing (recommended).

# Installation Guide
# Step 1: Install Python
Check if Python is installed by running:-
  python --version
If not installed, download and install it from python.org/downloads.

# Step 2: Install Required Libraries
Open the terminal in your project folder and run:-
  pip install opencv-python face-recognition PyQt6
If you face an installation issue with face-recognition, use:-
  pip install cmake dlib
  pip install face-recognition

# Step 3: Run the Main App
Now start the system using:-
  python pyqt_app.py

You’ll see a window with:-
A Control Panel (Start, Stop, Alarm buttons).
A Video Feed (live camera view).
A Log window for system messages.

# How to Use the App
| Button         | Function.                               |
| Start System   | Starts the camera and face detection.   |
| Stop System    | Stops the camera and recognition.       |
| Stop Alarm     | Stops the ringing alarm sound manually. |

When a criminal (from your list) is detected:-
Red box appears around their face.
Alarm plays (alarm.wav).
Popup window appears with name, photo, and CCTV location.
Log message appears in the event log panel.

# Step 5: Mark Criminals in the Code
In pyqt_app.py, find this part (in the App class):-
  self.criminal_list = {
      "kshitij",
      "rutvik"
  }

Add or remove names here as needed.
(These names must match the folder names in your dataset.)

# Step 6: Switch From Webcam to CCTV (Optional)
By default, the app uses your laptop/PC camera:-
  cap = cv2.VideoCapture(0)
  
To use a CCTV or IP camera, replace it with your RTSP stream:-
  cap = cv2.VideoCapture("rtsp://username:password@192.168.1.5:554/stream1")

Works with most CCTV brands (CP Plus, HikVision, Dahua, etc.)
Get this RTSP link from your network admin or DVR/NVR settings.

# Step 7: Add or Change Alarm Sound
Place a file named alarm.wav in your main project folder.
You can use any WAV sound — siren, beep, buzzer, etc.
To change the sound, just replace the file (keep the same name).

# Step 8: Package It as an Executable (Optional)
If you want to run it on systems without Python:-

1. Install PyInstaller:-
    pip install pyinstaller

2. Create an EXE:-
    pyinstaller --onefile --noconsole pyqt_app.py

3. After a few minutes, check the dist/ folder — you’ll find:-
    dist/pyqt_app.exe

You can now run this .exe on any Windows system.
Perfect for installation in college CCTV rooms or guard control systems.

# Optional: Add New Faces Easily
To add new people:-
Create a new folder inside dataset/ (e.g., dataset/arjun/)
Add 1–3 photos inside it
Re-run:
  python encode_face.py
Restart your app — done

# Developed By
Project Name: Counter Criminal Evasion (C.C.E).
Developer: Kshitij Kumar, Rutvik Chitre.
Technologies Used: Python, OpenCV, face_recognition, PyQt6.
Purpose: Real-time criminal detection and alert system using face recognition.
