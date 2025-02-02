
## Features
- **Real-Time Eye Tracking**: Tracks and visualizes iris and eye corner positions in real-time using webcam input.
- **Facial Landmark Detection**: Detects and displays up to 478 facial landmarks.
- **Data Logging**: Records tracking data to CSV files, including timestamps, eye positions, and optional facial landmark data. *Note: Enabling logging of all 478 facial landmarks can result in large log files.*
- **Socket Communication**: Supports transmitting only iris tracking data via UDP sockets for integration with other systems or applications.
- **Blink Detection**: Monitors and records blink frequency, enhancing eye movement analysis.
- **Real-Time Head Pose Estimation**: Accurately estimates the roll, pitch, and yaw of the user's head in real-time.
- **Filtering and Smoothing**: Implements filtering and smoothing algorithms to ensure stable and accurate head orientation readings.
- **Custom Real-Time Facial Landmark Visualization**: Utilize the `mediapipe_landmarks_test.py` script to visualize and track each of the MediaPipe facial landmark indices in real time. This feature is particularly useful for identifying the most relevant facial landmarks for your project and observing them directly in the video feed.

---

## Requirements
- Python 3.x
- OpenCV (opencv-python)
- MediaPipe (mediapipe)
- Other Python standard libraries: `math`, `socket`, `argparse`, `time`, `csv`, `datetime`, `os`

---

## Installation & Usage

1. **Clone the Repository:**
   ```
   git clone https://github.com/mxschlz/head_gaze_tracking.git
   ```

2. **Navigate to the Repository Directory:**
   ```
   cd head_gaze_tracking
   ```

3. **Install anaconda environment:**
   ```
   conda env create -f environment.yml
   ```

4. **Run the Application:**
   ```
   python retriev_head_gaze_data.py
   ```

---

## Config.py
- **USER_FACE_WIDTH**: The horizontal distance between the outer edges of the user's cheekbones in millimeters. Adjust this value based on your face width for accurate head pose estimation.
- **NOSE_TO_CAMERA_DISTANCE**: The distance from the tip of the nose to the camera lens in millimeters. Intended for future enhancements.
- **PRINT_DATA**: Enable or disable console data printing for debugging.
- **DEFAULT_WEBCAM**: Default camera source index. '0' usually refers to the built-in webcam.
- **SHOW_ALL_FEATURES**: Display all facial landmarks on the video feed if set to True.
- **LOG_DATA**: Enable or disable logging of data to a CSV file.
- **LOG_ALL_FEATURES**: Log all facial landmarks to the CSV file if set to True.
- **ENABLE_HEAD_POSE**: Enable the head position and orientation estimator.
- **LOG_FOLDER**: Directory for storing log files.
- **SERVER_IP**: IP address for UDP data transmission (default is localhost).
- **SERVER_PORT**: Port number for the server to listen on.
- **SHOW_ON_SCREEN_DATA**: Display blink count and head pose angles on the video feed if set to True.
- **EYES_BLINK_FRAME_COUNTER**: Counter for consecutive frames with detected potential blinks.
- **BLINK_THRESHOLD**: Eye aspect ratio threshold for blink detection.
- **EYE_AR_CONSEC_FRAMES**: Number of consecutive frames below the threshold required to confirm a blink.
- **MIN_DETECTION_CONFIDENCE**: Confidence threshold for model detection.
- **MIN_TRACKING_CONFIDENCE**: Confidence threshold for model tracking.
- **MOVING_AVERAGE_WINDOW**: Number of frames for calculating the moving average for smoothing angles.
- **SHOW_BLINK_COUNT_ON_SCREEN**: Toggle to show the blink count on the video feed.
- **IS_RECORDING**: Controls whether data is being logged automatically. Set to false to wait for the 'r' command to start logging.
- **SERVER_ADDRESS**: Tuple containing the SERVER_IP and SERVER_PORT for UDP communication.


---

## Interactive Commands

While running the Eye Tracking and Head Pose Estimation script, you can interact with the program using the following keyboard commands:

- **'c' Key**: Calibrate Head Pose
  - Pressing the 'c' key recalibrates the head pose estimation to the current orientation of the user's head. This sets the current head pose as the new reference point.

- **'r' Key**: Start/Stop Recording
  - Toggling the 'r' key starts or pauses the recording of data to log folder. 

- **'q' Key**: Quit Program
  - Pressing the 'q' key will exit the program. 


---
## Data Logging & Telemetry
- **CSV Logging**: The application generates CSV files with tracking data including timestamps, eye positions, and optional facial landmarks. These files are stored in the `logs` folder.

- **UDP Telemetry**: The application sends iris position data through UDP sockets as defined by `SERVER_IP` and `SERVER_PORT`. The data is sent in the following order: [Timestamp, Left Eye Center X, Left Eye Center Y, Left Iris Relative Pos Dx, Left Iris Relative Pos Dy].

### UDP Packet Structure
- **Packet Type**: Mixed (int64 for timestamp, int32 for other values)
- **Packet Structure**: 
  - Timestamp (int64)
  - Left Eye Center X (int32)
  - Left Eye Center Y (int32)
  - Left Iris Relative Pos Dx (int32)
  - Left Iris Relative Pos Dy (int32)
- **Packet Size**: 24 bytes (8 bytes for int64 timestamp, 4 bytes each for the four int32 values)

### Example Packets
- **Example**: 
  - Timestamp: 1623447890123
  - Left Eye Center X: 315
  - Left Eye Center Y: 225
  - Left Iris Relative Pos Dx: 66
  - Left Iris Relative Pos Dy: -3
  - Packet: [1623447890123, 315, 225, 66, -3]
