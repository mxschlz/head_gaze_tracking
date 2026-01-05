import cv2
import mediapipe as mp
import numpy as np
from scipy import signal

# 1. Setup MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

# 2. Input Data (From your successful model run)
video_path = "G:\\Meine Ablage\\PhD\\misc\\MK_video_RAW.avi"

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0: fps = 30.0  # Fallback if FPS cannot be read

# rPPG Buffers
green_signal = []
window_size = int(fps * 6)  # 6-second sliding window for estimation
current_hr = 0.0

print("Playback starting... Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    filtered_signal = None
    # Convert to RGB for MediaPipe
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(img_rgb)

    # 3. Draw the Head Frame if a face is detected
    if results.detections:
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape

            # Convert relative coordinates to pixels
            x, y, w, h = int(bbox.xmin * iw), int(bbox.ymin * ih), \
                int(bbox.width * iw), int(bbox.height * ih)

            # Draw the Green Box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # --- rPPG Signal Extraction & Estimation ---
            # 1. Extract ROI (Center 50% to avoid background/edges)
            roi = frame[y+int(h*0.25):y+int(h*0.75), x+int(w*0.25):x+int(w*0.75)]
            
            # 2. Get Mean Green Channel (Strongest PPG signal)
            g_mean = np.mean(roi[:, :, 1])
            green_signal.append(g_mean)

            # 3. Process Sliding Window
            if len(green_signal) > window_size:
                green_signal.pop(0)
                
                # Detrend and Filter (0.7Hz - 3.5Hz -> 42 - 210 BPM)
                detrended = signal.detrend(green_signal)
                b, a = signal.butter(3, [0.7 / (fps/2), 3.5 / (fps/2)], btype='bandpass')
                filtered = signal.filtfilt(b, a, detrended)
                filtered_signal = filtered
                
                # FFT to find dominant frequency (Heart Rate)
                # Use zero-padding (n=2048) for higher frequency resolution
                n_fft = 2048
                fft_mag = np.abs(np.fft.rfft(filtered, n=n_fft))
                freqs = np.fft.rfftfreq(n_fft, 1/fps)
                new_hr = freqs[np.argmax(fft_mag)] * 60

                # Smooth the HR estimate (Exponential Moving Average)
                if current_hr == 0.0:
                    current_hr = new_hr
                else:
                    current_hr = 0.9 * current_hr + 0.1 * new_hr

            # 4. Add Information Overlays
            # Top: Heart Rate
            cv2.putText(frame, f"Heart Rate: {current_hr:.1f} BPM", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # 4. Plot the Heart Beat Signal (Pulse Wave)
    h, w = frame.shape[:2]
    plot_h = 150
    plot_area = np.zeros((plot_h, w, 3), dtype=np.uint8)

    if filtered_signal is not None:
        # Normalize signal to fit plot area
        min_val, max_val = np.min(filtered_signal), np.max(filtered_signal)
        if max_val - min_val > 1e-5:
            norm_signal = (filtered_signal - min_val) / (max_val - min_val)
            # Map to plot coordinates (invert Y because 0 is top)
            ys = ((1 - norm_signal) * (plot_h - 20) + 10).astype(np.int32)
            xs = np.linspace(0, w, len(norm_signal)).astype(np.int32)
            pts = np.column_stack((xs, ys))
            cv2.polylines(plot_area, [pts], isClosed=False, color=(0, 255, 0), thickness=2)

    # 5. Display the Video
    combined_frame = np.vstack((frame, plot_area))
    cv2.imshow("PhD Heart Rate Analysis - MediaPipe Render", combined_frame)

    if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
face_detection.close()