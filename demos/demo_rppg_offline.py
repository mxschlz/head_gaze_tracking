import cv2
import mediapipe as mp
import numpy as np
from scipy import signal

# 1. Setup MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

video_path = "/home/max/Insync/schulz.max5@gmail.com/GoogleDrive/PhD/misc/MK_video_RAW.avi"
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0: fps = 30.0

# Buffers
green_signal = []
hr_buffer = []
window_size = int(fps * 6)
current_hr = 0.0
filtered_signal = np.zeros(window_size)
freqs = np.zeros(1025)
fft_mag = np.zeros(1025)
out = None


def create_dashboard_element(title, size=(200, 200)):
	panel = np.zeros((size[1], size[0], 3), dtype=np.uint8)
	cv2.rectangle(panel, (0, 0), (size[0] - 1, size[1] - 1), (150, 150, 150), 1)
	cv2.putText(panel, title, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
	return panel


while cap.isOpened():
	ret, frame = cap.read()
	if not ret: break

	# Create UI Panels
	side_panel = np.zeros((frame.shape[0], 300, 3), dtype=np.uint8)
	img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	results = face_detection.process(img_rgb)

	if results.detections:
		for detection in results.detections:
			bbox = detection.location_data.relative_bounding_box
			ih, iw, _ = frame.shape
			x, y, w, h = int(bbox.xmin * iw), int(bbox.ymin * ih), int(bbox.width * iw), int(bbox.height * ih)

			# --- ROI Extraction ---
			x1, y1, x2, y2 = max(0, x), max(0, y), min(iw, x + w), min(ih, y + h)
			face_roi = frame[y1:y2, x1:x2]

			# 1. Extract Green Channel Signal
			inner_roi = frame[y + int(h * 0.25):y + int(h * 0.75), x + int(w * 0.25):x + int(w * 0.75)]
			if inner_roi.size > 0:
				g_mean = np.mean(inner_roi[:, :, 1])
				green_signal.append(g_mean)

			if len(green_signal) > window_size:
				green_signal.pop(0)
				# Processing
				detrended = signal.detrend(green_signal)
				b, a = signal.butter(3, [0.7 / (fps / 2), 3.5 / (fps / 2)], btype='bandpass')
				filtered_signal = signal.filtfilt(b, a, detrended)

				# FFT
				n_fft = 2048
				fft_mag = np.abs(np.fft.rfft(filtered_signal, n=n_fft))
				freqs = np.fft.rfftfreq(n_fft, 1 / fps)

				# Filter FFT for heart range 40-200 BPM
				valid_idxs = np.where((freqs >= 0.7) & (freqs <= 3.5))
				peak_idx = valid_idxs[0][np.argmax(fft_mag[valid_idxs])]
				new_hr = freqs[peak_idx] * 60
				current_hr = 0.9 * current_hr + 0.1 * new_hr

			# --- RENDER FANCY PLOTS ---
			# Plot 1: Power Spectrum (Top Left style)
			spec_plt = create_dashboard_element("Power Spectrum (FFT)", (300, 150))
			if len(fft_mag) > 0:
				f_mask = (freqs > 0.5) & (freqs < 3.0)
				pts_x = (freqs[f_mask] - 0.5) / 2.5 * 280 + 10
				pts_y = 140 - (fft_mag[f_mask] / (np.max(fft_mag) + 1e-6) * 100)
				pts = np.column_stack((pts_x, pts_y)).astype(np.int32)
				cv2.polylines(spec_plt, [pts], False, (255, 255, 0), 1)

			# Plot 2: Heart State (Phase Space Plot)
			state_plt = create_dashboard_element("Heart State", (300, 150))
			if len(filtered_signal) > 10:
				grad = np.gradient(filtered_signal[-50:])
				# Normalize and center
				sig_norm = (filtered_signal[-50:] - np.mean(filtered_signal[-50:])) / (
							np.std(filtered_signal[-50:]) + 1e-6)
				grad_norm = (grad - np.mean(grad)) / (np.std(grad) + 1e-6)
				pts = np.column_stack((sig_norm * 40 + 150, grad_norm * 40 + 75)).astype(np.int32)
				cv2.polylines(state_plt, [pts], False, (255, 0, 255), 2)

			# Plot 3: ATTN Map (ROI Heatmap)
			attn_plt = create_dashboard_element("ATTN Map", (300, 150))
			if face_roi.size > 0:
				small_roi = cv2.resize(face_roi, (100, 100))
				heatmap = cv2.applyColorMap(small_roi[:, :, 1], cv2.COLORMAP_JET)
				attn_plt[20:120, 100:200] = heatmap

			# Stack side panel
			side_panel = np.vstack((spec_plt, attn_plt, state_plt))
			side_panel = cv2.resize(side_panel, (300, frame.shape[0]))

			# Main Frame Overlay
			cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
			cv2.putText(frame, f"HR: {current_hr:.1f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

	# Final Assembly
	combined = np.hstack((side_panel, frame))

	if out is None:
		fourcc = cv2.VideoWriter_fourcc(*'MJPG')
		out = cv2.VideoWriter('output.avi', fourcc, fps, (combined.shape[1], combined.shape[0]))
	out.write(combined)

	cv2.imshow("rPPG Diagnostic Monitor", combined)
	delay = int(1000 / fps)
	if cv2.waitKey(delay) & 0xFF == ord('q'): break

if out: out.release()
cap.release()
cv2.destroyAllWindows()
