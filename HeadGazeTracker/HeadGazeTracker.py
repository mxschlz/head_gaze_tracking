import numpy as np
import socket
import time
import csv
from datetime import datetime
import datetime as dt
import os
from AngleBuffer import AngleBuffer
import cv2 as cv
import mediapipe as mp
import yaml

# some good aesthetics
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=0)


class HeadGazeTracker(object):
	def __init__(self, subject_id=None, config_file_path="config.yaml", VIDEO_INPUT=None, VIDEO_OUTPUT=None, WEBCAM=0,
	             TRACKING_DATA_LOG_FOLDER=None, starting_timestamp=None, total_frames=None):
		self.load_config(file_path=config_file_path)
		if not starting_timestamp:
			self.starting_timestamp = datetime.now()
		elif starting_timestamp:
			self.starting_timestamp = datetime.strptime(str(starting_timestamp),
			                                            self.TIMESTAMP_FORMAT)  # must be UTC time (YYYYMMDDHHMMSSUUUUUU)
		if not total_frames:
			self.total_frames = 0
		elif total_frames:
			self.total_frames = total_frames
		self.subject_id = subject_id
		self.TOTAL_BLINKS = 0  # This will be reset if video is split

		# Store the base VIDEO_OUTPUT path from config. Actual output path will be derived.
		self.VIDEO_OUTPUT_BASE = VIDEO_OUTPUT
		self.VIDEO_INPUT = VIDEO_INPUT  # Passed directly
		self.TRACKING_DATA_LOG_FOLDER = TRACKING_DATA_LOG_FOLDER  # Passed directly
		self.WEBCAM = WEBCAM  # Passed directly

		self.initial_pitch, self.initial_yaw, self.initial_roll = None, None, None
		self.calibrated = False  # Calibration persists across parts
		self.SERVER_ADDRESS = (self.SERVER_IP, self.SERVER_PORT)
		self.IS_RECORDING = False
		self.EYES_BLINK_FRAME_COUNTER = 0
		self.cap = self.init_video_input()
		self.FPS = self.cap.get(cv.CAP_PROP_FPS)
		if self.FPS == 0:
			print("Warning: Video FPS reported as 0. Defaulting to 30 FPS for calculations.")
			self.FPS = 30.0

		self.face_mesh = self.init_face_mesh()
		self.socket = self.init_socket()
		self.csv_data = []  # Reset per part
		self._setup_column_names()

		# Video splitting parameters
		self.split_at_ms = getattr(self, "SPLIT_VIDEO_AT_MS", None)
		self.output_suffix_part1 = getattr(self, "OUTPUT_FILENAME_SUFFIX_PART1", "_part1")
		self.output_suffix_part2 = getattr(self, "OUTPUT_FILENAME_SUFFIX_PART2", "_part2")
		self.current_video_part = 1
		self.split_triggered_and_finalized = False  # To ensure part 1 is finalized only once

		# Initialize video output for the first part (or whole video if no split)
		current_output_suffix = self.output_suffix_part1 if self.split_at_ms is not None else ""
		if self.VIDEO_OUTPUT_BASE:
			self.out = self.init_video_output(part_suffix=current_output_suffix)
		else:
			self.out = None

		if self.ENABLE_VIDEO_TRIAL_DETECTION:
			self.trial_counter = 0  # Reset per part
			self.current_trial_data = None  # Reset per part
			self.all_trials_summary = []  # Reset per part
			self.last_trial_end_time_ms = -self.MIN_INTER_TRIAL_INTERVAL_MS  # Reset per part
			self.roi_brightness_samples = []  # Reset per part
			self.roi_baseline_brightness = None  # Reset per part
			self._validate_trial_detection_config()

		self.auto_calibrate_pending = getattr(self, "AUTO_CALIBRATE_ON_START", True)
		self.angle_buffer = AngleBuffer(size=self.MOVING_AVERAGE_WINDOW)  # Reset per part

		self._reset_per_frame_state()

		# Head Pose Auto-Calibration Attributes
		self.head_pose_calibration_samples = {'pitch': [], 'yaw': [], 'roll': []}
		self.head_pose_calibration_frame_counter = 0
		# self.auto_calibrate_pending is already loaded from config (AUTO_CALIBRATE_ON_START)
		# self.calibrated is already initialized

		# Ensure default values if not in config for these new params
		self.HEAD_POSE_AUTO_CALIBRATION_ENABLED = getattr(self, "HEAD_POSE_AUTO_CALIBRATION_ENABLED", False)
		self.HEAD_POSE_AUTO_CALIB_DURATION_FRAMES = getattr(self, "HEAD_POSE_AUTO_CALIB_DURATION_FRAMES", 150)
		self.HEAD_POSE_AUTO_CALIB_MIN_SAMPLES = getattr(self, "HEAD_POSE_AUTO_CALIB_MIN_SAMPLES", 30)
		self.HEAD_POSE_AUTO_CALIB_EYE_DX_RANGE = getattr(self, "HEAD_POSE_AUTO_CALIB_EYE_DX_RANGE", [-7, 7])
		self.HEAD_POSE_AUTO_CALIB_EYE_DY_RANGE = getattr(self, "HEAD_POSE_AUTO_CALIB_EYE_DY_RANGE", [-7, 7])

	def _reset_per_frame_state(self):
		"""Resets variables that store state for the current frame."""
		self.adj_pitch, self.adj_yaw, self.adj_roll = 0.0, 0.0, 0.0
		self.smooth_pitch, self.smooth_yaw, self.smooth_roll = 0.0, 0.0, 0.0
		self.raw_head_pitch, self.raw_head_yaw, self.raw_head_roll = 0.0, 0.0, 0.0

		self.l_cx, self.l_cy, self.r_cx, self.r_cy = 0, 0, 0, 0
		self.l_dx, self.l_dy, self.r_dx, self.r_dy = 0.0, 0.0, 0.0, 0.0

		self.face_looks_display_text = ""
		self.gaze_on_stimulus_display_text = ""
		self.is_looking_down_explicitly = False
		self.current_roi_brightness = 0.0

	def _setup_column_names(self):
		self.column_names = [
			"Timestamp (ms)", "Frame Nr",
			"Left Eye Center X", "Left Eye Center Y",
			"Right Eye Center X", "Right Eye Center Y",
			"Left Iris Relative Pos Dx", "Left Iris Relative Pos Dy",
			"Right Iris Relative Pos Dx", "Right Iris Relative Pos Dy",
			"Total Blink Count",
		]
		if self.ENABLE_HEAD_POSE:
			self.column_names.extend(["Pitch", "Yaw", "Roll"])
		if self.LOG_ALL_FEATURES:
			num_landmarks = 478 if self.USE_ATTENTION_MESH else 468
			self.column_names.extend(
				[f"Landmark_{i}_X" for i in range(num_landmarks)]
				+ [f"Landmark_{i}_Y" for i in range(num_landmarks)]
				+ ([f"Landmark_{i}_Z" for i in range(num_landmarks)] if self.LOG_Z_COORD else [])
			)

	def _validate_trial_detection_config(self):
		if not hasattr(self, 'ROI_BRIGHTNESS_THRESHOLD_FACTOR') and \
				not hasattr(self, 'ROI_ABSOLUTE_BRIGHTNESS_THRESHOLD'):
			print("WARNING: Neither ROI_BRIGHTNESS_THRESHOLD_FACTOR nor ROI_ABSOLUTE_BRIGHTNESS_THRESHOLD is set.")
			print("Trial detection might not work. Defaulting to a high absolute threshold.")
			self.ROI_ABSOLUTE_BRIGHTNESS_THRESHOLD = 250

		if not (isinstance(self.STIMULUS_ROI_COORDS, list) and len(self.STIMULUS_ROI_COORDS) == 4):
			raise ValueError("STIMULUS_ROI_COORDS must be a list of 4 integers [x, y, w, h]")

		if getattr(self, 'ENABLE_EYE_GAZE_CHECK', False):
			required_eye_configs = [
				'STIMULUS_LEFT_IRIS_DX_RANGE', 'STIMULUS_LEFT_IRIS_DY_RANGE',
				'STIMULUS_RIGHT_IRIS_DX_RANGE', 'STIMULUS_RIGHT_IRIS_DY_RANGE'
			]
			for cfg_item in required_eye_configs:
				if not hasattr(self, cfg_item) or not (
						isinstance(getattr(self, cfg_item), list) and len(getattr(self, cfg_item)) == 2):
					raise ValueError(
						f"'{cfg_item}' must be a list of 2 numbers [min, max] if ENABLE_EYE_GAZE_CHECK is True")

			if getattr(self, 'ENABLE_HEAD_POSE_FILTER_FOR_EYE_GAZE', False):
				required_filter_configs = ['HEAD_POSE_FILTER_PITCH_RANGE', 'HEAD_POSE_FILTER_YAW_RANGE']
				for cfg_item in required_filter_configs:
					if not hasattr(self, cfg_item) or not (
							isinstance(getattr(self, cfg_item), list) and len(getattr(self, cfg_item)) == 2):
						raise ValueError(
							f"'{cfg_item}' must be a list of 2 numbers [min, max] if ENABLE_HEAD_POSE_FILTER_FOR_EYE_GAZE is True")

			if not hasattr(self, 'DOWNWARD_LOOK_LEFT_IRIS_DY_MIN'):
				print(
					"Warning: DOWNWARD_LOOK_LEFT_IRIS_DY_MIN not in config. Downward look detection for blinks might be affected.")
				self.DOWNWARD_LOOK_LEFT_IRIS_DY_MIN = 999
			if not hasattr(self, 'DOWNWARD_LOOK_RIGHT_IRIS_DY_MIN'):
				print(
					"Warning: DOWNWARD_LOOK_RIGHT_IRIS_DY_MIN not in config. Downward look detection for blinks might be affected.")
				self.DOWNWARD_LOOK_RIGHT_IRIS_DY_MIN = 999

	def load_config(self, file_path):
		try:
			with open(file_path, 'r') as file:
				config_data = yaml.safe_load(file)
			for key, value in config_data.items():
				setattr(self, key, value)
		except FileNotFoundError:
			print(f"Error: Configuration file not found at '{file_path}'")
			raise
		except yaml.YAMLError as e:
			print(f"Error parsing YAML configuration file: {e}")
			raise

	@staticmethod
	def vector_position(point1, point2):
		x1, y1 = point1.ravel()
		x2, y2 = point2.ravel()
		return x2 - x1, y2 - y1

	@staticmethod
	def euclidean_distance_3D(points):
		P0, P3, P4, P5, P8, P11, P12, P13 = points
		numerator = (np.linalg.norm(P3 - P13) ** 3 + np.linalg.norm(P4 - P12) ** 3 + np.linalg.norm(P5 - P11) ** 3)
		denominator = 3 * np.linalg.norm(P0 - P8) ** 3
		return numerator / denominator if denominator else 0

	def estimate_head_pose(self, landmarks, image_size):
		scale_factor = self.USER_FACE_WIDTH / 150.0
		model_points = np.array([
			(0.0, 0.0, 0.0), (0.0, -330.0 * scale_factor, -65.0 * scale_factor),
			(-225.0 * scale_factor, 170.0 * scale_factor, -135.0 * scale_factor),
			(225.0 * scale_factor, 170.0 * scale_factor, -135.0 * scale_factor),
			(-150.0 * scale_factor, -150.0 * scale_factor, -125.0 * scale_factor),
			(150.0 * scale_factor, -150.0 * scale_factor, -125.0 * scale_factor)])
		focal_length = image_size[1]
		center = (image_size[1] / 2, image_size[0] / 2)
		camera_matrix = np.array([[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
		                         dtype="double")
		dist_coeffs = np.zeros((4, 1))
		image_points = np.array([
			landmarks[self.NOSE_TIP_INDEX], landmarks[self.CHIN_INDEX],
			landmarks[self.LEFT_EYE_OUTER_CORNER], landmarks[self.RIGHT_EYE_OUTER_CORNER],
			landmarks[self.LEFT_MOUTH_CORNER], landmarks[self.RIGHT_MOUTH_CORNER]], dtype="double")

		try:
			(success, rotation_vector, translation_vector) = cv.solvePnP(model_points, image_points, camera_matrix,
			                                                             dist_coeffs, flags=cv.SOLVEPNP_ITERATIVE)
			if not success: return 0.0, 0.0, 0.0
			rotation_matrix, _ = cv.Rodrigues(rotation_vector)
			projection_matrix = np.hstack((rotation_matrix, translation_vector.reshape(-1, 1)))
			_, _, _, _, _, _, euler_angles = cv.decomposeProjectionMatrix(projection_matrix)
			pitch, yaw, roll = euler_angles.flatten()[:3]
			return self.normalize_pitch(pitch), yaw, roll
		except cv.error as e:
			return 0.0, 0.0, 0.0

	@staticmethod
	def normalize_pitch(pitch):
		if pitch > 180: pitch -= 360
		if pitch < -90:
			pitch = -(180 + pitch)
		elif pitch > 90:
			pitch = 180 - pitch
		return -pitch

	def blinking_ratio(self, landmarks):
		right_eye_ratio = self.euclidean_distance_3D(landmarks[self.RIGHT_EYE_POINTS])
		left_eye_ratio = self.euclidean_distance_3D(landmarks[self.LEFT_EYE_POINTS])
		return (right_eye_ratio + left_eye_ratio + 1) / 2

	def init_face_mesh(self):
		if self.PRINT_DATA:
			print("Initializing the face mesh and camera...")
			print(f"Head pose estimation is {'enabled' if self.ENABLE_HEAD_POSE else 'disabled'}.")
		return mp.solutions.face_mesh.FaceMesh(
			max_num_faces=self.MAX_NUM_FACES, refine_landmarks=self.USE_ATTENTION_MESH,
			min_detection_confidence=self.MIN_DETECTION_CONFIDENCE,
			min_tracking_confidence=self.MIN_TRACKING_CONFIDENCE)

	def init_video_input(self):
		if self.WEBCAM is None and self.VIDEO_INPUT:
			cap = cv.VideoCapture(self.VIDEO_INPUT)
			if not cap or not cap.isOpened():
				raise IOError(f"Cannot open video file: {self.VIDEO_INPUT}")
		elif self.VIDEO_INPUT is None:
			cap = cv.VideoCapture(self.WEBCAM)
			if not cap or not cap.isOpened():
				raise IOError(f"Cannot open webcam: {self.WEBCAM}")
		else:
			raise ValueError("Provide EITHER VIDEO_INPUT OR WEBCAM, not both or neither.")
		return cap

	def init_video_output(self, part_suffix=""):
		if not self.VIDEO_OUTPUT_BASE: return None  # Use VIDEO_OUTPUT_BASE

		base, ext = os.path.splitext(self.VIDEO_OUTPUT_BASE)
		# If a suffix is provided (like _part1), VIDEO_OUTPUT_BASE itself might already have a suffix from batch processor
		# We need to ensure the part_suffix is added correctly.
		# Let's assume VIDEO_OUTPUT_BASE is the *final* intended filename *without* part suffix.
		# The batch processor should pass a VIDEO_OUTPUT path that is the base for this specific run.
		# So, if batch processor passes "output/videoA_processed.mp4",
		# and part_suffix is "_part1", we want "output/videoA_processed_part1.mp4".

		# Let's adjust how VIDEO_OUTPUT_BASE is used.
		# The VIDEO_OUTPUT parameter passed to __init__ should be the base for this run.
		# So, self.VIDEO_OUTPUT_BASE should be what's passed to __init__.

		final_video_output_path = self.VIDEO_OUTPUT_BASE  # This is the path passed to __init__
		if part_suffix:
			name, extension = os.path.splitext(final_video_output_path)
			final_video_output_path = f"{name}{part_suffix}{extension}"

		output_dir = os.path.dirname(final_video_output_path)
		if output_dir: os.makedirs(output_dir, exist_ok=True)

		frame_width = int(self.cap.get(cv.CAP_PROP_FRAME_WIDTH))
		frame_height = int(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT))
		output_fps = self.cap.get(cv.CAP_PROP_FPS) or self.FPS

		fourcc_str = getattr(self, 'OUTPUT_VIDEO_FOURCC', 'XVID').upper()
		if len(fourcc_str) != 4:
			print(f"Warning: OUTPUT_VIDEO_FOURCC '{fourcc_str}' invalid. Defaulting to 'XVID'.")
			fourcc_str = 'XVID'
		fourcc = cv.VideoWriter_fourcc(*fourcc_str)

		if self.PRINT_DATA: print(
			f"Initializing video output: {final_video_output_path} with FOURCC: {fourcc_str}, FPS: {output_fps:.2f}")
		writer = cv.VideoWriter(final_video_output_path, fourcc, output_fps, (frame_width, frame_height))
		if not writer.isOpened():
			print(f"Error: Could not open video writer for {final_video_output_path} with FOURCC {fourcc_str}.")
			return None
		return writer

	@staticmethod
	def init_socket():
		return socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

	def _get_and_preprocess_frame(self):
		ret, frame = self.cap.read()
		if not ret:
			return None, 0, 0, False

		if self.FLIP_VIDEO: frame = cv.flip(frame, 1)
		if self.ROTATE == 90:
			frame = cv.rotate(frame, cv.ROTATE_90_CLOCKWISE)
		elif self.ROTATE == 180:
			frame = cv.rotate(frame, cv.ROTATE_180)
		elif self.ROTATE == -90 or self.ROTATE == 270:
			frame = cv.rotate(frame, cv.ROTATE_90_COUNTERCLOCKWISE)

		img_h, img_w = frame.shape[:2]
		return frame, img_h, img_w, True

	def _process_face_mesh(self, frame):
		rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
		results = self.face_mesh.process(rgb_frame)
		mesh_points = None
		mesh_points_3D_normalized = None
		face_landmarks_mp = None
		if results.multi_face_landmarks:
			face_landmarks_mp = results.multi_face_landmarks[0]
			img_h, img_w = frame.shape[:2]
			mesh_points = np.array(
				[np.multiply([p.x, p.y], [img_w, img_h]).astype(int)
				 for p in face_landmarks_mp.landmark])
			mesh_points_3D_normalized = np.array(
				[[n.x, n.y, n.z] for n in face_landmarks_mp.landmark])
		return results, face_landmarks_mp, mesh_points, mesh_points_3D_normalized

	def _extract_eye_features(self, mesh_points):
		try:
			(l_cx_f, l_cy_f), _ = cv.minEnclosingCircle(mesh_points[self.LEFT_EYE_IRIS])
			(r_cx_f, r_cy_f), _ = cv.minEnclosingCircle(mesh_points[self.RIGHT_EYE_IRIS])
			self.l_cx, self.l_cy = int(l_cx_f), int(l_cy_f)
			self.r_cx, self.r_cy = int(r_cx_f), int(r_cy_f)
			center_left = np.array([l_cx_f, l_cy_f], dtype=np.float32)
			center_right = np.array([r_cx_f, r_cy_f], dtype=np.float32)
			outer_left_corner = mesh_points[self.LEFT_EYE_OUTER_CORNER].astype(np.float32)
			outer_right_corner = mesh_points[self.RIGHT_EYE_OUTER_CORNER].astype(np.float32)
			self.l_dx, self.l_dy = self.vector_position(outer_left_corner, center_left)
			self.r_dx, self.r_dy = self.vector_position(outer_right_corner, center_right)
		except Exception:
			self.l_cx, self.l_cy, self.r_cx, self.r_cy = 0, 0, 0, 0
			self.l_dx, self.l_dy, self.r_dx, self.r_dy = 0.0, 0.0, 0.0, 0.0

	def _check_downward_look(self):
		if hasattr(self, 'DOWNWARD_LOOK_LEFT_IRIS_DY_MIN') and hasattr(self, 'DOWNWARD_LOOK_RIGHT_IRIS_DY_MIN'):
			if (self.l_dy > self.DOWNWARD_LOOK_LEFT_IRIS_DY_MIN and
					self.r_dy > self.DOWNWARD_LOOK_RIGHT_IRIS_DY_MIN):
				return True
		return False

	def _update_blink_count(self, mesh_points_3D_normalized):
		eyes_aspect_ratio = self.blinking_ratio(mesh_points_3D_normalized)
		if eyes_aspect_ratio <= self.BLINK_THRESHOLD:
			if not self.is_looking_down_explicitly:
				self.EYES_BLINK_FRAME_COUNTER += 1
			else:
				if self.EYES_BLINK_FRAME_COUNTER > 0:
					self.EYES_BLINK_FRAME_COUNTER = 0
		else:
			if self.EYES_BLINK_FRAME_COUNTER > self.EYE_AR_CONSEC_FRAMES:
				self.TOTAL_BLINKS += 1
			self.EYES_BLINK_FRAME_COUNTER = 0

	def _process_head_pose(self, mesh_points, img_h, img_w, key_pressed):
		self.raw_head_pitch, self.raw_head_yaw, self.raw_head_roll = self.estimate_head_pose(mesh_points,
		                                                                                     (img_h, img_w))
		self.angle_buffer.add([self.raw_head_pitch, self.raw_head_yaw, self.raw_head_roll])
		self.smooth_pitch, self.smooth_yaw, self.smooth_roll = self.angle_buffer.get_average()

		# --- New Automatic Head Pose Calibration Logic ---
		if self.HEAD_POSE_AUTO_CALIBRATION_ENABLED and self.auto_calibrate_pending and not self.calibrated:
			self.head_pose_calibration_frame_counter += 1

			# Check if eyes are looking relatively forward
			# self.l_dx, self.l_dy etc. should be populated by _extract_eye_features before this
			eye_dx_ok = (self.HEAD_POSE_AUTO_CALIB_EYE_DX_RANGE[0] <= self.l_dx <=
			             self.HEAD_POSE_AUTO_CALIB_EYE_DX_RANGE[1] and
			             self.HEAD_POSE_AUTO_CALIB_EYE_DX_RANGE[0] <= self.r_dx <=
			             self.HEAD_POSE_AUTO_CALIB_EYE_DX_RANGE[1])
			eye_dy_ok = (self.HEAD_POSE_AUTO_CALIB_EYE_DY_RANGE[0] <= self.l_dy <=
			             self.HEAD_POSE_AUTO_CALIB_EYE_DY_RANGE[1] and
			             self.HEAD_POSE_AUTO_CALIB_EYE_DY_RANGE[0] <= self.r_dy <=
			             self.HEAD_POSE_AUTO_CALIB_EYE_DY_RANGE[1])

			if eye_dx_ok and eye_dy_ok:
				self.head_pose_calibration_samples['pitch'].append(self.smooth_pitch)
				self.head_pose_calibration_samples['yaw'].append(self.smooth_yaw)
				self.head_pose_calibration_samples['roll'].append(self.smooth_roll)
				if self.PRINT_DATA and self.frame_count % 15 == 0:  # Occasional print
					print(
						f"HP Calib sample: P={self.smooth_pitch:.1f} Y={self.smooth_yaw:.1f}. N={len(self.head_pose_calibration_samples['pitch'])}")

			# Check if calibration period is over or enough samples collected
			if self.head_pose_calibration_frame_counter >= self.HEAD_POSE_AUTO_CALIB_DURATION_FRAMES:
				if len(self.head_pose_calibration_samples['pitch']) >= self.HEAD_POSE_AUTO_CALIB_MIN_SAMPLES:
					self.initial_pitch = np.mean(self.head_pose_calibration_samples['pitch'])
					self.initial_yaw = np.mean(self.head_pose_calibration_samples['yaw'])
					self.initial_roll = np.mean(self.head_pose_calibration_samples['roll'])
					self.calibrated = True
					self.auto_calibrate_pending = False  # Stop trying to auto-calibrate
					if self.PRINT_DATA:
						print(
							f"Head pose auto-calibrated using {len(self.head_pose_calibration_samples['pitch'])} samples.")
						print(
							f"Initial Pose: P={self.initial_pitch:.1f}, Y={self.initial_yaw:.1f}, R={self.initial_roll:.1f}")
				else:
					if self.PRINT_DATA:
						print(
							f"Head pose auto-calibration failed: Not enough suitable samples ({len(self.head_pose_calibration_samples['pitch'])} collected). Manual calibration ('c') may be needed.")
					self.auto_calibrate_pending = False  # Stop trying to auto-calibrate to prevent repeated messages
		# --- End of New Automatic Head Pose Calibration Logic ---

		# Manual calibration by key press (overrides auto-calibration)
		if key_pressed == ord('c'):
			self.initial_pitch, self.initial_yaw, self.initial_roll = self.smooth_pitch, self.smooth_yaw, self.smooth_roll
			self.calibrated = True
			self.auto_calibrate_pending = False  # Manual calibration also means we stop pending auto-calib
			if self.PRINT_DATA: print(
				f"Head pose recalibrated by user: P={self.initial_pitch:.1f}, Y={self.initial_yaw:.1f}, R={self.initial_roll:.1f}")

		if self.calibrated:
			self.adj_pitch = self.smooth_pitch - self.initial_pitch
			self.adj_yaw = self.smooth_yaw - self.initial_yaw
			self.adj_roll = self.smooth_roll - self.initial_roll
		else:
			# If not calibrated (either auto failed or disabled, and 'c' not pressed), use raw smoothed values
			self.adj_pitch, self.adj_yaw, self.adj_roll = self.smooth_pitch, self.smooth_yaw, self.smooth_roll

	def _get_face_looks_text(self):
		angle_y = self.adj_yaw if self.calibrated else self.smooth_yaw
		angle_x = self.adj_pitch if self.calibrated else self.smooth_pitch
		threshold = 10
		if angle_y < -threshold: return "Face: Left"
		if angle_y > threshold: return "Face: Right"
		if angle_x < -threshold: return "Face: Down"
		if angle_x > threshold: return "Face: Up"
		return "Face: Forward"

	def _calculate_roi_brightness(self, frame, img_h, img_w):
		x, y, w, h = self.STIMULUS_ROI_COORDS
		x, y = max(0, x), max(0, y)
		roi_x2, roi_y2 = min(img_w, x + w), min(img_h, y + h)
		if roi_x2 > x and roi_y2 > y:
			stimulus_roi = frame[y:roi_y2, x:roi_x2]
			gray_stimulus_roi = cv.cvtColor(stimulus_roi, cv.COLOR_BGR2GRAY)
			return np.mean(gray_stimulus_roi)
		if self.PRINT_DATA and self.frame_count % 100 == 0:
			print(f"Warning: STIMULUS_ROI_COORDS {self.STIMULUS_ROI_COORDS} invalid for frame size ({img_w}x{img_h}).")
		return 0.0

	def _update_trial_state(self, current_frame_time_ms):
		if hasattr(self, 'ROI_BRIGHTNESS_THRESHOLD_FACTOR') and self.roi_baseline_brightness is None:
			if len(self.roi_brightness_samples) < self.ROI_BRIGHTNESS_BASELINE_FRAMES:
				if not (self.current_trial_data and self.current_trial_data['active']):
					self.roi_brightness_samples.append(self.current_roi_brightness)
			elif self.roi_brightness_samples:
				self.roi_baseline_brightness = np.mean(self.roi_brightness_samples)
				if self.PRINT_DATA: print(f"ROI Baseline Brightness: {self.roi_baseline_brightness:.2f}")
			else:
				self.roi_baseline_brightness = 0

		trial_can_start = (current_frame_time_ms >= self.last_trial_end_time_ms + self.MIN_INTER_TRIAL_INTERVAL_MS)
		stimulus_detected_by_brightness = False
		if self.roi_baseline_brightness is not None and hasattr(self, 'ROI_BRIGHTNESS_THRESHOLD_FACTOR'):
			if self.current_roi_brightness > self.roi_baseline_brightness * self.ROI_BRIGHTNESS_THRESHOLD_FACTOR:
				stimulus_detected_by_brightness = True
		elif hasattr(self, 'ROI_ABSOLUTE_BRIGHTNESS_THRESHOLD'):
			if self.current_roi_brightness > self.ROI_ABSOLUTE_BRIGHTNESS_THRESHOLD:
				stimulus_detected_by_brightness = True

		if trial_can_start and stimulus_detected_by_brightness and \
				not (self.current_trial_data and self.current_trial_data['active']):
			self.trial_counter += 1
			start_t = current_frame_time_ms
			stim_end_t = start_t + self.STIMULUS_DURATION_MS
			trial_end_t = stim_end_t + self.POST_STIMULUS_TRIAL_DURATION_MS
			self.current_trial_data = {
				'id': self.trial_counter, 'start_time_ms': start_t,
				'stimulus_end_time_ms': stim_end_t, 'trial_end_time_ms': trial_end_t,
				'active': True, 'stimulus_frames_processed_gaze': 0,
				'frames_on_stimulus_area': 0, 'looked_final': 0}
			if self.PRINT_DATA: print(
				f"Trial {self.trial_counter} START @{start_t}ms (Part {self.current_video_part}). ROI Bright: {self.current_roi_brightness:.2f}")

		if self.current_trial_data and self.current_trial_data['active']:
			if current_frame_time_ms >= self.current_trial_data['trial_end_time_ms']:
				if self.PRINT_DATA: print(
					f"Trial {self.current_trial_data['id']} END @{current_frame_time_ms}ms (Part {self.current_video_part}).")
				if self.current_trial_data['stimulus_frames_processed_gaze'] > 0:
					perc = (self.current_trial_data['frames_on_stimulus_area'] /
					        self.current_trial_data['stimulus_frames_processed_gaze']) * 100
					if perc >= self.LOOK_TO_STIMULUS_THRESHOLD_PERCENT:
						self.current_trial_data['looked_final'] = 1
				self.all_trials_summary.append(self.current_trial_data.copy())
				self.last_trial_end_time_ms = self.current_trial_data['trial_end_time_ms']
				self.current_trial_data = None

	def _classify_gaze_for_current_trial(self, current_frame_time_ms):
		is_stim_period = (current_frame_time_ms >= self.current_trial_data['start_time_ms'] and
		                  current_frame_time_ms < self.current_trial_data['stimulus_end_time_ms'])
		if not is_stim_period:
			self.gaze_on_stimulus_display_text = ""
			return

		self.current_trial_data['stimulus_frames_processed_gaze'] += 1
		gaze_on_stim_area_this_frame = False

		if getattr(self, 'ENABLE_EYE_GAZE_CHECK', False):
			eye_gaze_dx_ok = (
					self.STIMULUS_LEFT_IRIS_DX_RANGE[0] <= self.l_dx <= self.STIMULUS_LEFT_IRIS_DX_RANGE[1] and
					self.STIMULUS_RIGHT_IRIS_DX_RANGE[0] <= self.r_dx <= self.STIMULUS_RIGHT_IRIS_DX_RANGE[1])
			eye_gaze_dy_ok = (
					self.STIMULUS_LEFT_IRIS_DY_RANGE[0] <= self.l_dy <= self.STIMULUS_LEFT_IRIS_DY_RANGE[1] and
					self.STIMULUS_RIGHT_IRIS_DY_RANGE[0] <= self.r_dy <= self.STIMULUS_RIGHT_IRIS_DY_RANGE[1])
			eye_gaze_ok = eye_gaze_dx_ok and eye_gaze_dy_ok

			if eye_gaze_ok:
				if getattr(self, 'ENABLE_HEAD_POSE_FILTER_FOR_EYE_GAZE', False):
					if self.calibrated:
						head_filter_ok = (
								self.HEAD_POSE_FILTER_PITCH_RANGE[0] <= self.adj_pitch <=
								self.HEAD_POSE_FILTER_PITCH_RANGE[1] and
								self.HEAD_POSE_FILTER_YAW_RANGE[0] <= self.adj_yaw <= self.HEAD_POSE_FILTER_YAW_RANGE[
									1])
						if head_filter_ok: gaze_on_stim_area_this_frame = True
				else:
					gaze_on_stim_area_this_frame = True
		elif self.ENABLE_HEAD_POSE and self.calibrated:
			gaze_on_stim_area_this_frame = (
					self.STIMULUS_PITCH_RANGE[0] <= self.adj_pitch <= self.STIMULUS_PITCH_RANGE[1] and
					self.STIMULUS_YAW_RANGE[0] <= self.adj_yaw <= self.STIMULUS_YAW_RANGE[1])

		if gaze_on_stim_area_this_frame:
			self.current_trial_data['frames_on_stimulus_area'] += 1
			self.gaze_on_stimulus_display_text = "GAZE ON STIMULUS"
		else:
			self.gaze_on_stimulus_display_text = "GAZE OFF STIMULUS"

	def _log_frame_data(self, current_frame_time_ms, frame_count, results_face_mesh, img_w, img_h):
		current_log_timestamp = 0
		if not self.starting_timestamp:  # This case might not be hit if starting_timestamp is always set in __init__
			current_log_timestamp = int(time.time() * 1000)
		elif self.starting_timestamp:
			# Timestamps should be absolute from the original video start,
			# or relative to part start if we adjust self.starting_timestamp for part 2.
			# For now, let's keep it absolute from original video start.
			# The frame_count is also absolute.
			frame_increment_for_log = dt.timedelta(seconds=(1.0 / self.FPS if self.FPS > 0 else (1.0 / 30.0)))
			current_log_timestamp_dt = self.starting_timestamp + (frame_increment_for_log * frame_count)
			current_log_timestamp = int(current_log_timestamp_dt.strftime(self.TIMESTAMP_FORMAT))

		log_entry = [current_log_timestamp, frame_count,
		             self.l_cx, self.l_cy, self.r_cx, self.r_cy,
		             self.l_dx, self.l_dy, self.r_dx, self.r_dy,
		             self.TOTAL_BLINKS]  # TOTAL_BLINKS is per-part
		if self.ENABLE_HEAD_POSE:
			log_entry.extend([self.adj_pitch, self.adj_yaw, self.adj_roll])

		if self.LOG_ALL_FEATURES:
			if results_face_mesh and results_face_mesh.multi_face_landmarks:
				lm_flat = []
				for p in results_face_mesh.multi_face_landmarks[0].landmark:
					lm_flat.extend([p.x * img_w, p.y * img_h])
					if self.LOG_Z_COORD: lm_flat.append(p.z)
				log_entry.extend(lm_flat)
			else:
				num_landmarks = 478 if self.USE_ATTENTION_MESH else 468
				num_coords = 3 if self.LOG_Z_COORD else 2
				log_entry.extend([0] * (num_landmarks * num_coords))
		self.csv_data.append(log_entry)

		if self.USE_SOCKET:
			packet = np.array([current_log_timestamp], dtype=np.int64).tobytes() + \
			         np.array([self.l_cx, self.l_cy, int(self.l_dx), int(self.l_dy)],
			                  dtype=np.int32).tobytes()
			self.socket.sendto(packet, self.SERVER_ADDRESS)
			if self.PRINT_DATA: print(f'Sent UDP packet to {self.SERVER_ADDRESS}')

	def _draw_on_screen_data(self, frame, results_face_mesh, img_h, img_w, current_frame_time_ms):
		font_face = cv.FONT_HERSHEY_SIMPLEX
		font_scale_main = 0.55
		font_scale_small = 0.45
		font_thickness = 1
		line_h = 22
		text_color_green = (0, 255, 0)
		text_color_orange = (0, 165, 255)
		text_color_red = (0, 0, 255)
		text_color_magenta = (255, 0, 255)
		text_color_cyan = (255, 255, 0)

		# Top-left column
		tl_x, y_pos = 10, 20
		# Display current part number
		cv.putText(frame, f"Part: {self.current_video_part}", (tl_x, y_pos), font_face, font_scale_small,
		           text_color_orange, font_thickness)
		y_pos += line_h - 5

		cv.putText(frame, f"Blinks: {self.TOTAL_BLINKS}", (tl_x, y_pos), font_face, font_scale_main, text_color_green,
		           font_thickness)
		y_pos += line_h
		if self.ENABLE_HEAD_POSE:
			# ... (rest of head pose display, unchanged)
			if self.calibrated:
				cv.putText(frame, f"Cal Pitch: {self.adj_pitch:.1f}", (tl_x, y_pos), font_face, font_scale_main,
				           text_color_green, font_thickness)
				y_pos += line_h
				cv.putText(frame, f"Cal Yaw: {self.adj_yaw:.1f}", (tl_x, y_pos), font_face, font_scale_main,
				           text_color_green, font_thickness)
				y_pos += line_h
				cv.putText(frame, f"Cal Roll: {self.adj_roll:.1f}", (tl_x, y_pos), font_face, font_scale_main,
				           text_color_green, font_thickness)
				y_pos += line_h
			else:
				cv.putText(frame, f"Raw Pitch: {self.smooth_pitch:.1f}", (tl_x, y_pos), font_face, font_scale_main,
				           text_color_orange, font_thickness)
				y_pos += line_h
				cv.putText(frame, f"Raw Yaw: {self.smooth_yaw:.1f}", (tl_x, y_pos), font_face, font_scale_main,
				           text_color_orange, font_thickness)
				y_pos += line_h
				cv.putText(frame, f"Raw Roll: {self.smooth_roll:.1f}", (tl_x, y_pos), font_face, font_scale_main,
				           text_color_orange, font_thickness)
				y_pos += line_h
				status_text = ""
				if not (results_face_mesh and results_face_mesh.multi_face_landmarks):
					status_text = "(No face for pose)"
				elif self.auto_calibrate_pending:
					status_text = "(Wait auto-calib)"
				elif not self.calibrated:
					status_text = "(Press 'c' to calib)"
				if status_text: cv.putText(frame, status_text, (tl_x, y_pos), font_face, font_scale_small,
				                           text_color_orange, font_thickness); y_pos += line_h - 5

		# Top-right column
		tr_y_pos, tr_x_anchor = 20, img_w - 10
		if self.ENABLE_VIDEO_TRIAL_DETECTION and self.current_trial_data and self.current_trial_data['active']:
			trial_id = self.current_trial_data['id']
			stim_t_left = (self.current_trial_data['stimulus_end_time_ms'] - current_frame_time_ms) / 1000.0
			trial_t_left = (self.current_trial_data['trial_end_time_ms'] - current_frame_time_ms) / 1000.0
			trial_text = f"Trial {trial_id}" + (
				f" (Stim: {stim_t_left:.1f}s)" if stim_t_left > 0 else f" (Post: {trial_t_left:.1f}s)")
			(w, _), _ = cv.getTextSize(trial_text, font_face, font_scale_main, font_thickness)
			cv.putText(frame, trial_text, (tr_x_anchor - w, tr_y_pos), font_face, font_scale_main, text_color_magenta,
			           font_thickness)
			tr_y_pos += line_h

		if self.face_looks_display_text:
			(w, _), _ = cv.getTextSize(self.face_looks_display_text, font_face, font_scale_main, font_thickness)
			cv.putText(frame, self.face_looks_display_text, (tr_x_anchor - w, tr_y_pos), font_face, font_scale_main,
			           text_color_green, font_thickness)
			tr_y_pos += line_h

		if self.ENABLE_VIDEO_TRIAL_DETECTION:  # ROI Box and Text
			x_r, y_r, w_r, h_r = self.STIMULUS_ROI_COORDS
			cv.rectangle(frame, (x_r, y_r), (min(img_w, x_r + w_r), min(img_h, y_r + h_r)), text_color_cyan, 1)
			roi_base = f"{self.roi_baseline_brightness:.1f}" if self.roi_baseline_brightness is not None else "Wait"
			roi_text = f"ROI: {self.current_roi_brightness:.1f} (Base: {roi_base})"
			(w, _), _ = cv.getTextSize(roi_text, font_face, font_scale_small, font_thickness)
			cv.putText(frame, roi_text, (tr_x_anchor - w, tr_y_pos), font_face, font_scale_small, text_color_cyan,
			           font_thickness)
			tr_y_pos += line_h

		if results_face_mesh and results_face_mesh.multi_face_landmarks:
			if getattr(self, 'ENABLE_EYE_GAZE_CHECK', False):
				l_eye_text = f"L Eye D(xy): ({self.l_dx:.1f}, {self.l_dy:.1f})"
				r_eye_text = f"R Eye D(xy): ({self.r_dx:.1f}, {self.r_dy:.1f})"
				(w, _), _ = cv.getTextSize(l_eye_text, font_face, font_scale_small, font_thickness)
				cv.putText(frame, l_eye_text, (tr_x_anchor - w, tr_y_pos), font_face, font_scale_small, text_color_cyan,
				           font_thickness)
				tr_y_pos += int(line_h * 0.8)
				(w, _), _ = cv.getTextSize(r_eye_text, font_face, font_scale_small, font_thickness)
				cv.putText(frame, r_eye_text, (tr_x_anchor - w, tr_y_pos), font_face, font_scale_small, text_color_cyan,
				           font_thickness)
				tr_y_pos += int(line_h * 0.8)
			if self.is_looking_down_explicitly:
				down_text = "Eyes: Explicitly Down"
				(w, _), _ = cv.getTextSize(down_text, font_face, font_scale_small, font_thickness)
				cv.putText(frame, down_text, (tr_x_anchor - w, tr_y_pos), font_face, font_scale_small,
				           text_color_orange, font_thickness)
				tr_y_pos += int(line_h * 0.8)

		if self.gaze_on_stimulus_display_text:
			color = text_color_green if "ON" in self.gaze_on_stimulus_display_text else text_color_red
			(w, _), _ = cv.getTextSize(self.gaze_on_stimulus_display_text, font_face, font_scale_main, font_thickness)
			cv.putText(frame, self.gaze_on_stimulus_display_text, (img_w // 2 - w // 2, 20), font_face, font_scale_main,
			           color, font_thickness)

		cv.putText(frame, f'FPS: {self.FPS:.1f}', (tl_x, img_h - 10), font_face, font_scale_main, text_color_green,
		           font_thickness)

		if results_face_mesh and results_face_mesh.multi_face_landmarks:
			face_landmarks_mp = results_face_mesh.multi_face_landmarks[0]
			if self.SHOW_ALL_FEATURES:
				mp_drawing.draw_landmarks(image=frame, landmark_list=face_landmarks_mp,
				                          connections=mp_face_mesh.FACEMESH_TESSELATION,
				                          landmark_drawing_spec=drawing_spec,
				                          connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
			else:
				if self.l_cx != 0 or self.l_cy != 0: cv.circle(frame, (self.l_cx, self.l_cy), 5, (255, 0, 255), -1,
				                                               cv.LINE_AA)
				if self.r_cx != 0 or self.r_cy != 0: cv.circle(frame, (self.r_cx, self.r_cy), 5, (255, 0, 255), -1,
				                                               cv.LINE_AA)
				if self.ENABLE_HEAD_POSE and hasattr(self, '_indices_pose') and self.mesh_points is not None:
					for idx in self._indices_pose:
						if 0 <= idx < len(self.mesh_points): cv.circle(frame, self.mesh_points[idx], 2, (0, 0, 255), -1,
						                                               cv.LINE_AA)

	def _write_video_frame(self, frame):
		if self.out and self.out.isOpened():
			self.out.write(frame)
		elif self.out and not self.out.isOpened() and self.VIDEO_OUTPUT_BASE and self.frame_count % 100 == 0:
			if self.PRINT_DATA: print(f"Warning: VideoWriter for current part is not open.")

	def _handle_key_presses(self, key_pressed):
		if key_pressed == ord('q'):
			if self.PRINT_DATA: print("Exiting program...")
			return True
		return False

	def _finalize_part(self, part_suffix=""):
		"""Finalizes writing logs and video for the current part."""
		if self.PRINT_DATA: print(f"Finalizing data for Part {self.current_video_part} with suffix '{part_suffix}'...")
		if self.out and self.out.isOpened():
			self.out.release()
			self.out = None  # Important to set to None
			if self.PRINT_DATA: print(f"Released video writer for Part {self.current_video_part}.")

		self._save_trial_summary(part_suffix)
		self._save_main_log(part_suffix)

	def _prepare_for_next_part(self, split_time_ms_absolute):
		"""Resets state for processing the next part of the video."""
		if self.PRINT_DATA: print(
			f"Preparing for Part {self.current_video_part + 1} starting around {split_time_ms_absolute}ms (original time)...")
		self.current_video_part += 1

		# Reset data accumulators
		self.csv_data = []
		self.all_trials_summary = []
		self.trial_counter = 0
		self.TOTAL_BLINKS = 0
		self.EYES_BLINK_FRAME_COUNTER = 0

		# Reset trial detection state
		self.current_trial_data = None
		# Adjust last_trial_end_time_ms to be relative to the new part's start,
		# but using the absolute timeline for comparison with current_frame_time_ms.
		# Effectively, a trial can start soon after the split if conditions are met.
		self.last_trial_end_time_ms = split_time_ms_absolute - self.MIN_INTER_TRIAL_INTERVAL_MS
		self.roi_brightness_samples = []
		self.roi_baseline_brightness = None

		# Reset angle buffer for smoothing
		self.angle_buffer = AngleBuffer(size=self.MOVING_AVERAGE_WINDOW)

		# Re-initialize video output for the new part
		if self.VIDEO_OUTPUT_BASE:
			self.out = self.init_video_output(part_suffix=self.output_suffix_part2)
			if self.out is None and self.PRINT_DATA:
				print(f"Warning: Failed to initialize video output for Part {self.current_video_part}")

		# Calibration (self.calibrated, self.initial_pitch, etc.) persists.
		# self.auto_calibrate_pending also persists (or could be reset if desired).
		# self.starting_timestamp (original video start) persists for absolute frame time calculation.

		if self.PRINT_DATA: print(f"State reset for Part {self.current_video_part}.")

	def _save_trial_summary(self, part_suffix=""):
		if not (self.ENABLE_VIDEO_TRIAL_DETECTION and self.all_trials_summary):
			if self.ENABLE_VIDEO_TRIAL_DETECTION and self.PRINT_DATA: print(
				f"No trials to summarize for current part{part_suffix}.")
			return

		if self.current_trial_data and self.current_trial_data['active']:
			if self.PRINT_DATA: print(
				f"Finalizing active trial {self.current_trial_data['id']} on exit/split (Part {self.current_video_part}).")
			if self.current_trial_data['stimulus_frames_processed_gaze'] > 0:
				perc = (self.current_trial_data['frames_on_stimulus_area'] /
				        self.current_trial_data['stimulus_frames_processed_gaze']) * 100
				if perc >= self.LOOK_TO_STIMULUS_THRESHOLD_PERCENT:
					self.current_trial_data['looked_final'] = 1
			self.all_trials_summary.append(self.current_trial_data.copy())
			self.current_trial_data = None  # Clear after adding

		ts_str = datetime.now().strftime('%Y%m%d_%H%M%S')
		folder = self.TRACKING_DATA_LOG_FOLDER or "."
		subj = self.subject_id or 'NA'

		base_filename = self.OUTPUT_TRIAL_SUMMARY_FILENAME_PREFIX or "trial_summary_"
		summary_fn = os.path.join(folder, f"{subj}_{base_filename.replace('.csv', '')}{part_suffix}_{ts_str}.csv")

		os.makedirs(os.path.dirname(summary_fn), exist_ok=True)

		with open(summary_fn, "w", newline="") as f:
			writer = csv.writer(f)
			headers = ['trial_id', 'start_time_ms', 'stimulus_end_time_ms', 'trial_end_time_ms',
			           'stimulus_frames_processed_gaze', 'frames_on_stimulus_area', 'looked_at_stimulus']
			writer.writerow(headers)
			for trial_sum in self.all_trials_summary:
				writer.writerow([
					trial_sum['id'], trial_sum['start_time_ms'], trial_sum['stimulus_end_time_ms'],
					trial_sum['trial_end_time_ms'], trial_sum['stimulus_frames_processed_gaze'],
					trial_sum['frames_on_stimulus_area'], trial_sum['looked_final']])
		if self.PRINT_DATA: print(f"Trial summary saved: {summary_fn}")

	def _save_main_log(self, part_suffix=""):
		if not (self.LOG_DATA and self.csv_data):
			if self.LOG_DATA and self.PRINT_DATA: print(f"No main log data to write for current part{part_suffix}.")
			return

		if self.PRINT_DATA: print(f"Writing main log data for current part{part_suffix} to CSV...")
		ts_str = datetime.now().strftime('%Y%m%d_%H%M%S')
		folder = self.TRACKING_DATA_LOG_FOLDER or "."
		subj = self.subject_id or 'NA'

		base_filename = getattr(self, "OUTPUT_MAIN_LOG_FILENAME_PREFIX", "eye_tracking_log_")
		csv_fn = os.path.join(folder, f"{subj}_{base_filename.replace('.csv', '')}{part_suffix}_{ts_str}.csv")

		os.makedirs(os.path.dirname(csv_fn), exist_ok=True)

		# Padding logic - consider if this is still desired per part, or only for the whole video.
		# If per part, total_frames would need to be for that part.
		# For now, let's assume padding is not strictly needed if splitting, or needs more complex handling.
		# The current padding uses self.total_frames which is for the whole video.
		# This might lead to excessive padding for part 1.
		# Let's simplify: no padding if splitting, or user ensures total_frames is for the first part if they want padding for it.

		# if self.total_frames > 0 and len(self.csv_data) < self.total_frames and self.current_video_part == 1: # Example: Pad only part 1
		# ... padding logic ...

		with open(csv_fn, "w", newline="") as file:
			writer = csv.writer(file)
			writer.writerow(self.column_names)
			writer.writerows(self.csv_data)
		if self.PRINT_DATA: print(f"Main log data saved: {csv_fn}")

	def run(self):
		self.frame_count = -1
		# self.angle_buffer is now initialized in __init__ and _prepare_for_next_part

		try:
			while True:
				self.frame_count += 1
				self._reset_per_frame_state()

				frame, img_h, img_w, ret = self._get_and_preprocess_frame()
				if not ret: break

				current_frame_time_ms = int(self.frame_count * (1000.0 / self.FPS))
				if self.PRINT_DATA and self.frame_count % 60 == 0:
					print(
						f"Frame Nr.: {self.frame_count}, Time: {current_frame_time_ms}ms (Part {self.current_video_part})")

				# --- Video Splitting Logic ---
				if self.split_at_ms is not None and not self.split_triggered_and_finalized and \
						current_frame_time_ms >= self.split_at_ms:
					if self.PRINT_DATA: print(f"Split point reached at {current_frame_time_ms}ms. Finalizing Part 1.")
					self._finalize_part(part_suffix=self.output_suffix_part1)
					self._prepare_for_next_part(current_frame_time_ms)  # Pass the actual split time
					self.split_triggered_and_finalized = True
				# Continue processing the current frame as the first frame of part 2
				# --- End Splitting Logic ---

				key_pressed = cv.waitKey(1) & 0xFF

				results_face_mesh, face_landmarks_mp, mesh_points, mesh_points_3D_normalized = self._process_face_mesh(
					frame)
				self.mesh_points = mesh_points

				if results_face_mesh and results_face_mesh.multi_face_landmarks:
					self._extract_eye_features(mesh_points)
					self.is_looking_down_explicitly = self._check_downward_look()
					self._update_blink_count(mesh_points_3D_normalized)

					if self.ENABLE_HEAD_POSE:
						self._process_head_pose(mesh_points, img_h, img_w, key_pressed)
						self.face_looks_display_text = self._get_face_looks_text()

				if self.ENABLE_VIDEO_TRIAL_DETECTION:
					self.current_roi_brightness = self._calculate_roi_brightness(frame, img_h, img_w)
					self._update_trial_state(current_frame_time_ms)
					if self.current_trial_data and self.current_trial_data['active'] and \
							results_face_mesh and results_face_mesh.multi_face_landmarks:
						self._classify_gaze_for_current_trial(current_frame_time_ms)

				if self.LOG_DATA:
					self._log_frame_data(current_frame_time_ms, self.frame_count, results_face_mesh, img_w, img_h)

				if self.SHOW_ON_SCREEN_DATA:
					self._draw_on_screen_data(frame, results_face_mesh, img_h, img_w, current_frame_time_ms)

				cv.imshow("Eye Tracking", frame)
				self._write_video_frame(frame)  # Writes to self.out, which is managed for parts

				if self._handle_key_presses(key_pressed):
					break
		except Exception as e:
			print(f"An error occurred in run loop: {e}")
			import traceback
			traceback.print_exc()
		finally:
			self.cap.release()
			# Finalize the last part being processed (either part 1 if no split, or part 2 if split)
			final_suffix = self.output_suffix_part2 if self.split_triggered_and_finalized else \
				(self.output_suffix_part1 if self.split_at_ms is not None else "")
			self._finalize_part(part_suffix=final_suffix)

			cv.destroyAllWindows()
			if hasattr(self, 'socket'): self.socket.close()
			if self.PRINT_DATA: print("Program exited.")


if __name__ == "__main__":
	config_path = "config.yaml"  # Ensure this path is correct
	try:
		# Example:
		# tracker = HeadGazeTracker(
		# subject_id="test_split",
		# config_file_path=config_path,
		# VIDEO_INPUT="path/to/your/long_video.mp4",
		# VIDEO_OUTPUT="output/test_split_processed.mp4", # Base name for output video
		# TRACKING_DATA_LOG_FOLDER="output/logs_test_split"
		# )
		tracker = HeadGazeTracker(config_file_path=config_path)  # For default testing from config
		tracker.run()
	except IOError as e:
		print(f"IOError during initialization or run: {e}")
	except Exception as e:
		print(f"Failed to initialize or run HeadGazeTracker: {e}")
		import traceback

		traceback.print_exc()
