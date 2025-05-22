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
		self.TOTAL_BLINKS = 0
		self.VIDEO_INPUT = VIDEO_INPUT
		self.VIDEO_OUTPUT = VIDEO_OUTPUT  # This will be loaded from config
		self.TRACKING_DATA_LOG_FOLDER = TRACKING_DATA_LOG_FOLDER
		self.WEBCAM = WEBCAM
		self.initial_pitch, self.initial_yaw, self.initial_roll = None, None, None
		self.calibrated = False
		# SERVER_ADDRESS: Tuple containing the SERVER_IP and SERVER_PORT for UDP communication.
		self.SERVER_ADDRESS = (self.SERVER_IP, self.SERVER_PORT)
		self.IS_RECORDING = False
		self.EYES_BLINK_FRAME_COUNTER = (0)
		self.cap = self.init_video_input()
		self.FPS = self.cap.get(cv.CAP_PROP_FPS)
		if self.FPS == 0:  # Fallback if FPS is not reported
			print("Warning: Video FPS reported as 0. Defaulting to 30 FPS for calculations.")
			self.FPS = 30.0

		self.face_mesh = self.init_face_mesh()
		self.socket = self.init_socket()
		self.csv_data = []
		# Column names for CSV file
		self.column_names = [
			"Timestamp (ms)",
			"Frame Nr",
			"Left Eye Center X",
			"Left Eye Center Y",
			"Right Eye Center X",
			"Right Eye Center Y",
			"Left Iris Relative Pos Dx",
			"Left Iris Relative Pos Dy",
			"Right Iris Relative Pos Dx",
			"Right Iris Relative Pos Dy",
			"Total Blink Count",
		]
		if self.VIDEO_OUTPUT:  # Check if VIDEO_OUTPUT path is provided in config
			self.out = self.init_video_output()
		else:  # Ensure self.out exists even if not writing video
			self.out = None

		# New attributes for video-based trial detection
		if self.ENABLE_VIDEO_TRIAL_DETECTION:
			self.trial_counter = 0
			self.current_trial_data = None  # Stores info about the active trial
			self.all_trials_summary = []  # List to store summary of all detected trials
			self.last_trial_end_time_ms = -self.MIN_INTER_TRIAL_INTERVAL_MS  # Initialize to allow immediate first trial

			self.roi_brightness_samples = []
			self.roi_baseline_brightness = None
			if not hasattr(self, 'ROI_BRIGHTNESS_THRESHOLD_FACTOR') and not hasattr(self,
			                                                                        'ROI_ABSOLUTE_BRIGHTNESS_THRESHOLD'):
				print(
					"WARNING: Neither ROI_BRIGHTNESS_THRESHOLD_FACTOR nor ROI_ABSOLUTE_BRIGHTNESS_THRESHOLD is set in config.")
				print("Trial detection might not work. Defaulting to a high absolute threshold.")
				self.ROI_ABSOLUTE_BRIGHTNESS_THRESHOLD = 250  # A fallback

			# Ensure STIMULUS_ROI_COORDS is valid
			if not (isinstance(self.STIMULUS_ROI_COORDS, list) and len(self.STIMULUS_ROI_COORDS) == 4):
				raise ValueError("STIMULUS_ROI_COORDS must be a list of 4 integers [x, y, w, h]")
		# For head pose calibration logic (improved)
		self.auto_calibrate_pending = getattr(self, "AUTO_CALIBRATE_ON_START",
		                                      True)  # Add this to config if you want to disable auto-calibration on start

	def load_config(self, file_path):
		try:
			with open(file_path, 'r') as file:
				config_data = yaml.safe_load(file)

			# Set attributes directly from the config data
			for key, value in config_data.items():
				setattr(self, key, value)

		except FileNotFoundError:
			print(f"Error: Configuration file not found at '{file_path}'")
		# Consider re-raising or exiting if config is critical
		except yaml.YAMLError as e:
			print(f"Error parsing YAML configuration file: {e}")

	# Consider re-raising or exiting

	@staticmethod
	# Function to calculate vector position
	def vector_position(point1, point2):
		x1, y1 = point1.ravel()
		x2, y2 = point2.ravel()
		return x2 - x1, y2 - y1

	@staticmethod
	def euclidean_distance_3D(points):
		"""Calculates the Euclidean distance between two points in 3D space.

		Args:
			points: A list of 3D points.

		Returns:
			The Euclidean distance between the two points.
		"""
		P0, P3, P4, P5, P8, P11, P12, P13 = points
		numerator = (
				np.linalg.norm(P3 - P13) ** 3
				+ np.linalg.norm(P4 - P12) ** 3
				+ np.linalg.norm(P5 - P11) ** 3
		)
		denominator = 3 * np.linalg.norm(P0 - P8) ** 3
		distance = numerator / denominator
		return distance

	def estimate_head_pose(self, landmarks, image_size):
		scale_factor = self.USER_FACE_WIDTH / 150.0
		model_points = np.array([
			(0.0, 0.0, 0.0),
			(0.0, -330.0 * scale_factor, -65.0 * scale_factor),
			(-225.0 * scale_factor, 170.0 * scale_factor, -135.0 * scale_factor),
			(225.0 * scale_factor, 170.0 * scale_factor, -135.0 * scale_factor),
			(-150.0 * scale_factor, -150.0 * scale_factor, -125.0 * scale_factor),
			(150.0 * scale_factor, -150.0 * scale_factor, -125.0 * scale_factor)
		])
		focal_length = image_size[1]
		center = (image_size[1] / 2, image_size[0] / 2)
		camera_matrix = np.array(
			[[focal_length, 0, center[0]],
			 [0, focal_length, center[1]],
			 [0, 0, 1]], dtype="double"
		)
		dist_coeffs = np.zeros((4, 1))
		image_points = np.array([
			landmarks[self.NOSE_TIP_INDEX],
			landmarks[self.CHIN_INDEX],
			landmarks[self.LEFT_EYE_OUTER_CORNER],
			landmarks[self.RIGHT_EYE_OUTER_CORNER],
			landmarks[self.LEFT_MOUTH_CORNER],
			landmarks[self.RIGHT_MOUTH_CORNER]
		], dtype="double")
		(success, rotation_vector, translation_vector) = cv.solvePnP(model_points, image_points, camera_matrix,
		                                                             dist_coeffs, flags=cv.SOLVEPNP_ITERATIVE)
		rotation_matrix, _ = cv.Rodrigues(rotation_vector)
		projection_matrix = np.hstack((rotation_matrix, translation_vector.reshape(-1, 1)))
		_, _, _, _, _, _, euler_angles = cv.decomposeProjectionMatrix(projection_matrix)
		pitch, yaw, roll = euler_angles.flatten()[:3]
		pitch = self.normalize_pitch(pitch)
		return pitch, yaw, roll

	@staticmethod
	def normalize_pitch(pitch):
		if pitch > 180:
			pitch -= 360
		if pitch < -90:
			pitch = -(180 + pitch)
		elif pitch > 90:
			pitch = 180 - pitch
		pitch = -pitch
		return pitch

	def blinking_ratio(self, landmarks):
		right_eye_ratio = self.euclidean_distance_3D(landmarks[self.RIGHT_EYE_POINTS])
		left_eye_ratio = self.euclidean_distance_3D(landmarks[self.LEFT_EYE_POINTS])
		ratio = (right_eye_ratio + left_eye_ratio + 1) / 2
		return ratio

	@staticmethod
	def crop_bottom_half(image):
		half_height = image.shape[0] // 2
		cropped = image[half_height:, :, :]
		return cropped

	def init_face_mesh(self):
		if self.PRINT_DATA:
			print("Initializing the face mesh and camera...")
			head_pose_status = "enabled" if self.ENABLE_HEAD_POSE else "disabled"
			print(f"Head pose estimation is {head_pose_status}.")

		mp_face_mesh_instance = mp.solutions.face_mesh.FaceMesh(
			max_num_faces=self.MAX_NUM_FACES,
			refine_landmarks=self.USE_ATTENTION_MESH,
			min_detection_confidence=self.MIN_DETECTION_CONFIDENCE,
			min_tracking_confidence=self.MIN_TRACKING_CONFIDENCE,
		)
		return mp_face_mesh_instance

	def init_video_input(self):
		cap = None
		if self.WEBCAM is None:
			if self.VIDEO_INPUT:
				cap = cv.VideoCapture(self.VIDEO_INPUT)
			if not cap or not cap.isOpened():
				print(f"Error opening video file: {self.VIDEO_INPUT}")
				raise IOError(f"Cannot open video file: {self.VIDEO_INPUT}")
		elif self.VIDEO_INPUT is None:
			cap = cv.VideoCapture(self.WEBCAM)
			if not cap or not cap.isOpened():
				print(f"Error opening webcam: {self.WEBCAM}")
				raise IOError(f"Cannot open webcam: {self.WEBCAM}")
		else:
			raise ValueError(
				"Please provide EITHER a video file (VIDEO_INPUT) OR a webcam ID (WEBCAM), not both or neither.")
		return cap

	def init_video_output(self):
		# Ensure the directory exists
		# self.VIDEO_OUTPUT is loaded from config and should be the full path including filename and extension
		if not self.VIDEO_OUTPUT:  # Should not happen if self.out is being initialized due to VIDEO_OUTPUT being true
			print("Error: VIDEO_OUTPUT path is not defined for initializing video writer.")
			return None

		output_dir = os.path.dirname(self.VIDEO_OUTPUT)
		if output_dir and not os.path.exists(output_dir):
			os.makedirs(output_dir, exist_ok=True)

		frame_width = int(self.cap.get(cv.CAP_PROP_FRAME_WIDTH))
		frame_height = int(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT))
		output_fps = self.cap.get(cv.CAP_PROP_FPS)
		if output_fps == 0:
			output_fps = self.FPS  # Use self.FPS which has a fallback

		# Get the FOURCC from config, with a default of 'XVID'
		# Common FOURCCs:
		# 'mp4v' for .mp4 (widely compatible MPEG-4)
		# 'XVID' for .avi (MPEG-4 variant)
		# 'MJPG' for .avi (Motion JPEG)
		# 'X264' or 'H264' for .mp4 (H.264/AVC, might require specific OpenCV build/codecs)
		fourcc_str = getattr(self, 'OUTPUT_VIDEO_FOURCC', 'XVID').upper()
		if len(fourcc_str) != 4:
			print(f"Warning: OUTPUT_VIDEO_FOURCC '{fourcc_str}' is not 4 characters long. Defaulting to 'XVID'.")
			fourcc_str = 'XVID'

		fourcc = cv.VideoWriter_fourcc(*fourcc_str)

		if self.PRINT_DATA:
			print(f"Initializing video output: {self.VIDEO_OUTPUT} with FOURCC: {fourcc_str}, FPS: {output_fps:.2f}")

		writer = cv.VideoWriter(self.VIDEO_OUTPUT, fourcc, output_fps, (frame_width, frame_height))

		if not writer.isOpened():
			print(f"Error: Could not open video writer for path: {self.VIDEO_OUTPUT} with FOURCC: {fourcc_str}")
			print("Please check if the FOURCC is supported by your OpenCV build and system codecs,")
			print("and that the output directory is writable.")
			return None  # Return None if writer couldn't be opened
		return writer

	def send_data_through_socket(self, timestamp, l_cx, l_cy, l_dx, l_dy):
		packet = np.array([timestamp], dtype=np.int64).tobytes() + np.array([l_cx, l_cy, l_dx, l_dy],
		                                                                    dtype=np.int32).tobytes()
		self.socket.sendto(packet, self.SERVER_ADDRESS)
		if self.PRINT_DATA:
			print(f'Sent UDP packet to {self.SERVER_ADDRESS}')

	@staticmethod
	def init_socket():
		return socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

	def run(self):
		if self.ENABLE_HEAD_POSE:
			self.column_names.extend(["Pitch", "Yaw", "Roll"])

		if self.LOG_ALL_FEATURES:
			num_landmarks = 478 if self.USE_ATTENTION_MESH else 468
			self.column_names.extend(
				[f"Landmark_{i}_X" for i in range(num_landmarks)]
				+ [f"Landmark_{i}_Y" for i in range(num_landmarks)]
				+ ([f"Landmark_{i}_Z" for i in range(num_landmarks)] if self.LOG_Z_COORD else [])
			)

		frame_count = -1
		frame_increment_seconds = 1.0 / self.FPS if self.FPS > 0 else (1.0 / 30.0)
		frame_increment = dt.timedelta(seconds=frame_increment_seconds)

		adj_pitch, adj_yaw, adj_roll = 0.0, 0.0, 0.0

		current_roi_brightness = 0.0

		try:
			angle_buffer = AngleBuffer(size=self.MOVING_AVERAGE_WINDOW)

			while True:
				frame_count += 1
				ret, frame = self.cap.read()

				if not ret:
					if self.PRINT_DATA: print("End of video or cannot read frame.")
					break

				smooth_pitch, smooth_yaw, smooth_roll = 0.0, 0.0, 0.0
				face_looks_display_text = ""
				gaze_on_stimulus_display_text = ""

				current_frame_time_ms = int(frame_count * (1000.0 / self.FPS))

				if self.PRINT_DATA and frame_count % 60 == 0:
					print(f"Frame Nr.: {frame_count}, Time: {current_frame_time_ms}ms")

				if self.FLIP_VIDEO:
					frame = cv.flip(frame, 1)
				if self.ROTATE == 90:
					frame = cv.rotate(frame, cv.ROTATE_90_CLOCKWISE)
				elif self.ROTATE == 180:
					frame = cv.rotate(frame, cv.ROTATE_180)
				elif self.ROTATE == -90 or self.ROTATE == 270:
					frame = cv.rotate(frame, cv.ROTATE_90_COUNTERCLOCKWISE)

				img_h, img_w = frame.shape[:2]

				if self.ENABLE_VIDEO_TRIAL_DETECTION:
					x, y, w, h = self.STIMULUS_ROI_COORDS
					x = max(0, x)
					y = max(0, y)
					roi_x2 = min(img_w, x + w)
					roi_y2 = min(img_h, y + h)

					if roi_x2 > x and roi_y2 > y:
						stimulus_roi = frame[y:roi_y2, x:roi_x2]
						gray_stimulus_roi = cv.cvtColor(stimulus_roi, cv.COLOR_BGR2GRAY)
						current_roi_brightness = np.mean(gray_stimulus_roi)
					else:
						current_roi_brightness = 0
						if self.PRINT_DATA and frame_count % 100 == 0:
							print(
								f"Warning: STIMULUS_ROI_COORDS {self.STIMULUS_ROI_COORDS} invalid for frame size ({img_w}x{img_h}).")

					if hasattr(self, 'ROI_BRIGHTNESS_THRESHOLD_FACTOR') and self.roi_baseline_brightness is None:
						if len(self.roi_brightness_samples) < self.ROI_BRIGHTNESS_BASELINE_FRAMES:
							if not (self.current_trial_data and self.current_trial_data['active']):
								self.roi_brightness_samples.append(current_roi_brightness)
						else:
							if self.roi_brightness_samples:
								self.roi_baseline_brightness = np.mean(self.roi_brightness_samples)
								if self.PRINT_DATA: print(
									f"ROI Baseline Brightness: {self.roi_baseline_brightness:.2f}")
							else:
								self.roi_baseline_brightness = 0

					trial_can_start = (
								current_frame_time_ms >= self.last_trial_end_time_ms + self.MIN_INTER_TRIAL_INTERVAL_MS)
					stimulus_detected = False
					if self.roi_baseline_brightness is not None and hasattr(self, 'ROI_BRIGHTNESS_THRESHOLD_FACTOR'):
						if current_roi_brightness > self.roi_baseline_brightness * self.ROI_BRIGHTNESS_THRESHOLD_FACTOR:
							stimulus_detected = True
					elif hasattr(self, 'ROI_ABSOLUTE_BRIGHTNESS_THRESHOLD'):
						if current_roi_brightness > self.ROI_ABSOLUTE_BRIGHTNESS_THRESHOLD:
							stimulus_detected = True

					if trial_can_start and stimulus_detected and not (
							self.current_trial_data and self.current_trial_data['active']):
						self.trial_counter += 1
						start_t = current_frame_time_ms
						stim_end_t = start_t + self.STIMULUS_DURATION_MS
						trial_end_t = stim_end_t + self.POST_STIMULUS_TRIAL_DURATION_MS
						self.current_trial_data = {
							'id': self.trial_counter, 'start_time_ms': start_t,
							'stimulus_end_time_ms': stim_end_t, 'trial_end_time_ms': trial_end_t,
							'active': True, 'stimulus_frames_processed_gaze': 0,
							'frames_on_stimulus_area': 0, 'looked_final': 0
						}
						if self.PRINT_DATA: print(
							f"Trial {self.trial_counter} START @{start_t}ms. ROI Bright: {current_roi_brightness:.2f}")

					if self.current_trial_data and self.current_trial_data['active']:
						if current_frame_time_ms >= self.current_trial_data['trial_end_time_ms']:
							if self.PRINT_DATA: print(
								f"Trial {self.current_trial_data['id']} END @{current_frame_time_ms}ms.")
							if self.current_trial_data['stimulus_frames_processed_gaze'] > 0:
								perc = (self.current_trial_data['frames_on_stimulus_area'] /
								        self.current_trial_data['stimulus_frames_processed_gaze']) * 100
								if perc >= self.LOOK_TO_STIMULUS_THRESHOLD_PERCENT:
									self.current_trial_data['looked_final'] = 1
							self.all_trials_summary.append(self.current_trial_data.copy())
							self.last_trial_end_time_ms = self.current_trial_data['trial_end_time_ms']
							self.current_trial_data = None

				rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
				results = self.face_mesh.process(rgb_frame)

				l_cx, l_cy, r_cx, r_cy, l_dx, l_dy, r_dx, r_dy = (0,) * 8
				key_pressed = cv.waitKey(1) & 0xFF

				if results.multi_face_landmarks:
					face_landmarks_mp = results.multi_face_landmarks[0]
					mesh_points = np.array(
						[np.multiply([p.x, p.y], [img_w, img_h]).astype(int)
						 for p in face_landmarks_mp.landmark]
					)
					mesh_points_3D_normalized = np.array(
						[[n.x, n.y, n.z] for n in face_landmarks_mp.landmark]
					)

					eyes_aspect_ratio = self.blinking_ratio(mesh_points_3D_normalized)
					if eyes_aspect_ratio <= self.BLINK_THRESHOLD:
						self.EYES_BLINK_FRAME_COUNTER += 1
					else:
						if self.EYES_BLINK_FRAME_COUNTER > self.EYE_AR_CONSEC_FRAMES:
							self.TOTAL_BLINKS += 1
						self.EYES_BLINK_FRAME_COUNTER = 0

					(l_cx, l_cy), _ = cv.minEnclosingCircle(mesh_points[self.LEFT_EYE_IRIS])
					(r_cx, r_cy), _ = cv.minEnclosingCircle(mesh_points[self.RIGHT_EYE_IRIS])
					center_left = np.array([l_cx, l_cy], dtype=np.int32)
					center_right = np.array([r_cx, r_cy], dtype=np.int32)
					l_dx, l_dy = self.vector_position(mesh_points[self.LEFT_EYE_OUTER_CORNER], center_left)
					r_dx, r_dy = self.vector_position(mesh_points[self.RIGHT_EYE_OUTER_CORNER], center_right)

					if self.ENABLE_HEAD_POSE:
						raw_head_pitch, raw_head_yaw, raw_head_roll = self.estimate_head_pose(mesh_points,
						                                                                      (img_h, img_w))
						angle_buffer.add([raw_head_pitch, raw_head_yaw, raw_head_roll])
						smooth_pitch, smooth_yaw, smooth_roll = angle_buffer.get_average()

						if self.auto_calibrate_pending and self.initial_pitch is None:
							self.initial_pitch, self.initial_yaw, self.initial_roll = smooth_pitch, smooth_yaw, smooth_roll
							self.calibrated = True
							self.auto_calibrate_pending = False
							if self.PRINT_DATA: print("Head pose auto-calibrated (initial).")

						if key_pressed == ord('c'):
							self.initial_pitch, self.initial_yaw, self.initial_roll = smooth_pitch, smooth_yaw, smooth_roll
							self.calibrated = True
							if self.PRINT_DATA: print("Head pose recalibrated by user.")

						if self.calibrated:
							adj_pitch = smooth_pitch - self.initial_pitch
							adj_yaw = smooth_yaw - self.initial_yaw
							adj_roll = smooth_roll - self.initial_roll
						else:
							adj_pitch, adj_yaw, adj_roll = smooth_pitch, smooth_yaw, smooth_roll

						angle_y_for_looks = adj_yaw if self.calibrated else smooth_yaw
						angle_x_for_looks = adj_pitch if self.calibrated else smooth_pitch
						threshold_angle_looks = 10
						if angle_y_for_looks < -threshold_angle_looks:
							face_looks_display_text = "Face: Left"
						elif angle_y_for_looks > threshold_angle_looks:
							face_looks_display_text = "Face: Right"
						elif angle_x_for_looks < -threshold_angle_looks:
							face_looks_display_text = "Face: Down"
						elif angle_x_for_looks > threshold_angle_looks:
							face_looks_display_text = "Face: Up"
						else:
							face_looks_display_text = "Face: Forward"

						if self.ENABLE_VIDEO_TRIAL_DETECTION and self.current_trial_data and \
								self.current_trial_data['active'] and self.calibrated:
							is_stim_period = (current_frame_time_ms >= self.current_trial_data['start_time_ms'] and
							                  current_frame_time_ms < self.current_trial_data['stimulus_end_time_ms'])
							if is_stim_period:
								self.current_trial_data['stimulus_frames_processed_gaze'] += 1
								gaze_on_stim_area = (
										self.STIMULUS_PITCH_RANGE[0] <= adj_pitch <= self.STIMULUS_PITCH_RANGE[1] and
										self.STIMULUS_YAW_RANGE[0] <= adj_yaw <= self.STIMULUS_YAW_RANGE[1])
								if gaze_on_stim_area:
									self.current_trial_data['frames_on_stimulus_area'] += 1
									gaze_on_stimulus_display_text = "GAZE ON STIMULUS"
								else:
									gaze_on_stimulus_display_text = "GAZE OFF STIMULUS"

					if self.SHOW_ALL_FEATURES:
						mp_drawing.draw_landmarks(
							image=frame, landmark_list=face_landmarks_mp,
							connections=mp_face_mesh.FACEMESH_TESSELATION,
							landmark_drawing_spec=drawing_spec,
							connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
					else:
						cv.circle(frame, (int(l_cx), int(l_cy)), 5, (255, 0, 255), -1, cv.LINE_AA)
						cv.circle(frame, (int(r_cx), int(r_cy)), 5, (255, 0, 255), -1, cv.LINE_AA)
						if self.ENABLE_HEAD_POSE:
							# Ensure self._indices_pose is defined in config or __init__
							# Example: self._indices_pose = [self.NOSE_TIP_INDEX, self.CHIN_INDEX, ...]
							if hasattr(self, '_indices_pose'):
								for idx in self._indices_pose:
									if 0 <= idx < len(mesh_points):  # Bounds check
										cv.circle(frame, mesh_points[idx], 2, (0, 0, 255), -1, cv.LINE_AA)
					# else:
					# if self.PRINT_DATA and frame_count % 300 == 0 : print("Warning: self._indices_pose not defined for drawing head pose landmarks.")

				if self.LOG_DATA:
					current_log_timestamp = 0
					if not self.starting_timestamp:
						current_log_timestamp = int(time.time() * 1000)
					elif self.starting_timestamp:
						current_log_timestamp_dt = self.starting_timestamp + (frame_increment * frame_count)
						current_log_timestamp = int(current_log_timestamp_dt.strftime(self.TIMESTAMP_FORMAT))

					log_entry = [current_log_timestamp, frame_count, l_cx, l_cy, r_cx, r_cy, l_dx, l_dy, r_dx, r_dy,
					             self.TOTAL_BLINKS]
					if self.ENABLE_HEAD_POSE:
						log_entry.extend([adj_pitch, adj_yaw, adj_roll])

					if self.LOG_ALL_FEATURES and results.multi_face_landmarks:
						lm_flat = []
						for p in results.multi_face_landmarks[0].landmark:
							lm_flat.extend([p.x * img_w, p.y * img_h])
							if self.LOG_Z_COORD: lm_flat.append(p.z)
						log_entry.extend(lm_flat)
					elif self.LOG_ALL_FEATURES:
						num_landmarks = 478 if self.USE_ATTENTION_MESH else 468
						num_coords = 3 if self.LOG_Z_COORD else 2
						log_entry.extend([0] * (num_landmarks * num_coords))
					self.csv_data.append(log_entry)

					if self.USE_SOCKET:
						self.send_data_through_socket(current_log_timestamp, l_cx, l_cy, l_dx, l_dy)

				if self.SHOW_ON_SCREEN_DATA:
					font_face = cv.FONT_HERSHEY_SIMPLEX
					font_scale = 0.55  # General font scale
					font_thickness = 1
					line_h = 22
					text_color_green = (0, 255, 0)
					text_color_orange = (0, 165, 255)
					text_color_red = (0, 0, 255)
					text_color_magenta = (255, 0, 255)
					text_color_cyan = (255, 255, 0)

					# --- Top-left column ---
					tl_x = 10
					y_pos = 20  # Start y position for top-left

					# OLD LOGIC OF SETTING RECORDING MODE MANUALLY
					"""
					if self.IS_RECORDING:
						cv.circle(frame, (tl_x + 5, y_pos - 7), 7, text_color_red, -1)
						cv.putText(frame, "REC", (tl_x + 15, y_pos), font_face, font_scale, text_color_red,
						           font_thickness)
						y_pos += line_h"""

					cv.putText(frame, f"Blinks: {self.TOTAL_BLINKS}", (tl_x, y_pos), font_face, font_scale,
					           text_color_green, font_thickness)
					y_pos += line_h

					if self.ENABLE_HEAD_POSE:
						if self.calibrated:
							cv.putText(frame, f"Cal Pitch: {adj_pitch:.1f}", (tl_x, y_pos), font_face, font_scale,
							           text_color_green, font_thickness)
							y_pos += line_h
							cv.putText(frame, f"Cal Yaw: {adj_yaw:.1f}", (tl_x, y_pos), font_face, font_scale,
							           text_color_green, font_thickness)
							y_pos += line_h
							cv.putText(frame, f"Cal Roll: {adj_roll:.1f}", (tl_x, y_pos), font_face, font_scale,
							           text_color_green, font_thickness)
							y_pos += line_h
						else:
							cv.putText(frame, f"Raw Pitch: {smooth_pitch:.1f}", (tl_x, y_pos), font_face, font_scale,
							           text_color_orange, font_thickness)
							y_pos += line_h
							cv.putText(frame, f"Raw Yaw: {smooth_yaw:.1f}", (tl_x, y_pos), font_face, font_scale,
							           text_color_orange, font_thickness)
							y_pos += line_h
							cv.putText(frame, f"Raw Roll: {smooth_roll:.1f}", (tl_x, y_pos), font_face, font_scale,
							           text_color_orange, font_thickness)
							y_pos += line_h
							if not results.multi_face_landmarks:
								cv.putText(frame, "(No face for pose)", (tl_x, y_pos), font_face, 0.45,
								           text_color_orange, font_thickness)
								y_pos += line_h - 5
							elif self.auto_calibrate_pending:
								cv.putText(frame, "(Wait auto-calib)", (tl_x, y_pos), font_face, 0.45,
								           text_color_orange, font_thickness)
								y_pos += line_h - 5
							elif not self.calibrated:
								cv.putText(frame, "(Press 'c' to calib)", (tl_x, y_pos), font_face, 0.45,
								           text_color_orange, font_thickness)
								y_pos += line_h - 5

					# --- Top-right column ---
					tr_y_pos = 20  # Start y position for top-right
					tr_x_anchor = img_w - 10  # Anchor from right edge

					if self.ENABLE_VIDEO_TRIAL_DETECTION and self.current_trial_data and self.current_trial_data[
						'active']:
						trial_id = self.current_trial_data['id']
						stim_t_left = (self.current_trial_data['stimulus_end_time_ms'] - current_frame_time_ms) / 1000.0
						trial_t_left = (self.current_trial_data['trial_end_time_ms'] - current_frame_time_ms) / 1000.0
						trial_text = f"Trial {trial_id}"
						if stim_t_left > 0:
							trial_text += f" (Stim: {stim_t_left:.1f}s)"
						else:
							trial_text += f" (Post: {trial_t_left:.1f}s)"
						(w_text, _), _ = cv.getTextSize(trial_text, font_face, font_scale, font_thickness)
						cv.putText(frame, trial_text, (tr_x_anchor - w_text, tr_y_pos), font_face, font_scale,
						           text_color_magenta, font_thickness)
						tr_y_pos += line_h

					if face_looks_display_text:
						(w_text, _), _ = cv.getTextSize(face_looks_display_text, font_face, font_scale, font_thickness)
						cv.putText(frame, face_looks_display_text, (tr_x_anchor - w_text, tr_y_pos), font_face,
						           font_scale, text_color_green, font_thickness)
						tr_y_pos += line_h  # Increment tr_y_pos after drawing face direction

					# --- ROI Box and its Text (Moved to Top-Right, below Face Direction) ---
					if self.ENABLE_VIDEO_TRIAL_DETECTION:
						# Draw ROI Box itself (position unchanged, based on STIMULUS_ROI_COORDS)
						x_r, y_r, w_r, h_r = self.STIMULUS_ROI_COORDS
						cv.rectangle(frame, (x_r, y_r), (min(img_w, x_r + w_r), min(img_h, y_r + h_r)), text_color_cyan,
						             1)

						# Prepare ROI diagnostic text
						roi_base_text = f"{self.roi_baseline_brightness:.1f}" if self.roi_baseline_brightness is not None else "Wait"
						roi_disp_text = f"ROI: {current_roi_brightness:.1f} (Base: {roi_base_text})"

						# Position ROI diagnostic text in the top-right column, using current tr_y_pos
						roi_font_scale = 0.45  # Smaller font for ROI text, or use font_scale
						(roi_w_text, _), _ = cv.getTextSize(roi_disp_text, font_face, roi_font_scale, font_thickness)

						# tr_y_pos is already at the correct height to be below previous top-right text
						cv.putText(frame, roi_disp_text, (tr_x_anchor - roi_w_text, tr_y_pos), font_face,
						           roi_font_scale, text_color_cyan,
						           font_thickness)
						tr_y_pos += line_h  # Increment tr_y_pos for any potential future text in this column

					# --- Top-center text ---
					if gaze_on_stimulus_display_text:
						gaze_color = text_color_green if "ON" in gaze_on_stimulus_display_text else text_color_red
						(w_text, _), _ = cv.getTextSize(gaze_on_stimulus_display_text, font_face, font_scale,
						                                font_thickness)
						cv.putText(frame, gaze_on_stimulus_display_text, (img_w // 2 - w_text // 2, 20), font_face,
						           font_scale, gaze_color, font_thickness)

					# --- Bottom-left for FPS ---
					cv.putText(frame, f'FPS: {self.FPS:.1f}', (tl_x, img_h - 10), font_face, font_scale,
					           text_color_green, font_thickness)

				cv.imshow("Eye Tracking", frame)
				if self.out and self.out.isOpened():  # Check if writer is valid and opened
					self.out.write(frame)
				elif self.out and not self.out.isOpened() and self.VIDEO_OUTPUT and frame_count % 100 == 0:  # Print warning periodically
					if self.PRINT_DATA: print(
						f"Warning: VideoWriter for {self.VIDEO_OUTPUT} is not open. Cannot write frame.")
				# OLD LOGIC OF SETTING RECORDING MODE MANUALLY
				"""
				if key_pressed == ord('r'):
					self.IS_RECORDING = not self.IS_RECORDING
					if self.PRINT_DATA: print(f"Recording {'started' if self.IS_RECORDING else 'paused'}.")
				if key_pressed == ord('q'):
					if self.PRINT_DATA: print("Exiting program...")
					break"""

		except Exception as e:
			print(f"An error occurred in run loop: {e}")
			import traceback
			traceback.print_exc()

		finally:
			self.cap.release()
			if self.out and self.out.isOpened():
				self.out.release()
			cv.destroyAllWindows()
			if hasattr(self, 'socket'): self.socket.close()

			if self.ENABLE_VIDEO_TRIAL_DETECTION and self.all_trials_summary:
				if self.current_trial_data and self.current_trial_data['active']:
					if self.PRINT_DATA: print(f"Finalizing active trial {self.current_trial_data['id']} on exit.")
					if self.current_trial_data['stimulus_frames_processed_gaze'] > 0:
						perc = (self.current_trial_data['frames_on_stimulus_area'] /
						        self.current_trial_data['stimulus_frames_processed_gaze']) * 100
						if perc >= self.LOOK_TO_STIMULUS_THRESHOLD_PERCENT:
							self.current_trial_data['looked_final'] = 1
					self.all_trials_summary.append(self.current_trial_data.copy())

				ts_str = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
				folder = self.TRACKING_DATA_LOG_FOLDER if self.TRACKING_DATA_LOG_FOLDER else "."
				subj = self.subject_id if self.subject_id else 'NA'
				summary_fn = os.path.join(folder, f"{subj}_{self.OUTPUT_TRIAL_SUMMARY_FILENAME_PREFIX}{ts_str}.csv")
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
							trial_sum['frames_on_stimulus_area'], trial_sum['looked_final']
						])
				if self.PRINT_DATA: print(f"Trial summary: {summary_fn}")
			elif self.ENABLE_VIDEO_TRIAL_DETECTION:
				if self.PRINT_DATA: print("No trials detected/summarized.")

			if self.LOG_DATA and self.csv_data:
				if self.PRINT_DATA: print("Writing main log data to CSV...")
				ts_str = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
				folder = self.TRACKING_DATA_LOG_FOLDER if self.TRACKING_DATA_LOG_FOLDER else "."
				subj = self.subject_id if self.subject_id else 'NA'
				csv_fn = os.path.join(folder, f"{subj}_eye_tracking_log_{ts_str}.csv")
				os.makedirs(os.path.dirname(csv_fn), exist_ok=True)

				if self.total_frames > 0 and len(self.csv_data) < self.total_frames:
					if self.PRINT_DATA: print(
						f"Padding CSV data from {len(self.csv_data)} to {self.total_frames} frames.")
					num_cols = len(self.column_names)
					# last_valid_entry = self.csv_data[-1] if self.csv_data else [0] * num_cols # Not used

					for i in range(len(self.csv_data), self.total_frames):
						padded_frame_count = frame_count + (i - len(self.csv_data) + 1)
						padded_timestamp = 0
						if self.starting_timestamp:
							padded_timestamp_dt = self.starting_timestamp + (frame_increment * padded_frame_count)
							padded_timestamp = int(padded_timestamp_dt.strftime(self.TIMESTAMP_FORMAT))
						# else: # Fallback for timestamp if not starting_timestamp
						# padded_timestamp = last_valid_entry[0] + int(frame_increment_seconds * 1000) if self.csv_data else 0

						default_row_values = [0] * (num_cols - 2)
						padded_entry = [padded_timestamp, padded_frame_count] + default_row_values
						self.csv_data.append(padded_entry[:num_cols])

				with open(csv_fn, "w", newline="") as file:
					writer = csv.writer(file)
					writer.writerow(self.column_names)
					writer.writerows(self.csv_data)
				if self.PRINT_DATA: print(f"Main log data: {csv_fn}")
			elif self.LOG_DATA:
				if self.PRINT_DATA: print("No data in main log to write.")

			if self.PRINT_DATA: print("Program exited.")


if __name__ == "__main__":
	config_path = "config.yaml"
	# script_dir = os.path.dirname(os.path.abspath(__file__))
	# config_path = os.path.join(script_dir, "config.yaml")
	try:
		# Example: tracker = HeadGazeTracker(subject_id="test_subject", config_file_path=config_path)
		tracker = HeadGazeTracker(config_file_path=config_path)
		tracker.run()
	except IOError as e:  # Catch specific IOErrors from init_video_input/output
		print(f"IOError during initialization: {e}")
	except Exception as e:
		print(f"Failed to initialize or run HeadGazeTracker: {e}")
		import traceback

		traceback.print_exc()
