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


class HeadGazeTracker(object):
	def __init__(self, subject_id=None, config_file_path="config.yaml", VIDEO_INPUT=None, VIDEO_OUTPUT=None, WEBCAM=0,
				TRACKING_DATA_LOG_FOLDER=None, starting_timestamp=None, total_frames=None):
		self.load_config(file_path=config_file_path)
		self.starting_timestamp = datetime.strptime(str(starting_timestamp), self.TIMESTAMP_FORMAT)  # must be UTC time (YYYYMMDDHHMMSSUUUUUU)
		self.total_frames = total_frames
		self.subject_id = subject_id
		self.TOTAL_BLINKS = 0
		self.VIDEO_INPUT = VIDEO_INPUT
		self.VIDEO_OUTPUT = VIDEO_OUTPUT
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
		if self.VIDEO_OUTPUT:
			self.out = self.init_video_output()

	def load_config(self, file_path):
		try:
			with open(file_path, 'r') as file:
				config_data = yaml.safe_load(file)

			# Set attributes directly from the config data
			for key, value in config_data.items():
				setattr(self, key, value)

		except FileNotFoundError:
			print(f"Error: Configuration file not found at '{file_path}'")
		except yaml.YAMLError as e:
			print(f"Error parsing YAML configuration file: {e}")

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

			# Comment: This function calculates the Euclidean distance between two points in 3D space.
		"""

		# Get the three points.
		P0, P3, P4, P5, P8, P11, P12, P13 = points

		# Calculate the numerator.
		numerator = (
				np.linalg.norm(P3 - P13) ** 3
				+ np.linalg.norm(P4 - P12) ** 3
				+ np.linalg.norm(P5 - P11) ** 3
		)

		# Calculate the denominator.
		denominator = 3 * np.linalg.norm(P0 - P8) ** 3

		# Calculate the distance.
		distance = numerator / denominator

		return distance

	def estimate_head_pose(self, landmarks, image_size):
		# Scale factor based on user's face width (assumes model face width is 150mm)
		scale_factor = self.USER_FACE_WIDTH / 150.0
		# 3D model points.
		model_points = np.array([
			(0.0, 0.0, 0.0),  # Nose tip
			(0.0, -330.0 * scale_factor, -65.0 * scale_factor),  # Chin
			(-225.0 * scale_factor, 170.0 * scale_factor, -135.0 * scale_factor),  # Left eye left corner
			(225.0 * scale_factor, 170.0 * scale_factor, -135.0 * scale_factor),  # Right eye right corner
			(-150.0 * scale_factor, -150.0 * scale_factor, -125.0 * scale_factor),  # Left Mouth corner
			(150.0 * scale_factor, -150.0 * scale_factor, -125.0 * scale_factor)  # Right mouth corner
		])

		# Camera internals
		focal_length = image_size[1]
		center = (image_size[1] / 2, image_size[0] / 2)
		camera_matrix = np.array(
			[[focal_length, 0, center[0]],
			 [0, focal_length, center[1]],
			 [0, 0, 1]], dtype="double"
		)

		# Assuming no lens distortion
		dist_coeffs = np.zeros((4, 1))

		# 2D image points from landmarks, using defined indices
		image_points = np.array([
			landmarks[self.NOSE_TIP_INDEX],  # Nose tip
			landmarks[self.CHIN_INDEX],  # Chin
			landmarks[self.LEFT_EYE_LEFT_CORNER_INDEX],  # Left eye left corner
			landmarks[self.RIGHT_EYE_RIGHT_CORNER_INDEX],  # Right eye right corner
			landmarks[self.LEFT_MOUTH_CORNER_INDEX],  # Left mouth corner
			landmarks[self.RIGHT_MOUTH_CORNER_INDEX]  # Right mouth corner
		], dtype="double")

		# Solve for pose
		(success, rotation_vector, translation_vector) = cv.solvePnP(model_points, image_points, camera_matrix,
		                                                             dist_coeffs, flags=cv.SOLVEPNP_ITERATIVE)

		# Convert rotation vector to rotation matrix
		rotation_matrix, _ = cv.Rodrigues(rotation_vector)

		# Combine rotation matrix and translation vector to form a 3x4 projection matrix
		projection_matrix = np.hstack((rotation_matrix, translation_vector.reshape(-1, 1)))

		# Decompose the projection matrix to extract Euler angles
		_, _, _, _, _, _, euler_angles = cv.decomposeProjectionMatrix(projection_matrix)
		pitch, yaw, roll = euler_angles.flatten()[:3]

		# Normalize the pitch angle
		pitch = self.normalize_pitch(pitch)

		return pitch, yaw, roll

	@staticmethod
	def normalize_pitch(pitch):
		"""
		Normalize the pitch angle to be within the range of [-90, 90].

		Args:
			pitch (float): The raw pitch angle in degrees.

		Returns:
			float: The normalized pitch angle.
		"""
		# Map the pitch angle to the range [-180, 180]
		if pitch > 180:
			pitch -= 360

		# Invert the pitch angle for intuitive up/down movement
		pitch = -pitch

		# Ensure that the pitch is within the range of [-90, 90]
		if pitch < -90:
			pitch = -(180 + pitch)
		elif pitch > 90:
			pitch = 180 - pitch

		pitch = -pitch

		return pitch

	# This function calculates the blinking ratio of a person.
	def blinking_ratio(self, landmarks):
		"""Calculates the blinking ratio of a person.

		Args:
			landmarks: A facial landmarks in 3D normalized.

		Returns:
			The blinking ratio of the person, between 0 and 1, where 0 is fully open and 1 is fully closed.

		"""

		# Get the right eye ratio.
		right_eye_ratio = self.euclidean_distance_3D(landmarks[self.RIGHT_EYE_POINTS])

		# Get the left eye ratio.
		left_eye_ratio = self.euclidean_distance_3D(landmarks[self.LEFT_EYE_POINTS])

		# Calculate the blinking ratio.
		ratio = (right_eye_ratio + left_eye_ratio + 1) / 2

		return ratio

	# crop frame in half
	@staticmethod
	def crop_bottom_half(image):
		half_height = image.shape[0] // 2
		cropped = image[half_height:, :, :]
		return cropped

	def init_face_mesh(self):
		# Initializing MediaPipe face mesh and camera
		if self.PRINT_DATA:
			print("Initializing the face mesh and camera...")
			if self.PRINT_DATA:
				head_pose_status = "enabled" if self.ENABLE_HEAD_POSE else "disabled"
				print(f"Head pose estimation is {head_pose_status}.")

		mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
			max_num_faces=self.MAX_NUM_FACES,
			refine_landmarks=True,
			min_detection_confidence=self.MIN_DETECTION_CONFIDENCE,
			min_tracking_confidence=self.MIN_TRACKING_CONFIDENCE,
		)
		return mp_face_mesh

	def init_video_input(self):
		if self.WEBCAM is None:  # Check if a video file path is provided
			if self.VIDEO_INPUT:
				cap = cv.VideoCapture(self.VIDEO_INPUT)  # Replace with your file path
			if not cap.isOpened():
				print("Error opening video file")
				return  # Exit the function if the video can't be opened
			# Get the original FPS of the video
		else:  # Use the default webcam
			cap = cv.VideoCapture(self.WEBCAM)
			if not cap.isOpened():
				print("Error opening webcam")
				return  # Exit if webcam can't be opened
		return cap

	def init_video_output(self):
		return cv.VideoWriter(self.VIDEO_OUTPUT, cv.VideoWriter_fourcc(*'XVID'), self.cap.get(cv.CAP_PROP_FPS),
		                     (int(self.cap.get(3)), int(self.cap.get(4))))

	@staticmethod
	def init_socket():
		# Initializing socket for data transmission
		return socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

	def run(self):
		# Add head pose columns if head pose estimation is enabled
		if self.ENABLE_HEAD_POSE:
			self.column_names.extend(["Pitch", "Yaw", "Roll"])

		if self.LOG_ALL_FEATURES:
			self.column_names.extend(
				[f"Landmark_{i}_X" for i in range(468)]
				+ [f"Landmark_{i}_Y" for i in range(468)]
			)

		# self.init_data_handle()
		# Main loop for video capture and processing
		frame_count = -1  # count frames from -1 because 0 is first
		increment = dt.timedelta(seconds=1 / self.FPS)  # get frame duration

		try:
			angle_buffer = AngleBuffer(size=self.MOVING_AVERAGE_WINDOW)  # Adjust size for smoothing

			while True:
				ret, frame = self.cap.read()
				# frame = crop_bottom_half(frame)
				if not ret:
					break

				frame_count += 1  # add one per iteration
				print(frame_count)

				# Flipping the frame for a mirror effect
				# I think we better not flip to correspond with real world... need to make sure later...
				if self.FLIP_VIDEO:
					frame = cv.flip(frame, 1)
				rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
				img_h, img_w = frame.shape[:2]
				results = self.face_mesh.process(rgb_frame)

				if results.multi_face_landmarks:
					mesh_points = np.array(
						[
							np.multiply([p.x, p.y], [img_w, img_h]).astype(int)
							for p in results.multi_face_landmarks[0].landmark
						]
					)

					# Get the 3D landmarks from facemesh x, y and z(z is distance from 0 points)
					# just normalize values
					mesh_points_3D = np.array(
						[[n.x, n.y, n.z] for n in results.multi_face_landmarks[0].landmark]
					)
					# getting the head pose estimation 3d points
					head_pose_points_3D = np.multiply(
						mesh_points_3D[self._indices_pose], [img_w, img_h, 1]
					)
					head_pose_points_2D = mesh_points[self._indices_pose]

					# collect nose three dimension and two dimension points
					nose_3D_point = np.multiply(head_pose_points_3D[0], [1, 1, 3000])
					nose_2D_point = head_pose_points_2D[0]

					# create the camera matrix
					focal_length = 1 * img_w

					cam_matrix = np.array(
						[[focal_length, 0, img_h / 2], [0, focal_length, img_w / 2], [0, 0, 1]]
					)

					# The distortion parameters
					dist_matrix = np.zeros((4, 1), dtype=np.float64)

					head_pose_points_2D = np.delete(head_pose_points_3D, 2, axis=1)
					head_pose_points_3D = head_pose_points_3D.astype(np.float64)
					head_pose_points_2D = head_pose_points_2D.astype(np.float64)
					# Solve PnP
					success, rot_vec, trans_vec = cv.solvePnP(
						head_pose_points_3D, head_pose_points_2D, cam_matrix, dist_matrix
					)
					# Get rotational matrix
					rotation_matrix, jac = cv.Rodrigues(rot_vec)

					# Get angles
					angles, mtxR, mtxQ, Qx, Qy, Qz = cv.RQDecomp3x3(rotation_matrix)

					# Get the y rotation degree
					angle_x = angles[0] * 360
					angle_y = angles[1] * 360
					z = angles[2] * 360

					# if angle cross the values then
					threshold_angle = 10
					# See where the user's head tilting
					if angle_y < -threshold_angle:
						face_looks = "Left"
					elif angle_y > threshold_angle:
						face_looks = "Right"
					elif angle_x < -threshold_angle:
						face_looks = "Down"
					elif angle_x > threshold_angle:
						face_looks = "Up"
					else:
						face_looks = "Forward"
					if self.SHOW_ON_SCREEN_DATA:
						cv.putText(
							frame,
							f"Face Looking at {face_looks}",
							(img_w - 400, 80),
							cv.FONT_HERSHEY_TRIPLEX,
							0.8,
							(0, 255, 0),
							2,
							cv.LINE_AA,
						)
					# Display the nose direction
					nose_3d_projection, jacobian = cv.projectPoints(
						nose_3D_point, rot_vec, trans_vec, cam_matrix, dist_matrix
					)

					p1 = nose_2D_point
					p2 = (
						int(nose_2D_point[0] + angle_y * 10),
						int(nose_2D_point[1] - angle_x * 10),
					)

					cv.line(frame, p1, p2, (255, 0, 255), 3)
					# getting the blinking ratio
					eyes_aspect_ratio = self.blinking_ratio(mesh_points_3D)
					# print(f"Blinking ratio : {ratio}")
					# checking if ear less then or equal to required threshold if yes then
					# count the number of frame frame while eyes are closed.
					if eyes_aspect_ratio <= self.BLINK_THRESHOLD:
						self.EYES_BLINK_FRAME_COUNTER += 1
					# else check if eyes are closed is greater EYE_AR_CONSEC_FRAMES frame then
					# count the this as a blink
					# make frame counter equal to zero

					else:
						if self.EYES_BLINK_FRAME_COUNTER > self.EYE_AR_CONSEC_FRAMES:
							self.TOTAL_BLINKS += 1
						self.EYES_BLINK_FRAME_COUNTER = 0

					# Display all facial landmarks if enabled
					if self.SHOW_ALL_FEATURES:
						for point in mesh_points:
							cv.circle(frame, tuple(point), 1, (0, 255, 0), -1)
					# Process and display eye features
					(l_cx, l_cy), l_radius = cv.minEnclosingCircle(mesh_points[self.LEFT_EYE_IRIS])
					(r_cx, r_cy), r_radius = cv.minEnclosingCircle(mesh_points[self.RIGHT_EYE_IRIS])
					center_left = np.array([l_cx, l_cy], dtype=np.int32)
					center_right = np.array([r_cx, r_cy], dtype=np.int32)

					# Highlighting the irises and corners of the eyes
					cv.circle(
						frame, center_left, int(l_radius), (255, 0, 255), 2, cv.LINE_AA
					)  # Left iris
					cv.circle(
						frame, center_right, int(r_radius), (255, 0, 255), 2, cv.LINE_AA
					)  # Right iris
					cv.circle(
						frame, mesh_points[self.LEFT_EYE_INNER_CORNER][0], 3, (255, 255, 255), -1, cv.LINE_AA
					)  # Left eye right corner
					cv.circle(
						frame, mesh_points[self.LEFT_EYE_OUTER_CORNER][0], 3, (0, 255, 255), -1, cv.LINE_AA
					)  # Left eye left corner
					cv.circle(
						frame, mesh_points[self.RIGHT_EYE_INNER_CORNER][0], 3, (255, 255, 255), -1, cv.LINE_AA
					)  # Right eye right corner
					cv.circle(
						frame, mesh_points[self.RIGHT_EYE_OUTER_CORNER][0], 3, (0, 255, 255), -1, cv.LINE_AA
					)  # Right eye left corner

					# Calculating relative positions
					l_dx, l_dy = self.vector_position(mesh_points[self.LEFT_EYE_OUTER_CORNER], center_left)
					r_dx, r_dy = self.vector_position(mesh_points[self.RIGHT_EYE_OUTER_CORNER], center_right)

					# Printing data if enabled
					if self.PRINT_DATA:
						print(f"Total Blinks: {self.TOTAL_BLINKS}")
						print(f"Left Eye Center X: {l_cx} Y: {l_cy}")
						print(f"Right Eye Center X: {r_cx} Y: {r_cy}")
						print(f"Left Iris Relative Pos Dx: {l_dx} Dy: {l_dy}")
						print(f"Right Iris Relative Pos Dx: {r_dx} Dy: {r_dy}\n")
						# Check if head pose estimation is enabled
						if self.ENABLE_HEAD_POSE:
							pitch, yaw, roll = self.estimate_head_pose(mesh_points, (img_h, img_w))
							angle_buffer.add([pitch, yaw, roll])
							pitch, yaw, roll = angle_buffer.get_average()

							# Set initial angles on first successful estimation or recalibrate
							if self.initial_pitch is None or (key == ord('c') and self.calibrated):
								self.initial_pitch, self.initial_yaw, self.initial_roll = pitch, yaw, roll
								self.calibrated = True
								if self.PRINT_DATA:
									print("Head pose recalibrated.")

							# Adjust angles based on initial calibration
							if self.calibrated:
								pitch -= self.initial_pitch
								yaw -= self.initial_yaw
								roll -= self.initial_roll

							if self.PRINT_DATA:
								print(f"Head Pose Angles: Pitch={pitch}, Yaw={yaw}, Roll={roll}")
				elif not results.multi_face_landmarks:
					l_cx = l_cy = r_cx = r_cy = l_dx = l_dy = r_dx = r_dy = roll = pitch = yaw =0

				# Logging data
				if self.LOG_DATA:
					if not self.starting_timestamp:
						timestamp = int(time.time() * 1000)  # Current timestamp in milliseconds
					elif self.starting_timestamp:
						timestamp = self.starting_timestamp + increment * frame_count
						timestamp = int(timestamp.strftime(self.TIMESTAMP_FORMAT))
						print(timestamp)
					log_entry = [timestamp, frame_count, l_cx, l_cy, r_cx, r_cy, l_dx, l_dy, r_dx, r_dy,
					             self.TOTAL_BLINKS]  # Include blink count in CSV

					# Append head pose data if enabled
					if self.ENABLE_HEAD_POSE:
						log_entry.extend([pitch, yaw, roll])
					if self.LOG_ALL_FEATURES:
						log_entry.extend([p for point in mesh_points for p in point])
					self.csv_data.append(log_entry)

					# Sending data through socket
					timestamp = int(time.time() * 1000)  # Current timestamp in milliseconds
					# Create a packet with mixed types (int64 for timestamp and int32 for the rest)
					packet = np.array([timestamp], dtype=np.int64).tobytes() + np.array([l_cx, l_cy, l_dx, l_dy],
					                                                                    dtype=np.int32).tobytes()

					iris_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
					iris_socket.sendto(packet, self.SERVER_ADDRESS)

					print(f'Sent UDP packet to {self.SERVER_ADDRESS}: {packet}')

					# Writing the on screen data on the frame
					if self.SHOW_ON_SCREEN_DATA:
						if self.IS_RECORDING:
							cv.circle(frame, (30, 30), 10, (0, 0, 255), -1)  # Red circle at the top-left corner
						cv.putText(frame, f"Blinks: {self.TOTAL_BLINKS}", (30, 80), cv.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0),
						           2, cv.LINE_AA)
						if self.ENABLE_HEAD_POSE:
							cv.putText(frame, f"Pitch: {int(pitch)}", (30, 110), cv.FONT_HERSHEY_DUPLEX, 0.8,
							           (0, 255, 0), 2, cv.LINE_AA)
							cv.putText(frame, f"Yaw: {int(yaw)}", (30, 140), cv.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0),
							           2, cv.LINE_AA)
							cv.putText(frame, f"Roll: {int(roll)}", (30, 170), cv.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0),
							           2, cv.LINE_AA)

				# Displaying the processed frame
				cv.imshow("Eye Tracking", frame)
				if self.out:
					self.out.write(frame)  # Write the frame to the output video file
				# Handle key presses
				key = cv.waitKey(1) & 0xFF

				# Calibrate on 'c' key press
				if key == ord('c'):
					self.initial_pitch, self.initial_yaw, self.initial_roll = pitch, yaw, roll
					if self.PRINT_DATA:
						print("Head pose recalibrated.")

				# Inside the main loop, handle the 'r' key press
				if key == ord('r'):

					self.IS_RECORDING = not self.IS_RECORDING
					if self.IS_RECORDING:
						print("Recording started.")
					else:
						print("Recording paused.")

				# Exit on 'q' key press
				if key == ord('q'):
					if self.PRINT_DATA:
						print("Exiting program...")
					break
		except Exception as e:
			print(f"An error occurred: {e}")
		finally:
			# Releasing camera and closing windows
			self.cap.release()
			self.out.release()
			cv.destroyAllWindows()
			self.socket.close()
			if self.PRINT_DATA:
				print("Program exited successfully.")

			# Writing data to CSV file
			if self.LOG_DATA and self.IS_RECORDING:
				if self.PRINT_DATA:
					print("Writing data to CSV...")
				timestamp_str = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
				csv_file_name = os.path.join(
					self.TRACKING_DATA_LOG_FOLDER, f"{self.subject_id}_eye_tracking_log_{timestamp_str}.csv"
				)
				with open(csv_file_name, "w", newline="") as file:
					writer = csv.writer(file)
					writer.writerow(self.column_names)  # Writing column names
					writer.writerows(self.csv_data)  # Writing data rows
				if self.PRINT_DATA:
					print(f"Data written to {csv_file_name}")


if __name__ == "__main__":
	tracker = HeadGazeTracker("/home/max/Projects/Python-Gaze-Face-Tracker/config.yml")
	tracker.run()
