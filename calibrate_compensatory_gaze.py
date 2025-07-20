# calibrate_compensatory_gaze.py

import cv2 as cv
from HeadGazeTracker.HeadGazeTracker import HeadGazeTracker


def run_calibration_tool():
	"""
    A utility to help find the optimal eye and head angle values for the
    'compensatory_gaze' method.
    """
	print("--- Compensatory Gaze Calibration Tool ---")
	print("Instructions:")
	print("1. Look directly at the camera/screen.")
	print("2. Press 'c' to calibrate the 'forward' head pose.")
	print("3. Follow the on-screen prompts to turn your head while keeping your eyes on the screen.")
	print("4. Observe the printed 'Head Yaw' and 'Eye DX Sum' values.")
	print("5. Use these observed value ranges to update your config.yml.")
	print("6. Press 'q' to quit.")
	print("-" * 40)

	try:
		# Initialize the tracker for webcam input.
		# We disable trial detection as it's not needed for this tool.
		tracker = HeadGazeTracker(
			config_file_path="config.yml",
			WEBCAM=0
		)
		tracker.ENABLE_VIDEO_TRIAL_DETECTION = False
		tracker.SHOW_ON_SCREEN_DATA = False  # Disable default drawing for a cleaner view

	except Exception as e:
		print(f"Error initializing HeadGazeTracker: {e}")
		print("Please ensure 'config.yml' is present and your webcam is working.")
		return

	# --- Video Saving Setup ---
	video_writer = None
	recording_enabled = False
	output_filename = "calibration_output.mp4"
	try:
		# Get video properties from the tracker's capture object
		frame_width = int(tracker.cap.get(cv.CAP_PROP_FRAME_WIDTH))
		frame_height = int(tracker.cap.get(cv.CAP_PROP_FRAME_HEIGHT))
		fps = tracker.cap.get(cv.CAP_PROP_FPS)
		if fps == 0:  # Handle cases where FPS is not reported by webcam
			fps = 30  # Assume 30 FPS
			print(f"Warning: Webcam did not report FPS. Defaulting to {fps} FPS for output video.")

		# Get the FourCC code from the config file for consistency
		fourcc_str = tracker.OUTPUT_VIDEO_FOURCC
		fourcc = cv.VideoWriter_fourcc(*fourcc_str)

		# Initialize VideoWriter
		video_writer = cv.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))
		print(f"✅ Video recording enabled. Output will be saved to '{output_filename}'")
		recording_enabled = True
	except Exception as e:
		print(f"❌ Could not initialize video writer: {e}")
		# The tool will still run, but without saving video.

	# These ranges from your config are used for visual feedback in this script
	yaw_center = tracker.COMPENSATORY_HEAD_YAW_RANGE_CENTER
	yaw_left = tracker.COMPENSATORY_HEAD_YAW_RANGE_LEFT
	yaw_right = tracker.COMPENSATORY_HEAD_YAW_RANGE_RIGHT

	while True:
		# --- Simplified main loop from HeadGazeTracker.run() ---
		tracker._reset_per_frame_state()

		frame, img_h, img_w, ret = tracker._get_and_preprocess_frame()
		if not ret:
			print("Failed to grab frame from webcam.")
			break

		key_pressed = cv.waitKey(1) & 0xFF
		if key_pressed == ord('q'):
			print("\nExiting tool.")
			break

		# Process the frame to get all the necessary data
		results, _, mesh_points, _ = tracker._process_face_mesh(frame)
		tracker.mesh_points = mesh_points

		if results and results.multi_face_landmarks:
			tracker._extract_eye_features(mesh_points)
			tracker._process_head_pose(mesh_points, img_h, img_w, key_pressed)

			# --- Custom Calibration Logic and Display ---
			if tracker.calibrated:
				head_yaw = tracker.adj_yaw
				eye_dx_sum = tracker.l_dx + tracker.r_dx

				# Determine current head state for on-screen prompt
				head_state = "UNKNOWN"
				if yaw_left[0] <= head_yaw <= yaw_left[1]:
					head_state = "TURNED LEFT"
				elif yaw_right[0] <= head_yaw <= yaw_right[1]:
					head_state = "TURNED RIGHT"
				elif yaw_center[0] <= head_yaw <= yaw_center[1]:
					head_state = "CENTERED"

				# Print continuous data to the console
				print(
					f"\r[CALIBRATED] Head Yaw: {head_yaw:6.1f} | Eye DX Sum: {eye_dx_sum:6.1f} | Head State: {head_state:12s}",
					end=""
				)

				# Draw information on the frame
				cv.putText(frame, f"Head Yaw: {head_yaw:.1f}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
				cv.putText(frame, f"Eye DX Sum (L+R): {eye_dx_sum:.1f}", (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.7,
				           (0, 255, 0), 2)
				cv.putText(frame, f"STATE: {head_state}", (10, 90), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

				# Draw eye centers (green) and outer corners (blue) for visual feedback
				if tracker.l_cx != 0: cv.circle(frame, (tracker.l_cx, tracker.l_cy), 3, (0, 255, 0), -1)
				if tracker.r_cx != 0: cv.circle(frame, (tracker.r_cx, tracker.r_cy), 3, (0, 255, 0), -1)
				if mesh_points is not None:
					cv.circle(frame, tuple(mesh_points[tracker.LEFT_EYE_OUTER_CORNER]), 3, (255, 0, 0), -1)
					cv.circle(frame, tuple(mesh_points[tracker.RIGHT_EYE_OUTER_CORNER]), 3, (255, 0, 0), -1)

			else:
				# Prompt to calibrate
				print("\r[NOT CALIBRATED] Look forward and press 'c' to begin...", end="")
				(w, h), _ = cv.getTextSize("LOOK FORWARD & PRESS 'C'", cv.FONT_HERSHEY_SIMPLEX, 1, 2)
				cv.putText(frame, "LOOK FORWARD & PRESS 'C'", ((img_w - w) // 2, (img_h - h) // 2),
				           cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
		else:
			# No face detected
			print("\r[NO FACE DETECTED] Make sure your face is visible.", end="")
			(w, h), _ = cv.getTextSize("NO FACE DETECTED", cv.FONT_HERSHEY_SIMPLEX, 1, 2)
			cv.putText(frame, "NO FACE DETECTED", ((img_w - w) // 2, (img_h - h) // 2), cv.FONT_HERSHEY_SIMPLEX, 1,
			           (0, 0, 255), 2)

		cv.imshow("Compensatory Gaze Calibration Tool", frame)

		# Write the frame to the output video file if recording is enabled
		if recording_enabled and video_writer is not None:
			video_writer.write(frame)

	# Cleanup
	if recording_enabled and video_writer is not None:
		video_writer.release()
		print(f"\nVideo saved to {output_filename}")

	tracker._cleanup(finalize_data=False)


if __name__ == "__main__":
	run_calibration_tool()
