#!/usr/bin/env python
import argparse
import os
import sys

# Name of your target conda environment
CONDA_ENV_NAME = "hgt"


def perform_actual_video_processing(input_folder, output_folder, config_file):
	"""
	This function contains the core video processing logic.
	It's called only when the script is running inside the target conda environment.
	"""
	# Import HeadGazeTracker here to ensure it's imported by the
	# Python interpreter from the 'hgt' environment.
	from HeadGazeTracker.HeadGazeTracker import HeadGazeTracker

	print(f"--- Running video processing in Python: {sys.executable} (should be from '{CONDA_ENV_NAME}' env) ---")

	if not os.path.isdir(input_folder):
		print(f"Error: Input folder '{input_folder}' not found.")
		return

	if not os.path.isdir(output_folder):
		print(f"Output folder '{output_folder}' not found. Creating it.")
		os.makedirs(output_folder, exist_ok=True)

	if not os.path.isfile(config_file):
		print(f"Error: Config file '{config_file}' not found.")
		return

	video_extensions = ['.mkv', '.mp4', '.avi', '.mov', '.mpeg', '.mpg']
	videos_processed_count = 0

	print(f"Scanning for videos in: {input_folder}")
	for item_name in os.listdir(input_folder):
		base_name, ext = os.path.splitext(item_name)
		if ext.lower() not in video_extensions:
			continue

		video_input_path = os.path.join(input_folder, item_name)
		print(f"\nProcessing video: {video_input_path}")

		subject_id = base_name
		processed_video_filename = f"{base_name}_processed{ext}"
		video_output_path = os.path.join(output_folder, processed_video_filename)
		# Logs will be placed in the output_folder, named by HeadGazeTracker
		tracking_data_log_folder = output_folder

		print(f"  Subject ID: {subject_id}")
		print(f"  Output video will be: {video_output_path}")
		print(f"  Log folder will be: {tracking_data_log_folder}")

		try:
			tracker = HeadGazeTracker(
				subject_id=subject_id,
				config_file_path=config_file,
				WEBCAM=None,
				VIDEO_INPUT=video_input_path,
				VIDEO_OUTPUT=video_output_path,
				TRACKING_DATA_LOG_FOLDER=tracking_data_log_folder,
			)
			tracker.run()
			videos_processed_count += 1
			print(f"Finished processing: {item_name}")
		except IOError as e:
			print(f"  IOError processing {item_name}: {e}. Skipping.")
		except Exception as e:
			print(f"  An unexpected error occurred while processing {item_name}: {e}. Skipping.")
			import traceback
			traceback.print_exc()

	if videos_processed_count > 0:
		print(f"\nSuccessfully processed {videos_processed_count} video(s).")
	else:
		print("\nNo video files were processed. Check input folder and file extensions.")


if __name__ == "__main__":
	#TODO: use argparse for algorithm input instead of hard coding
	input_folder = "/home/max/Insync/schulz.max5@gmail.com/GoogleDrive/PhD/data/baby_vids/video_input/"
	output_folder = "/home/max/Insync/schulz.max5@gmail.com/GoogleDrive/PhD/data/baby_vids/video_output/"
	config_file = "/home/max/Projects/head_gaze_tracking/config.yml"
	perform_actual_video_processing(input_folder, output_folder, config_file)
