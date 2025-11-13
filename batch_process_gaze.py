#!/usr/bin/env python
import argparse
import os
import pathlib
import sys


def process_batch(input_folder, output_folder, config_file):
	"""
	This function contains the core video processing logic.
	It's called only when the script is running inside the target conda environment.
	"""
	# Import HeadGazeTracker here to ensure it's imported by the
	# Python interpreter from the 'hgt' environment.
	from HeadGazeTracker import get_data_path
	from HeadGazeTracker.HeadGazeTracker import HeadGazeTracker

	print(f"--- Running video processing in Python: {sys.executable} ---")

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
	eeg_stimulus_description = "Stimulus"  # As seen in retrieve_head_gaze_data.py
	eeg_events_of_interest = ["S 21", "S 22", "S 23", "S 24"] # As seen in retrieve_head_gaze_data.py
	videos_processed_count = 0

	print(f"Scanning for videos in: {input_folder}")
	for item_name in sorted(os.listdir(input_folder)):
		base_name, ext = os.path.splitext(item_name)
		if ext.lower() not in video_extensions:
			continue

		video_input_path = os.path.join(input_folder, item_name)
		print(f"\nProcessing video: {video_input_path}")
		
		# --- Enhanced Logic to handle Subject/Session and find EEG files ---
		try:
			subject_id, session = base_name.split('_')
		except ValueError:
			print(f"  Warning: Could not parse subject and session from '{item_name}'. Skipping.")
			continue

		# Construct paths based on the structure in retrieve_head_gaze_data.py
		# This assumes the batch script is run from a location where get_data_path() is meaningful
		# or that the paths are absolute.
		eeg_base_path = pathlib.Path(f"{get_data_path()}input/{subject_id}_EEG/{session}")
		eeg_header_file = eeg_base_path / f"{subject_id}_{session}_EEG.vhdr"
		eeg_marker_file = eeg_base_path / f"{subject_id}_{session}_EEG.vmrk"

		if not eeg_header_file.is_file() or not eeg_marker_file.is_file():
			print(f"  Error: Missing EEG files for {subject_id} session {session}.")
			print(f"    - Looked for: {eeg_header_file}")
			print(f"    - Looked for: {eeg_marker_file}")
			print("  Skipping this video.")
			continue

		processed_video_filename = f"{base_name}_processed{ext}"
		video_output_path = os.path.join(output_folder, processed_video_filename)
		tracking_data_log_folder = os.path.join(output_folder, "logs")
		os.makedirs(tracking_data_log_folder, exist_ok=True)

		print(f"  Subject ID: {subject_id}")
		print(f"  Session: {session}")
		print(f"  Output video will be: {video_output_path}")
		print(f"  Log folder will be: {tracking_data_log_folder}")

		try:
			tracker = HeadGazeTracker(
				subject_id=subject_id,
				session=session,
				config_file_path=config_file,
				WEBCAM=None, # Assuming batch processing is always from files
				VIDEO_INPUT=video_input_path,
				VIDEO_OUTPUT=video_output_path,
				TRACKING_DATA_LOG_FOLDER=tracking_data_log_folder,
			)

			# Perform EEG synchronization for each video
			print("  Synchronizing with EEG data...")
			tracker.sync_with_eeg_and_set_onsets(
				header_file=eeg_header_file,
				marker_file=eeg_marker_file,
				stimulus_description=eeg_stimulus_description,
				events_of_interest=eeg_events_of_interest
			)

			print("  Starting analysis...")
			tracker.run()
			videos_processed_count += 1
			print(f"Finished processing: {item_name}")
		except IOError as e:
			print(f"  IOError processing {item_name}: {e}. Skipping.")
		except Exception as e:
			print(f"  An unexpected error occurred while processing {item_name}: {e}. Skipping video.")
			import traceback
			traceback.print_exc()

	if videos_processed_count > 0:
		print(f"\nSuccessfully processed {videos_processed_count} video(s).")
	else:
		print("\nNo video files were processed. Check input folder, file extensions, and EEG file availability.")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Batch process videos with HeadGazeTracker and EEG synchronization.")
	parser.add_argument('-i', '--input', required=True, help="Folder containing the input videos (e.g., '.../trimmed/').")
	parser.add_argument('-o', '--output', required=True, help="Folder to save processed videos and logs.")
	parser.add_argument('-c', '--config', default="config.yml", help="Path to the configuration file (e.g., 'config.yml').")

	args = parser.parse_args()

	process_batch(args.input, args.output, args.config)
