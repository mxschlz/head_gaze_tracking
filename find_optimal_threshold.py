#!/usr/bin/env python
import os
import sys
import time
import glob
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import copy

# Ensure the custom modules can be found
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
	sys.path.insert(0, project_root)

from ocapi.Ocapi import Ocapi
from inter_rater_reliability import calculate_cohens_kappa


def find_summary_file_for_run(log_folder, subject_id, part_suffix, run_start_time):
	"""Finds the trial summary CSV created for a specific run."""
	time.sleep(2)  # Allow a few seconds of grace time for file creation

	# This pattern correctly looks for the first part's log file, which has no suffix.
	search_pattern = os.path.join(
		log_folder,
		f"{subject_id}_trial_summary{part_suffix}*.csv"
	)
	files = glob.glob(search_pattern)
	if not files:
		print(f"Warning: No summary files found with pattern: {search_pattern}")
		return None

	candidate_files = [f for f in files if os.path.getctime(f) > run_start_time]
	if not candidate_files:
		print(f"Warning: No summary files found that were created after the run started.")
		return None

	latest_file = max(candidate_files, key=os.path.getctime)
	print(f"Found generated summary file: {os.path.basename(latest_file)}")
	return latest_file


def run_optimization(subject, video, truth, truth_col, output_folder, config, thresholds, part1_duration_ms, labels):
	"""
    Main function to loop through thresholds, run the tracker, and calculate reliability.
    This function now explicitly controls the video processing duration.
    """
	# --- Internal Configuration ---
	GENERATED_DATA_COLUMN_INDEX = 6
	# The suffix is an empty string because we are only processing one part.
	PART_SUFFIX_TO_ANALYZE = ""
	# --- End Internal Configuration ---

	try:
		with open(config, 'r') as f:
			base_config = yaml.safe_load(f)
	except Exception as e:
		print(f"Error: Could not load base config file '{config}'. {e}")
		return

	temp_config_path = "temp_config_for_optimization.yml"
	all_results = []

	for threshold in thresholds:
		print(f"\n{'=' * 20} TESTING THRESHOLD: {threshold}% {'=' * 20}")

		# 1. Modify and save the temporary config file
		# Use deepcopy to safely handle nested dictionaries
		current_config = copy.deepcopy(base_config)

		# --- CORRECTED LOGIC: Modify the nested keys, not the top-level ---
		# This ensures the correct parameters are updated for the tracker.
		current_config['Trials']['LOOK_TO_STIMULUS_THRESHOLD_PERCENT'] = threshold
		current_config['Files']['SPLIT_VIDEO_AT_MS'] = part1_duration_ms

		# Disable visual features for maximum speed, as intended by the original comment
		current_config['System']['SHOW_ON_SCREEN_DATA'] = False
		# --- END OF CORRECTION ---

		with open(temp_config_path, 'w') as f:
			yaml.dump(current_config, f)

		# 2. Run the ocapi
		try:
			print(f"Processing video: {video} (up to {part1_duration_ms} ms)")
			run_start_time = time.time()
			tracker = Ocapi(
				subject_id=subject,
				config_file_path=temp_config_path,
				WEBCAM=None,
				VIDEO_INPUT=video,
				VIDEO_OUTPUT=None,  # Explicitly disable video output for speed
				TRACKING_DATA_LOG_FOLDER=output_folder
			)
			tracker.run()
		except Exception as e:
			print(f"An error occurred during Ocapi execution for threshold {threshold}%: {e}")
			import traceback
			traceback.print_exc()
			continue

		# 3. Find the generated summary file
		generated_file = find_summary_file_for_run(
			log_folder=output_folder,
			subject_id=subject,
			part_suffix=PART_SUFFIX_TO_ANALYZE,
			run_start_time=run_start_time
		)
		if not generated_file:
			print(f"Could not find the output summary file for threshold {threshold}%. Skipping.")
			continue

		# 4. Calculate Cohen's Kappa
		print(f"Comparing ground truth '{os.path.basename(truth)}' with generated file.")
		kappa, summary = calculate_cohens_kappa(
			file_path1=truth,
			file_path2=generated_file,
			column_index1=truth_col,
			column_index2=GENERATED_DATA_COLUMN_INDEX,
			labels=labels
		)
		if kappa is not None and summary is not None:
			print(f"Result for {threshold}% -> Cohen's Kappa: {kappa:.4f}")
			result_row = {"threshold": threshold, **summary}
			all_results.append(result_row)
		else:
			print(f"Could not calculate Kappa for threshold {threshold}%.")

	# 5. Clean up and show final results
	if os.path.exists(temp_config_path):
		os.remove(temp_config_path)

	if not all_results:
		print("\nNo results were generated. Please check your configuration and file paths.")
		return

	results_df = pd.DataFrame(all_results)
	results_df = results_df.sort_values(by="cohen_kappa", ascending=False)

	summary_filename = os.path.join(output_folder, f"optimization_summary_{subject}.csv")
	results_df.to_csv(summary_filename, index=False, float_format='%.4f')

	print(f"\n{'=' * 20} OPTIMIZATION COMPLETE {'=' * 20}")
	print(f"Results summary saved to: {summary_filename}")
	print("\nTop 5 Thresholds by Cohen's Kappa score:")
	print(results_df.head(5).to_string(index=False))

	best_threshold = results_df.iloc[0]['threshold']
	best_kappa = results_df.iloc[0]['cohen_kappa']
	print(f"\nOptimal Threshold: {best_threshold}% (Kappa: {best_kappa:.4f})")


def analyze_and_plot_results(subject, output_folder):
	"""
	Finds the optimization summary file, reads it, and plots the results.
	"""
	print(f"\n{'=' * 20} ANALYZING AND PLOTTING RESULTS FOR {subject} {'=' * 20}")
	summary_filename = os.path.join(output_folder, f"optimization_summary_{subject}.csv")
	if not os.path.exists(summary_filename):
		print(f"Error: Summary file not found at '{summary_filename}'")
		return

	print(f"Reading data from: {summary_filename}")
	results_df = pd.read_csv(summary_filename)

	# The original plot function is now a helper
	_plot_optimization_results(results_df, subject, output_folder)


def _plot_optimization_results(results_df, subject, output_folder):
	"""Helper function to plot optimization results from a DataFrame."""
	if results_df.empty:
		print("The results data is empty. No data to plot.")
		return

	# Ensure data is sorted by threshold for a clean line plot
	results_df = results_df.sort_values(by="threshold").reset_index(drop=True)

	if 'cohen_kappa' not in results_df.columns:
		print("Error: 'cohen_kappa' column not found in the DataFrame.")
		return

	best_point = results_df.loc[results_df['cohen_kappa'].idxmax()]
	best_threshold = best_point['threshold']
	best_kappa = best_point['cohen_kappa']

	plt.style.use('seaborn-v0_8-whitegrid')
	fig, ax = plt.subplots(figsize=(12, 7))
	ax.plot(results_df['threshold'], results_df['cohen_kappa'], marker='o', linestyle='-', label="Cohen's Kappa Score")
	ax.plot(best_threshold, best_kappa, 'o', markersize=12, markerfacecolor='gold', markeredgecolor='black',
	        label=f'Best Kappa: {best_kappa:.4f} at {best_threshold}%')
	ax.set_title(f"Threshold Optimization for Subject: {subject}", fontsize=16, fontweight='bold')
	ax.set_xlabel("Look to Stimulus Threshold (%)", fontsize=12)
	ax.set_ylabel("Cohen's Kappa (Inter-Rater Reliability)", fontsize=12)
	ax.legend()
	ax.grid(True, which='both', linestyle='--', linewidth=0.5)

	# Use a unique name for the plot from re-analysis
	plot_filename = os.path.join(output_folder, f"reanalyzed_optimization_plot_{subject}.png")
	plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
	print(f"\nSuccessfully saved plot to: {plot_filename}")


def reanalyze_and_plot_from_logs(subject_id, log_folder, truth_file_path, truth_file_column_index, thresholds_tested, labels):
	"""
	Re-analyzes existing raw summary files from a log folder by recalculating Kappa.

	This function is useful if the optimization summary was lost or needs to be
	regenerated. It assumes that the raw gaze summary CSV files generated by
	each threshold run exist in the log folder and can be identified by their
	creation time.

	Args:
		subject_id (str): The subject identifier.
		log_folder (str): The path to the folder containing the raw summary CSVs.
		truth_file_path (str): The path to the ground truth data file.
		truth_file_column_index (int): The column index for the truth data.
		thresholds_tested (list): A list of the thresholds that were run,
								  in the exact order they were executed.
	"""
	print(f"\n{'=' * 20} RE-ANALYZING RAW LOGS FOR {subject_id} {'=' * 20}")

	# This must match the column index used in the original run_optimization
	GENERATED_DATA_COLUMN_INDEX = 6

	# 1. Find all raw summary files for the subject
	search_pattern = os.path.join(log_folder, f"{subject_id}_trial_summary*.csv")
	all_files = glob.glob(search_pattern)

	# Exclude any aggregate summary files from the list of raw files
	summary_file_to_exclude = os.path.join(log_folder, f"optimization_summary_{subject_id}.csv")
	reanalyzed_summary_to_exclude = os.path.join(log_folder, f"reanalyzed_optimization_summary_{subject_id}.csv")

	candidate_files = [
		f for f in all_files if os.path.normpath(f) not in
		                        [os.path.normpath(summary_file_to_exclude),
		                         os.path.normpath(reanalyzed_summary_to_exclude)]
	]

	if not candidate_files:
		print(f"Error: No raw summary files found with pattern: {search_pattern}")
		return

	# 2. Sort files by creation time to match the order of thresholds tested
	candidate_files.sort(key=os.path.getctime)

	print(f"Found {len(candidate_files)} raw summary files. Expecting {len(thresholds_tested)} based on input.")
	if len(candidate_files) != len(thresholds_tested):
		print("\nWARNING: The number of found files does not match the number of thresholds.")
		print("The mapping of files to thresholds may be incorrect.")
		print("Please ensure the log folder contains only the files from a single, complete optimization run.")

	# 3. Pair files with thresholds and calculate Kappa for each
	all_results = []
	# Use zip; it will stop when the shorter of the two lists is exhausted.
	for generated_file, threshold in zip(candidate_files, thresholds_tested):
		print(f"\nProcessing file: {os.path.basename(generated_file)} for Assumed Threshold: {threshold}%")

		kappa, summary = calculate_cohens_kappa(
			file_path1=truth_file_path,
			file_path2=generated_file,
			column_index1=truth_file_column_index,
			column_index2=GENERATED_DATA_COLUMN_INDEX,
			labels=labels
		)
		if kappa is not None and summary is not None:
			print(f"  -> Result: Cohen's Kappa = {kappa:.4f}")
			result_row = {"threshold": threshold, **summary}
			all_results.append(result_row)
		else:
			print(f"  -> Could not calculate Kappa for threshold {threshold}%.")

	# 4. Save and plot the re-analyzed results
	if not all_results:
		print("\nNo results could be calculated from the found files.")
		return

	results_df = pd.DataFrame(all_results)
	results_df = results_df.sort_values(by="cohen_kappa", ascending=False)

	summary_filename = os.path.join(log_folder, f"reanalyzed_optimization_summary_{subject_id}.csv")
	results_df.to_csv(summary_filename, index=False, float_format='%.4f')

	print(f"\n{'=' * 20} RE-ANALYSIS COMPLETE {'=' * 20}")
	print(f"Re-analyzed summary saved to: {summary_filename}")
	print("\nTop 5 Thresholds by Cohen's Kappa score:")
	print(results_df.head(5).to_string(index=False))

	best_threshold = results_df.iloc[0]['threshold']
	best_kappa = results_df.iloc[0]['cohen_kappa']
	print(f"\nOptimal Threshold from Re-analysis: {best_threshold}% (Kappa: {best_kappa:.4f})")

	# Use the plotting helper function
	_plot_optimization_results(results_df, subject_id, log_folder)


def main():
	"""
    Main entry point of the script.
    Configure your paths and settings here before running.
    """
	# ==================================================================
	# ===               PLEASE CONFIGURE THESE VALUES                ===
	# ==================================================================

	# --- Thresholds to Test ---
	# This MUST match the thresholds used to generate the files in the log folder
	THRESHOLDS_TO_TEST = list(range(1, 90, 5))  # e.g., 25, 30, 35, ..., 70

	# --- Subject and File Configuration ---
	SUBJECT_ID = "SMS019_A"
	BASE_DATA_DIR = "/home/max/Insync/schulz.max5@gmail.com/GoogleDrive/PhD/data/baby_vids"
	INPUT_DIR = os.path.join(BASE_DATA_DIR, "input")
	OUTPUT_FOLDER = os.path.join(BASE_DATA_DIR, "logs")

	VIDEO_PATH = os.path.join(INPUT_DIR, f"{SUBJECT_ID}_Video.mkv")
	TRUTH_FILE_PATH = os.path.join(INPUT_DIR, f"{SUBJECT_ID}_VideoCoding.xlsx")
	TRUTH_FILE_COLUMN_INDEX = 0
	BASE_CONFIG_FILE = "config.yml"
	labels = [1, 2]  # coding scheme in the files

	# --- KEY SETTING: Define the duration of "part1" in milliseconds ---
	PART1_DURATION_MS = 655000

	# ==================================================================
	# ===                  END OF CONFIGURATION                    ===
	# ==================================================================
	"""
    # --- OPTION 1: Run the full optimization from video (Original) ---
	print("--- RUNNING FULL OPTIMIZATION ---")
	run_optimization(
		subject=SUBJECT_ID,
		video=VIDEO_PATH,
		truth=TRUTH_FILE_PATH,
		truth_col=TRUTH_FILE_COLUMN_INDEX,
		output_folder=OUTPUT_FOLDER,
		config=BASE_CONFIG_FILE,
		thresholds=THRESHOLDS_TO_TEST,
		part1_duration_ms=PART1_DURATION_MS,
		labels=labels)
	
	analyze_and_plot_results(
		subject=SUBJECT_ID,
		output_folder=OUTPUT_FOLDER)
	"""

	# --- OPTION 2: Re-analyze existing log files (Your New Function) ---
	print("\n--- RE-ANALYZING FROM EXISTING LOGS ---")
	reanalyze_and_plot_from_logs(
		subject_id=SUBJECT_ID,
		log_folder=OUTPUT_FOLDER,
		truth_file_path=TRUTH_FILE_PATH,
		truth_file_column_index=TRUTH_FILE_COLUMN_INDEX,
		thresholds_tested=THRESHOLDS_TO_TEST,
		labels=labels
	)


if __name__ == "__main__":
	main()
