from HeadGazeTracker.HeadGazeTracker import HeadGazeTracker
from HeadGazeTracker import get_data_path
from datetime import datetime, timedelta
import re
import os

# setup
subject = "SMS019"  # subject id
session = "A"
CONFIG_FILE = "config.yml"
video_input = f"{get_data_path()}input\\{subject}_EEG\\video\\{subject}_{session}.mkv"  # if webcam==None, use this variable as video input
webcam = None # can be 0 or None
video_output = f"{get_data_path()}output\\{subject}_processed_video_output.mkv"
tracking_data_log_folder = f"{get_data_path()}logs\\"

# --- EEG Log File Setup (for BrainVision .vmrk files) ---
eeg_base_path = f"{get_data_path()}input\\{subject}_EEG\\{session}"
EEG_HEADER_FILE = os.path.join(eeg_base_path, f"{subject}_{session}_EEG.vhdr")
EEG_MARKER_FILE = os.path.join(eeg_base_path, f"{subject}_{session}_EEG.vmrk")

# The description of the stimulus marker to search for in the .vmrk file.
# For "Mk2=Stimulus,S 10,..." this would be "Stimulus".
EEG_STIMULUS_DESCRIPTION = "Stimulus"


def get_eeg_stimulus_time(header_file, marker_file, stimulus_description):
    """
    Parses BrainVision header and marker files to find the absolute timestamp
    of the first stimulus event.

    Args:
        header_file (str): Path to the .vhdr file.
        marker_file (str): Path to the .vmrk file.
        stimulus_description (str): The description of the stimulus marker (e.g., "Stimulus").

    Returns:
        datetime: A datetime object representing the absolute time of the first stimulus.
        Returns None if the stimulus or necessary info isn't found.
    """
    # --- 1. Read Sampling Rate from Header File (.vhdr) ---
    sampling_interval_us = None
    with open(header_file, 'r') as f:
        for line in f:
            # The interval is often given in microseconds per sample
            if "SamplingInterval" in line:
                sampling_interval_us = float(re.search(r'=(\d+)', line).group(1))
                break
    if sampling_interval_us is None:
        raise ValueError("Could not find 'SamplingInterval' in the header file.")
    sampling_rate_hz = 1_000_000.0 / sampling_interval_us
    print(f"Found EEG Sampling Rate: {sampling_rate_hz:.2f} Hz")

    # --- 2. Read Marker File (.vmrk) ---
    eeg_start_time_dt = None
    first_stim_sample = None
    with open(marker_file, 'r') as f:
        for line in f:
            # Find the "New Segment" marker for the absolute start time
            if "New Segment" in line and eeg_start_time_dt is None:
                # Timestamp is the last value, e.g., ...20240722093211705104
                timestamp_str = line.strip().split(',')[-1]
                # BrainVision timestamps are YYYYMMDDHHMMSS + 6 digits (microseconds)
                eeg_start_time_dt = datetime.strptime(timestamp_str, "%Y%m%d%H%M%S%f")
                print(f"Found EEG recording start time: {eeg_start_time_dt}")

            # Find the first marker matching the stimulus description
            if f"={stimulus_description}," in line and first_stim_sample is None:
                parts = line.strip().split(',')
                first_stim_sample = int(parts[2]) # e.g., 6023
                print(f"Found first stimulus '{stimulus_description}' at sample: {first_stim_sample}")

    if not eeg_start_time_dt or first_stim_sample is None:
        raise ValueError(f"Could not find start time or a '{stimulus_description}' marker in the EEG logs.")

    # --- 3. Calculate Absolute Stimulus Time ---
    stimulus_offset_ms = (first_stim_sample / sampling_rate_hz) * 1000
    absolute_stim_time = eeg_start_time_dt + timedelta(milliseconds=stimulus_offset_ms)
    return absolute_stim_time


def main():
    # --- Part 1: Find the stimulus in the video ---
    try:
        # Initialize the tracker just for finding the stimulus
        tracker = HeadGazeTracker(subject_id=subject, config_file_path=CONFIG_FILE, WEBCAM=webcam,
                                  VIDEO_INPUT=video_input,
                                  VIDEO_OUTPUT=video_output,
                                  TRACKING_DATA_LOG_FOLDER=tracking_data_log_folder,
                                  starting_timestamp=None,
                                  total_frames=None)

        # Use the new method to get the video frame and time of the first stimulus
        video_stim_frame, video_stim_ms = tracker.find_first_stimulus_onset()

        if video_stim_frame is None:
            print("Could not find the first stimulus in the video. Check your ROI settings in config.yaml.")
            return

    except Exception as e:
        print(f"An error occurred during video analysis: {e}")
        return

    # --- Part 2: Find the stimulus in the EEG log ---
    try:
        eeg_stim_time_dt = get_eeg_stimulus_time(EEG_HEADER_FILE, EEG_MARKER_FILE, EEG_STIMULUS_DESCRIPTION)
    except Exception as e:
        print(f"An error occurred reading the EEG log: {e}")
        return

    # --- Part 3: Calculate the video's true start time (The Goal!) ---
    # This is the crucial calculation
    video_start_time = eeg_stim_time_dt - timedelta(milliseconds=video_stim_ms)

    # Convert to the string format your HeadGazeTracker expects
    video_start_timestamp_str = video_start_time.strftime(tracker.TIMESTAMP_FORMAT)

    print("\n--- Synchronization Complete ---")
    print(f"First EEG stimulus time: {eeg_stim_time_dt.strftime('%Y-%m-%d %H:%M:%S.%f')}")
    print(f"Video's calculated start time (aligned to EEG): {video_start_timestamp_str}")
    print("You can now run the full analysis using this timestamp.")

    # --- Part 4: (Optional) Run the full analysis immediately ---
    print("\n--- Starting Full Analysis Pass ---")
    try:
        # Note: Re-initializing the tracker ensures it starts from the beginning of the video file
        full_analysis_tracker = HeadGazeTracker(
            subject_id=subject,
            config_file_path=CONFIG_FILE,
            VIDEO_INPUT=video_input,
            VIDEO_OUTPUT=video_output,
            TRACKING_DATA_LOG_FOLDER=tracking_data_log_folder,
            starting_timestamp=video_start_timestamp_str  # Pass the aligned timestamp here!
        )
        full_analysis_tracker.run()
    except Exception as e:
        print(f"An error occurred during the full analysis run: {e}")


if __name__ == "__main__":
    main()
