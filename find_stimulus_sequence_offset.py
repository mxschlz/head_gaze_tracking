from HeadGazeTracker.HeadGazeTracker import HeadGazeTracker
from HeadGazeTracker import get_data_path
import re
import os
import pathlib

# setup
subject = "SMS019"  # subject id
session = "B"
CONFIG_FILE = "config.yml"
video_input = pathlib.Path(f"{get_data_path()}input/{subject}_EEG/video/trimmed/{subject}_{session}.mkv")  # if webcam==None, use this variable as video input
webcam = None # can be 0 or None
video_output = pathlib.Path(f"{get_data_path()}output/{subject}_{session}_processed_video_output.mkv")
tracking_data_log_folder = pathlib.Path(f"{get_data_path()}logs/")

# --- EEG Log File Setup (for BrainVision .vmrk files) ---
eeg_base_path = pathlib.Path(f"{get_data_path()}input/{subject}_EEG/{session}")
EEG_HEADER_FILE = pathlib.Path(os.path.join(eeg_base_path, f"{subject}_{session}_EEG.vhdr"))
EEG_MARKER_FILE = pathlib.Path(os.path.join(eeg_base_path, f"{subject}_{session}_EEG.vmrk"))

# The description of the stimulus marker to search for in the .vmrk file.
# For "Mk2=Stimulus,S 10,..." this would be "Stimulus".
EEG_STIMULUS_DESCRIPTION = "Stimulus"

USE_SYNTHETIC_ONSETS = True


def get_eeg_stimulus_times(header_file, marker_file, stimulus_description):
    """
    Parses BrainVision header and marker files to find the absolute timestamp
    of the first stimulus event and all raw stimulus sample points.

    Args:
        header_file (str): Path to the .vhdr file.
        marker_file (str): Path to the .vmrk file.
        stimulus_description (str): The description of the stimulus marker (e.g., "Stimulus").

    Returns:
        tuple: (sampling_rate_hz, all_stim_samples)
               - The EEG sampling rate in Hz.
               - A list of raw, unmodified integer sample points for ALL stimuli.
    """
    # --- 1. Read Sampling Rate from Header File (.vhdr) ---
    sampling_interval_us = None
    with open(header_file, 'r') as f:
        for line in f:
            # Use a more specific regex to match ONLY the SamplingInterval line
            # This prevents accidentally matching other numeric values like 'DataPoints'.
            match = re.search(r'^SamplingInterval=(\d+)', line.strip())
            if match:
                sampling_interval_us = float(match.group(1))
                break
    if sampling_interval_us is None:
        raise ValueError("Could not find 'SamplingInterval' in the header file.")
    sampling_rate_hz = 1_000_000.0 / sampling_interval_us
    print(f"Found EEG Sampling Rate: {sampling_rate_hz:.2f} Hz")

    # --- 2. Read Raw Marker Samples from .vmrk File ---
    all_stim_samples = []
    with open(marker_file, 'r') as f:
        for line in f:
            # Find any marker matching the stimulus description
            if f"={stimulus_description}," in line:
                parts = line.strip().split(',')
                stim_sample = int(parts[2]) # e.g., 6023
                all_stim_samples.append(stim_sample)

    if not all_stim_samples:
        raise ValueError(f"Could not find any '{stimulus_description}' markers in the EEG log file.")

    print(f"Found {len(all_stim_samples)} stimulus markers. First is at sample: {all_stim_samples[0]}")
    return sampling_rate_hz, all_stim_samples


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
        # Get the raw, unmodified sample numbers from the .vmrk file
        eeg_sampling_rate, raw_eeg_samples = get_eeg_stimulus_times(EEG_HEADER_FILE, EEG_MARKER_FILE, EEG_STIMULUS_DESCRIPTION)
    except Exception as e:
        print(f"An error occurred reading the EEG log: {e}")
        return

    # --- Part 3: Align Timelines and Convert to Milliseconds ---
    # 1. Convert the video's first stimulus time from ms to EEG samples
    video_stim_samples = (video_stim_ms / 1000.0) * eeg_sampling_rate

    # 2. Calculate the offset: the difference in samples between the video's start and the EEG's start
    #    offset = (EEG time of stim1) - (Video time of stim1)
    sample_offset = raw_eeg_samples[0] - video_stim_samples

    # 3. Adjust all EEG sample points to be relative to the video's start time
    adjusted_eeg_samples = [s - sample_offset for s in raw_eeg_samples]

    # 4. Convert the final, adjusted sample points to milliseconds for the tracker
    final_eeg_onsets_ms = [int((s / eeg_sampling_rate) * 1000) for s in adjusted_eeg_samples]

    if USE_SYNTHETIC_ONSETS:
        num_test_trials = 10
        interval_ms = 5000  # 5 seconds

        # The first trial is at the time we found it in the video.
        # Subsequent trials are at 5-second intervals from that point.
        final_eeg_onsets_ms = [video_stim_ms + (i * interval_ms) for i in range(num_test_trials)]

    # --- Part 4: Run the full analysis ---
    print("\n--- Starting Full Analysis Pass ---")
    try:
        # Note: Re-initializing the tracker ensures it starts from the beginning of the video file
        full_analysis_tracker = HeadGazeTracker(
            subject_id=subject,
            config_file_path=CONFIG_FILE,
            VIDEO_INPUT=video_input,
            VIDEO_OUTPUT=video_output,
            TRACKING_DATA_LOG_FOLDER=tracking_data_log_folder,
            starting_timestamp=None,  # We don't need absolute timestamps for this relative analysis
            eeg_trial_onsets_ms=final_eeg_onsets_ms, # Pass the final, aligned onsets in ms
	        WEBCAM=None
        )
        full_analysis_tracker.run()
    except Exception as e:
        print(f"An error occurred during the full analysis run: {e}")


if __name__ == "__main__":
    main()
