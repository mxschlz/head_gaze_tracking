from HeadGazeTracker.HeadGazeTracker import HeadGazeTracker
from HeadGazeTracker import get_data_path
import pathlib
import re
import os


# setup
subject = "SMS019"  # subject id
session = "C"
CONFIG_FILE = "config.yml"
video_input = pathlib.Path(f"{get_data_path()}input/{subject}_EEG/video/trimmed/{subject}_{session}.mkv")  # if webcam==None, use this variable as video input
video_output = pathlib.Path(f"{get_data_path()}output/{subject}_{session}_processed_video_output.mkv")
webcam = None # can be 0 or None
tracking_data_log_folder = pathlib.Path(f"{get_data_path()}/logs/")


# ======================================================================================
# --- ANALYSIS MODE SWITCH ---
# Set to True to use the EEG .vmrk file for trial onsets.
# Set to False to use the video-based brightness detection defined in config.yml.
USE_EEG_SYNC = True
# ======================================================================================

# --- EEG Log File Setup (only used if USE_EEG_SYNC is True) ---
eeg_base_path = pathlib.Path(f"{get_data_path()}input/{subject}_EEG/{session}")
EEG_HEADER_FILE = pathlib.Path(os.path.join(eeg_base_path, f"{subject}_{session}_EEG.vhdr"))
EEG_MARKER_FILE = pathlib.Path(os.path.join(eeg_base_path, f"{subject}_{session}_EEG.vmrk"))
EEG_STIMULUS_DESCRIPTION = "Stimulus"


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
            if f"={stimulus_description}," in line:
                parts = line.strip().split(',')
                stim_sample = int(parts[2])
                all_stim_samples.append(stim_sample)

    if not all_stim_samples:
        raise ValueError(f"Could not find any '{stimulus_description}' markers in the EEG log file.")

    print(f"Found {len(all_stim_samples)} stimulus markers. First is at sample: {all_stim_samples[0]}")
    return sampling_rate_hz, all_stim_samples


if __name__ == "__main__":
    if not USE_EEG_SYNC:
        print("--- Running in Video-Based Trial Detection Mode ---")
        tracker = HeadGazeTracker(subject_id=subject, config_file_path=CONFIG_FILE, WEBCAM=webcam,
                                  VIDEO_INPUT=video_input, VIDEO_OUTPUT=video_output,
                                  TRACKING_DATA_LOG_FOLDER=tracking_data_log_folder)
        tracker.run()
    else:
        print("--- Running in EEG-Synchronized Trial Detection Mode ---")
        try:
            tracker = HeadGazeTracker(subject_id=subject, config_file_path=CONFIG_FILE, WEBCAM=webcam,
                                      VIDEO_INPUT=video_input, VIDEO_OUTPUT=video_output,
                                      TRACKING_DATA_LOG_FOLDER=tracking_data_log_folder)

            video_stim_frame, video_stim_ms = tracker.find_first_stimulus_onset()
            if video_stim_frame is None:
                raise RuntimeError("Could not find the first stimulus in the video. Check ROI settings in config.")

            eeg_sampling_rate, raw_eeg_samples = get_eeg_stimulus_times(EEG_HEADER_FILE, EEG_MARKER_FILE, EEG_STIMULUS_DESCRIPTION)
            video_stim_samples = (video_stim_ms / 1000.0) * eeg_sampling_rate
            sample_offset = raw_eeg_samples[0] - video_stim_samples
            adjusted_eeg_samples = [s - sample_offset for s in raw_eeg_samples]
            final_eeg_onsets_ms = [int((s / eeg_sampling_rate) * 1000) for s in adjusted_eeg_samples]
            print(f"Calculated {len(final_eeg_onsets_ms)} real trial onsets (ms).")

            print("\n--- Starting Full Analysis Pass (reusing primed tracker) ---")
            tracker.eeg_trial_onsets_ms = final_eeg_onsets_ms
            tracker.run()

        except Exception as e:
            print(f"An error occurred during the EEG-synchronized analysis process: {e}")
