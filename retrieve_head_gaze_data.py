from HeadGazeTracker.HeadGazeTracker import HeadGazeTracker
from HeadGazeTracker import get_data_path
import pathlib
import os


# setup
subject = "SMS019"  # subject id
session = "B"
CONFIG_FILE = "config.yml"
video_input = pathlib.Path(f"{get_data_path()}input/{subject}_EEG/video/trimmed/{subject}_{session}.mkv")  # if webcam==None, use this variable as video input
video_output = pathlib.Path(f"{get_data_path()}output/{subject}_{session}_processed_video_output.mkv")
webcam = None # can be 0 or None
tracking_data_log_folder = pathlib.Path(f"{get_data_path()}/logs/")

# --- EEG Log File Setup (only used if USE_EEG_SYNC is True) ---
eeg_base_path = pathlib.Path(f"{get_data_path()}input/{subject}_EEG/{session}")
EEG_HEADER_FILE = pathlib.Path(os.path.join(eeg_base_path, f"{subject}_{session}_EEG.vhdr"))
EEG_MARKER_FILE = pathlib.Path(os.path.join(eeg_base_path, f"{subject}_{session}_EEG.vmrk"))
EEG_STIMULUS_DESCRIPTION = "Stimulus"
EEG_EVENTS_OF_INTEREST = ["S 21", "S 22", "S 23", "S 24"]

if __name__ == "__main__":
	tracker = HeadGazeTracker(subject_id=subject, config_file_path=CONFIG_FILE, WEBCAM=webcam,
                              VIDEO_INPUT=video_input, VIDEO_OUTPUT=video_output,
                              TRACKING_DATA_LOG_FOLDER=tracking_data_log_folder)

	tracker.sync_with_eeg_and_set_onsets(header_file=EEG_HEADER_FILE, marker_file=EEG_MARKER_FILE,
                                         stimulus_description=EEG_STIMULUS_DESCRIPTION,
                                         events_of_interest=EEG_EVENTS_OF_INTEREST)

	tracker.run()
