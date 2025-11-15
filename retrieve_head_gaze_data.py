from HeadGazeTracker.HeadGazeTracker import HeadGazeTracker
from HeadGazeTracker import get_data_path
import pathlib


# setup
subject = "SMS059"  # subject id
session = "B"
CONFIG_FILE = "config.yml"

# --- Video Input (using glob for flexibility) ---
video_dir = pathlib.Path(f"{get_data_path()}input/{subject}/video/trimmed/")
video_pattern = f"{subject}_{session}*.mkv"
found_videos = list(video_dir.glob(video_pattern))

if not found_videos:
    raise FileNotFoundError(f"No video file found for subject '{subject}', session '{session}' in '{video_dir}'")
if len(found_videos) > 1:
    raise FileNotFoundError(f"Multiple video files found for subject '{subject}', session '{session}': {found_videos}. Please specify one or clean the directory.")

video_input = found_videos[0]
print(f"Found video input: {video_input}")

video_output = pathlib.Path(f"{get_data_path()}output/{subject}_{session}_processed_video_output.mkv")
webcam = None # can be 0 or None
tracking_data_log_folder = pathlib.Path(f"{get_data_path()}/logs/")

# --- EEG Log File Setup (only used if USE_EEG_SYNC is True) ---
eeg_base_path = pathlib.Path(f"{get_data_path()}input/{subject}/{session}")
eeg_header_pattern = f"{subject}_{session}*.vhdr"
found_eeg_headers = list(eeg_base_path.glob(eeg_header_pattern))
if not found_eeg_headers:
    raise FileNotFoundError(f"No EEG header file found for subject '{subject}', session '{session}' in '{eeg_base_path}'")
if len(found_eeg_headers) > 1:
    raise FileNotFoundError(f"Multiple EEG header files found for subject '{subject}', session '{session}': {found_eeg_headers}. Please specify one or clean the directory.")
EEG_HEADER_FILE = found_eeg_headers[0]

eeg_marker_pattern = f"{subject}_{session}*.vmrk"
found_eeg_markers = list(eeg_base_path.glob(eeg_marker_pattern))
if not found_eeg_markers:
    raise FileNotFoundError(f"No EEG marker file found for subject '{subject}', session '{session}' in '{eeg_base_path}'")
if len(found_eeg_markers) > 1:
    raise FileNotFoundError(f"Multiple EEG marker files found for subject '{subject}', session '{session}': {found_eeg_markers}. Please specify one or clean the directory.")
EEG_MARKER_FILE = found_eeg_markers[0]
EEG_STIMULUS_DESCRIPTION = "Stimulus"
EEG_EVENTS_OF_INTEREST = ["S 21", "S 22", "S 23", "S 24"]

if __name__ == "__main__":
	tracker = HeadGazeTracker(subject_id=subject, config_file_path=CONFIG_FILE, WEBCAM=webcam,
                              VIDEO_INPUT=video_input, VIDEO_OUTPUT=video_output,
                              TRACKING_DATA_LOG_FOLDER=tracking_data_log_folder, session=session)

	tracker.sync_with_eeg_and_set_onsets(header_file=EEG_HEADER_FILE, marker_file=EEG_MARKER_FILE,
                                         stimulus_description=EEG_STIMULUS_DESCRIPTION,
                                         events_of_interest=EEG_EVENTS_OF_INTEREST)

	tracker.run()
