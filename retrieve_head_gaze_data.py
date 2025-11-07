from HeadGazeTracker.HeadGazeTracker import HeadGazeTracker
from HeadGazeTracker import get_data_path


# setup
subject = "SMS019"  # subject id
session = "A"
video_input = f"{get_data_path()}input\\{subject}_EEG\\video\\{subject}_{session}.mkv"  # if webcam==None, use this variable as video input
video_output = f"{get_data_path()}output\\{subject}_processed_video_output.mkv"
webcam = None # can be 0 or None
tracking_data_log_folder = f"{get_data_path()}/logs/"
tracker = HeadGazeTracker(subject_id=subject, config_file_path="config.yml", WEBCAM=webcam,
                          VIDEO_INPUT=video_input,
                          VIDEO_OUTPUT=video_output,
                          TRACKING_DATA_LOG_FOLDER=tracking_data_log_folder,
                          starting_timestamp=None,
                          total_frames=None)

# run
tracker.run()
