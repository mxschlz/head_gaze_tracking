from HeadGazeTracker.HeadGazeTracker import HeadGazeTracker
from HeadGazeTracker import get_data_path


subject = "sub-110"  # subject id
video_input = f"{get_data_path()}sourcedata/raw/{subject}/headgaze/{subject}_block-0.asf"  # if webcam==None, use this variable as video input
webcam = None # can be 0 or None
video_output = f"{get_data_path()}derivatives/preprocessing/{subject}/{subject}_hgt_output.avi"
tracking_data_log_folder = f"{get_data_path()}derivatives/preprocessing/{subject}/"
tracker = HeadGazeTracker(subject_id=subject, config_file_path="config.yml", WEBCAM=webcam,
                          VIDEO_INPUT=video_input,
                          VIDEO_OUTPUT=video_output,
                          TRACKING_DATA_LOG_FOLDER=tracking_data_log_folder,
                          starting_timestamp=None,
                          total_frames=None)

# run
tracker.run()
