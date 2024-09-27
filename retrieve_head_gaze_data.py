from HeadGazeTracker.HeadGazeTracker import HeadGazeTracker


subject = "sub-99"  # subject id
video_input = f"C:\\data\\behavior\\eyetracking\\raw\\{subject}_block_1.asf"  # if webcam==0, use this variable as video input
webcam = None  # can be 0 or None
video_output = f"C:\\data\\behavior\\eyetracking\\hgt_video_output\\{subject}_hgt_output.avi"
tracking_data_log_folder = "C:\\data\\behavior\\eyetracking\\tracking_log_data"
tracker = HeadGazeTracker(subject_id=subject, config_file_path="config.yml", WEBCAM=webcam,
                          VIDEO_INPUT=video_input,
                          VIDEO_OUTPUT=video_output,
                          TRACKING_DATA_LOG_FOLDER=tracking_data_log_folder)

# run
tracker.run()