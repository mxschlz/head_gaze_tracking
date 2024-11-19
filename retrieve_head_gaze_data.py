from HeadGazeTracker.HeadGazeTracker import HeadGazeTracker


subject = "sub-99"  # subject id
video_input = f"/home/max/Downloads/max_hgt_output.avi"  # if webcam==None, use this variable as video input
webcam = None # can be 0 or None
video_output = "output.avi"
tracking_data_log_folder = "/home/max/PycharmProjects/head_gaze_tracking/test/"
tracker = HeadGazeTracker(subject_id=subject, config_file_path="config.yml", WEBCAM=webcam,
                          VIDEO_INPUT=video_input,
                          VIDEO_OUTPUT=video_output,
                          TRACKING_DATA_LOG_FOLDER=tracking_data_log_folder,
                          starting_timestamp=None,
                          total_frames=None)

# run
tracker.run()
