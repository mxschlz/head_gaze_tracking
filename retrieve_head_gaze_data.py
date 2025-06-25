from HeadGazeTracker.HeadGazeTracker import HeadGazeTracker


# setup
subject = "SMS019_A"  # subject id
video_input = f"/home/max/Insync/schulz.max5@gmail.com/GoogleDrive/PhD/data/baby_vids/input/{subject}_Video.mkv"  # if webcam==None, use this variable as video input
webcam = None # can be 0 or None
video_output = f"/home/max/Insync/schulz.max5@gmail.com/GoogleDrive/PhD/data/baby_vids/output/{subject}_processed_video_output.mkv"
tracking_data_log_folder = f"/home/max/Insync/schulz.max5@gmail.com/GoogleDrive/PhD/data/baby_vids/logs/"
tracker = HeadGazeTracker(subject_id=subject, config_file_path="config.yml", WEBCAM=webcam,
                          VIDEO_INPUT=video_input,
                          VIDEO_OUTPUT=video_output,
                          TRACKING_DATA_LOG_FOLDER=tracking_data_log_folder,
                          starting_timestamp=None,
                          total_frames=None)

# run
tracker.run()
