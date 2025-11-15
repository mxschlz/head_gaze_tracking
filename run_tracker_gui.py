#!/usr/bin/env python
from gooey import Gooey, GooeyParser
import os
from HeadGazeTracker.HeadGazeTracker import HeadGazeTracker
import logging


@Gooey(
    program_name="Head & Gaze Tracker",
    default_size=(800, 900),
    navigation='TABBED',
    tabbed_groups=True,
    menu=[{
        'name': 'File',
        'items': [{
            'type': 'AboutDialog',
            'menuTitle': 'About',
            'name': 'Head & Gaze Tracker',
            'description': 'An application for processing head and eye gaze from video.',
            'version': '1.1.0',
            'copyright': '2024',
        }]
    }]
)
def main():
    """
    Main function to parse arguments and run the HeadGazeTracker.
    The @Gooey decorator will automatically turn this into a GUI.
    """
    parser = GooeyParser(description="Configure and run the Head & Gaze Tracker")

    # --- I/O Group ---
    io_group = parser.add_argument_group('Input & Output', gooey_options={'columns': 1})
    io_group.add_argument('-i', '--video_input', help="Path to the input video file", widget="FileChooser")
    io_group.add_argument('--webcam', type=int, help="Webcam ID to use (e.g., 0 for the default webcam)")
    io_group.add_argument('-o', '--video_output', help="Path to save the processed output video", widget="FileSaver")
    io_group.add_argument('--subject_id', help="Subject ID (e.g., SMS019)", default="TEST")
    io_group.add_argument('--session', help="Session identifier (e.g., A, B)", default="A")
    io_group.add_argument('--log_folder', help="Folder to save log files", widget="DirChooser", default="./output/logs")

    # --- Calibration Group ---
    calib_group = parser.add_argument_group('Calibration', gooey_options={'columns': 2})
    calib_group.add_argument(
        '--calibration_method',
        choices=['clustering', 'gaze_informed', 'manual', 'none'],
        default='clustering',
        help='Method for head pose calibration'
    )
    calib_group.add_argument(
        '--clustering_duration',
        type=float, default=5.0,
        help='Duration (seconds) for clustering calibration'
    )

    # --- Gaze Classification Group ---
    gaze_group = parser.add_argument_group('Gaze Classification', gooey_options={'columns': 2})
    gaze_group.add_argument(
        '--gaze_classification_method',
        choices=['head_pose_only', 'eye_gaze_with_head_filter', 'compensatory_gaze', 'eye_gaze_only'],
        default='head_pose_only',
        help='Algorithm to determine if gaze is on stimulus'
    )
    gaze_group.add_argument(
        '--stimulus_pitch_range',
        nargs=2, type=int, default=[-25, 25],
        help='Min/Max pitch for "head_pose_only" (e.g., -25 25)'
    )
    gaze_group.add_argument(
        '--stimulus_yaw_range',
        nargs=2, type=int, default=[-25, 25],
        help='Min/Max yaw for "head_pose_only" (e.g., -25 25)'
    )

    # --- System Group ---
    sys_group = parser.add_argument_group('System & Display')
    sys_group.add_argument(
        '--show_on_screen_data',
        action='store_true', default=True,
        help='Show video output window with data overlays'
    )
    sys_group.add_argument(
        '--tuning_mode',
        action='store_true', default=False,
        help='Enable tuning mode to find head pose boundaries'
    )
    sys_group.add_argument(
        '--rotate',
        type=int, choices=[0, 90, 180, 270], default=180,
        help='Rotate video input'
    )

    args = parser.parse_args()

    # --- Assemble the config dictionary from parsed arguments ---
    # This structure mirrors your config.yml file, allowing us to pass it directly.
    config_dict = {
        'System': {
            'PRINT_DATA': True,  # Always print data when running from GUI for feedback
            'SHOW_ON_SCREEN_DATA': args.show_on_screen_data,
            'TUNING_MODE': args.tuning_mode,
        },
        'Files': {
            'ROTATE': args.rotate,
            'LOG_DATA': True,
        },
        'Features': {
            'ENABLE_HEAD_POSE': True,
        },
        'Calibration': {
            'METHOD': args.calibration_method,
            'CLUSTERING_CALIB_DURATION_SECONDS': args.clustering_duration,
        },
        'Trials': {
            'ENABLE': True, # Assume trials are enabled; can be made a GUI option
            'GAZE_CLASSIFICATION_METHOD': args.gaze_classification_method,
            'STIMULUS_PITCH_RANGE': args.stimulus_pitch_range,
            'STIMULUS_YAW_RANGE': args.stimulus_yaw_range,
            # Add other trial params here if you want them in the GUI
        }
    }

    # --- Validate Inputs ---
    if not args.video_input and args.webcam is None:
        print("Error: You must provide either a Video Input file or a Webcam ID.")
        return

    if args.video_input and not os.path.exists(args.video_input):
        print(f"Error: Video file not found at '{args.video_input}'")
        return

    # --- Run the Tracker ---
    try:
        print("="*50)
        print("Initializing HeadGazeTracker with settings from GUI...")
        print("="*50)

        tracker = HeadGazeTracker(
            subject_id=args.subject_id,
            session=args.session,
            VIDEO_INPUT=args.video_input,
            WEBCAM=args.webcam,
            VIDEO_OUTPUT=args.video_output,
            TRACKING_DATA_LOG_FOLDER=args.log_folder,
            config_dict=config_dict,
            config_file_path=None # Explicitly disable loading from file
        )
        tracker.run()
        print("\nProcessing finished successfully!")

    except Exception as e:
        # Gooey will display this in a popup on error
        print(f"An error occurred: {e}")
        logging.basicConfig()
        logging.critical(f"Failed to initialize or run HeadGazeTracker: {e}", exc_info=True)
        # In a real app, you might want more robust error handling/logging here.


if __name__ == "__main__":
    main()