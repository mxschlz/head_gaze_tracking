import os
import sys
import cv2
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plotting

# We need to import the tracker class itself
from HeadGazeTracker.HeadGazeTracker import HeadGazeTracker

def find_baseline_with_dbscan(pose_data, eps, min_samples):
    """
    Applies the DBSCAN algorithm to find the most stable pose cluster.
    This is a helper function adapted from the simulation script.
    """
    print("\nRunning DBSCAN to find the densest cluster of poses...")
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(pose_data)
    labels = db.labels_

    # Find the largest cluster (ignoring noise, which is labeled -1)
    unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)

    if len(counts) == 0:
        print("DBSCAN could not find any stable clusters.")
        print("TRY INCREASING 'CLUSTERING_DBSCAN_EPS' in config.yml to be more lenient.")
        # Fallback: use the median of all data
        return np.median(pose_data, axis=0), labels

    largest_cluster_label = unique_labels[np.argmax(counts)]
    num_in_cluster = np.max(counts)
    num_noise = np.sum(labels == -1)

    print("DBSCAN analysis complete:")
    print(f"  - Found largest cluster (label {largest_cluster_label}) with {num_in_cluster} points.")
    print(f"  - Classified {num_noise} points as noise.\n")

    largest_cluster_points = pose_data[labels == largest_cluster_label]
    calculated_baseline = np.median(largest_cluster_points, axis=0)

    return calculated_baseline, labels

def visualize_clustering_results(pose_data, labels, calculated_baseline):
    """Creates a 3D scatter plot to visualize the clustering results."""
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    core_samples_mask = (labels != -1)
    noise_mask = ~core_samples_mask

    # Plot the main cluster points
    ax.scatter(pose_data[core_samples_mask, 1], pose_data[core_samples_mask, 0], pose_data[core_samples_mask, 2],
               c='blue', alpha=0.6, label=f'Main Cluster (n={np.sum(core_samples_mask)})')

    # Plot the noise points
    ax.scatter(pose_data[noise_mask, 1], pose_data[noise_mask, 0], pose_data[noise_mask, 2],
               c='grey', alpha=0.3, s=10, label=f'Noise (n={np.sum(noise_mask)})')

    # Plot the baseline calculated by our algorithm
    ax.scatter(calculated_baseline[1], calculated_baseline[0], calculated_baseline[2],
               c='red', s=200, marker='X', edgecolor='black', label='Calculated Baseline')

    ax.set_title('DBSCAN Clustering on Real Video Data', fontsize=16)
    ax.set_xlabel('Yaw (degrees)')
    ax.set_ylabel('Pitch (degrees)')
    ax.set_zlabel('Roll (degrees)')
    ax.legend()
    plt.show(block=True)

def main():
    """
    Main function to run the tracker for the calibration period and visualize the data.
    """
    # --- 1. Configuration ---
    # EDIT THE VARIABLES BELOW TO POINT TO YOUR VIDEO AND CONFIG
    # --------------------------------------------------------------------
    VIDEO_INPUT_PATH = "/home/max/Insync/schulz.max5@gmail.com/GoogleDrive/PhD/data/baby_vids/input/SMS019_A_Video.mkv"
    CONFIG_FILE_PATH = "config.yml"
    # --------------------------------------------------------------------

    # --- 2. Validation ---
    if not os.path.exists(VIDEO_INPUT_PATH):
        print(f"Error: Input video file not found at '{VIDEO_INPUT_PATH}'")
        sys.exit(1)
    if not os.path.exists(CONFIG_FILE_PATH):
        print(f"Error: Config file not found at '{CONFIG_FILE_PATH}'")
        sys.exit(1)

    # --- 3. Initialize Tracker ---
    # We initialize the tracker but won't use its full .run() method.
    # We pass dummy values for output files as we only need to collect data.
    print("--- Initializing HeadGazeTracker to collect data ---")
    tracker = HeadGazeTracker(
        subject_id="visualization_test",
        config_file_path=CONFIG_FILE_PATH,
        VIDEO_INPUT=VIDEO_INPUT_PATH,
        VIDEO_OUTPUT=None,
        TRACKING_DATA_LOG_FOLDER=None,
        WEBCAM=None
    )

    # Ensure the config is set to clustering mode for this script to work
    if tracker.CALIBRATION_METHOD != 'clustering':
        print(f"Error: CALIBRATION_METHOD in '{CONFIG_FILE_PATH}' must be 'clustering'.")
        sys.exit(1)

    # --- 4. Process Video for Calibration Duration Only ---
    cap = cv2.VideoCapture(VIDEO_INPUT_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open video file {VIDEO_INPUT_PATH}")
        sys.exit(1)

    frame_limit = tracker.CLUSTERING_CALIB_DURATION_FRAMES
    print(f"\nProcessing video for {tracker.CLUSTERING_CALIB_DURATION_SECONDS} seconds ({frame_limit} frames)...")

    frame_num = 0
    while cap.isOpened() and frame_num < frame_limit:
        success, frame = cap.read()
        if not success:
            break

        # --- CORRECTED PROCESSING LOGIC ---
        # Replicate the core processing steps from the tracker's main run loop
        # to collect the necessary data for calibration.

        # 1. Preprocess frame to match tracker's settings (flip/rotate)
        if tracker.FLIP_VIDEO: frame = cv2.flip(frame, 1)
        if tracker.ROTATE == 90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif tracker.ROTATE == 180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif tracker.ROTATE == -90 or tracker.ROTATE == 270:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        img_h, img_w = frame.shape[:2]

        # 2. Run face mesh detection
        results, _, mesh_points, _ = tracker._process_face_mesh(frame)

        # 3. If a face is found, process features to collect calibration data
        if results and results.multi_face_landmarks:
            # Note: We call _extract_eye_features because the 'gaze_informed'
            # calibration method depends on it. It's good practice to keep it
            # even when testing the 'clustering' method.
            tracker._extract_eye_features(mesh_points)

            # This is the crucial call that collects calibration samples
            tracker._process_head_pose(mesh_points, img_h, img_w, key_pressed=-1)
        # --- END OF CORRECTION ---

        frame_num += 1
        # Print progress
        if tracker.FPS > 0 and frame_num % int(tracker.FPS) == 0:
            print(f"  Processed {frame_num}/{frame_limit} frames...")

    cap.release()
    print("Data collection complete.")

    # --- 5. Analyze and Visualize the Collected Data ---
    pose_data = np.array(tracker.clustering_calib_all_samples)

    if len(pose_data) == 0:
        print("\nError: No head pose samples were collected. Was a face visible in the first part of the video?")
        sys.exit(1)

    # Get DBSCAN parameters from the tracker's config
    eps = getattr(tracker, "CLUSTERING_DBSCAN_EPS", 3.0)
    min_samples = getattr(tracker, "CLUSTERING_DBSCAN_MIN_SAMPLES", 15)

    # Run the analysis
    calculated_baseline, labels = find_baseline_with_dbscan(pose_data, eps, min_samples)

    print("--- Result ---")
    print(f"Calculated Baseline Pose: P={calculated_baseline[0]:.2f}, Y={calculated_baseline[1]:.2f}, R={calculated_baseline[2]:.2f}")

    # Show the 3D plot
    visualize_clustering_results(pose_data, labels, calculated_baseline)


if __name__ == "__main__":
    main()
