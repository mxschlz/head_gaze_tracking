import os
import matplotlib
matplotlib.use('TkAgg')
import sys
import cv2
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from ocapi.ocapi import Ocapi
import pathlib
from ocapi import get_data_path
plt.ion()

# --- 1. Configuration ---
# EDIT THE VARIABLES BELOW TO POINT TO YOUR VIDEO AND CONFIG
# --------------------------------------------------------------------
VIDEO_INPUT_PATH = f"{get_data_path()}input\\SCS048\\video\\trimmed\\SCS048_C_Video.mkv"
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
print("--- Initializing ocapi to collect data ---")
tracker = Ocapi(
    subject_id="visualization_test",
    config_file_path=CONFIG_FILE_PATH,
    VIDEO_INPUT=VIDEO_INPUT_PATH,
    VIDEO_OUTPUT=None,
    TRACKING_DATA_LOG_FOLDER=None,
    WEBCAM=None
)

# --- 4. Process Video for Calibration Duration Only ---
cap = cv2.VideoCapture(VIDEO_INPUT_PATH)
if not cap.isOpened():
    print(f"Error: Could not open video file {VIDEO_INPUT_PATH}")
    sys.exit(1)

frame_limit = tracker.CLUSTERING_CALIB_DURATION_FRAMES
print(f"\nProcessing video for {tracker.calib_duration_sec} seconds ({frame_limit} frames)...")

frame_num = 0
while cap.isOpened() and frame_num < frame_limit:
    success, frame = cap.read()
    if not success:
        break

    # Replicate the core processing steps from the tracker's main run loop
    # to collect the necessary data for calibration.
    if tracker.FLIP_VIDEO: frame = cv2.flip(frame, 1)
    if tracker.ROTATE == 90:
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif tracker.ROTATE == 180:
        frame = cv2.rotate(frame, cv2.ROTATE_180)
    elif tracker.ROTATE == -90 or tracker.ROTATE == 270:
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

    img_h, img_w = frame.shape[:2]
    results, _, mesh_points, _ = tracker._process_face_mesh(frame)

    if results and results.multi_face_landmarks:
        # This is the crucial call that collects calibration samples
        tracker._process_head_pose(mesh_points, img_h, img_w, key_pressed=-1)

    frame_num += 1
    if tracker.FPS > 0 and frame_num % int(tracker.FPS) == 0:
        print(f"  Processed {frame_num}/{frame_limit} frames...")

cap.release()
print("Data collection complete.")

# --- 5. Analyze and Visualize the Collected Data ---
pose_data = np.array(tracker.clustering_calib_all_samples)

if len(pose_data) == 0:
    print("\nError: No head pose samples were collected. Was a face visible in the first part of the video?")
    sys.exit(1)

# --- MODIFIED: Use the tracker's internal calibration method ---
print("\n--- Running Tracker's Internal Clustering Calibration ---")
labels = tracker._perform_clustering_calibration()
if labels is None:
    print("Calibration method returned no labels. Exiting.")
    sys.exit(1)

# Retrieve the calculated baseline from the tracker instance
calculated_baseline = np.array([tracker.initial_pitch, tracker.initial_yaw, tracker.initial_roll])
# --- END MODIFICATION ---

print("\n--- Result ---")
print(f"Calculated Baseline Pose: P={calculated_baseline[0]:.2f}, Y={calculated_baseline[1]:.2f}, R={calculated_baseline[2]:.2f}")

# Show the 3D plot
plots_dir = pathlib.Path(f"{get_data_path()}showcase")

# --- NEW: Apply a poster-friendly theme ---
sns.set_theme(style="ticks", context="poster", font_scale=1.1)

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

unique_labels = set(labels)
# Use a vibrant palette for the clusters
colors = sns.color_palette("viridis", len(unique_labels) - (1 if -1 in unique_labels else 0))

# --- MODIFIED: Plot each cluster and noise separately ---
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Plot noise points in grey
        noise_mask = (labels == k)
        # Make noise points very transparent
        ax.scatter(pose_data[noise_mask, 1], pose_data[noise_mask, 2], pose_data[noise_mask, 0], c='grey', alpha=0.05, s=20, label=f'Noise (n={np.sum(noise_mask)})')
    else:
        # Plot points for the current cluster
        cluster_mask = (labels == k)
        # Make cluster points more transparent to reveal the star
        ax.scatter(pose_data[cluster_mask, 1], pose_data[cluster_mask, 2], pose_data[cluster_mask, 0], color=col, alpha=0.15, s=50, label=f'Cluster {k} (n={np.sum(cluster_mask)})')

# --- MODIFIED: Highlight the baseline with a more prominent marker ---
# Make the star larger and bolder to ensure it's visible
ax.scatter(calculated_baseline[1], calculated_baseline[2], calculated_baseline[0], c='red', s=1000, marker='*', edgecolor='black', linewidth=3, label='Calculated Baseline', depthshade=False, zorder=100)

# --- MODIFIED: Improve titles and labels ---
ax.set_title('Head Pose Clustering', fontsize=24, pad=20)
ax.set_xlabel('\nYaw (°)')
ax.set_ylabel('\nRoll (°)')
ax.set_zlabel('\nPitch (°)')

# Improve legend
#ax.legend(title="Data Points", loc='center left', bbox_to_anchor=(1.0, 0.5))

# Set a better viewing angle
ax.view_init(elev=10, azim=-50)

fig.tight_layout()

# --- NEW: Save the figure ---
save_path = plots_dir / "clustering_visualization.svg"
plt.savefig(save_path)
print(f"\nClustering plot saved to: {save_path}")
