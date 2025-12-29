import matplotlib
matplotlib.use("TkAgg")
import cv2
import mediapipe as mp
import yaml
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ==========================================
# CONFIGURATION & SWITCHES
# ==========================================
# [SWITCH] Set to True to enable full body (Holistic), False for Face Mesh only.
ENABLE_FULL_BODY = True 

def main():
    # 1. Setup Connections
    body_connections = list(mp.solutions.pose.POSE_CONNECTIONS)

    # 2. Setup MediaPipe
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    
    if ENABLE_FULL_BODY:
        print("Initializing MediaPipe Holistic (Face + Body)...")
        mp_model = mp.solutions.holistic.Holistic(
            refine_face_landmarks=True, # Essential for Iris landmarks (468-477)
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    else:
        print("Initializing MediaPipe FaceMesh (Face Only)...")
        mp_model = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=2,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    # 3. Setup Matplotlib for 3D Plotting
    plt.ion()
    fig = plt.figure(figsize=(12, 6))

    # Subplot 2: Body 3D (if enabled)
    body_lines = []
    if ENABLE_FULL_BODY:
        ax_body = fig.add_subplot(121, projection='3d')
        ax_body.set_title("Full Body Pose (3D)")
        ax_body.view_init(elev=10, azim=-90)
        ax_body.set_xlim(-1, 1)
        ax_body.set_zlim(-1, 1)
        body_scatter = ax_body.scatter([], [], [], c='b', marker='^')
        # Initialize lines for body
        body_lines = [ax_body.plot([], [], [], 'black', linewidth=2)[0] for _ in body_connections]

        # Subplot 3: Hand Position Temporal Plot (2D)
        ax_hand = fig.add_subplot(122)
        ax_hand.set_title("Hand Height (Relative to Hips)")
        ax_hand.set_xlabel("Time (Frames)")
        ax_hand.set_ylabel("Height (m)")
        ax_hand.set_ylim(-1, 1)
        ax_hand.grid(True)
        
        history_len = 100
        lh_data = [0] * history_len
        rh_data = [0] * history_len
        line_lh, = ax_hand.plot(range(history_len), lh_data, label='Left Hand', color='orange')
        line_rh, = ax_hand.plot(range(history_len), rh_data, label='Right Hand', color='purple')
        ax_hand.legend(loc='upper right')
    
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # To improve performance, mark the image as not writeable to pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process
        results = mp_model.process(image)

        # Draw annotations on the image
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Data containers for plotting
        detected_body_points = {}

        # --- Logic Extraction ---
        if ENABLE_FULL_BODY:
            # Holistic returns single face_landmarks (usually)
            if results.face_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    results.face_landmarks,
                    mp.solutions.face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp.solutions.pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
                
                # Use pose_world_landmarks for absolute metric coordinates (meters)
                for i, lm in enumerate(results.pose_world_landmarks.landmark):
                    if lm.visibility > 0.5:
                        detected_body_points[i] = (lm.x, lm.z, -lm.y)

        else:
            # FaceMesh returns multi_face_landmarks
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        image,
                        face_landmarks,
                        mp.solutions.face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())

        # --- Update 3D Plots ---
        
        # Update Body Plot
        if ENABLE_FULL_BODY:
            if detected_body_points:
                xs = [p[0] for p in detected_body_points.values()]
                ys = [p[1] for p in detected_body_points.values()]
                zs = [p[2] for p in detected_body_points.values()]
                ax_body.set_ylim(-1, 1)
                body_scatter._offsets3d = (xs, ys, zs)
            else:
                body_scatter._offsets3d = ([], [], [])
            
            # Update Body Lines
            for line, (start, end) in zip(body_lines, body_connections):
                if start in detected_body_points and end in detected_body_points:
                    start_pt = detected_body_points[start]
                    end_pt = detected_body_points[end]
                    line.set_data([start_pt[0], end_pt[0]], [start_pt[1], end_pt[1]])
                    line.set_3d_properties([start_pt[2], end_pt[2]])
                else:
                    line.set_data([], [])
                    line.set_3d_properties([])
            
            # Update Hand Plot
            # Indices: 15 = Left Wrist, 16 = Right Wrist
            # detected_body_points values are (x, z, height)
            # If hand is not detected, default to 0 (hip level)
            current_lh = detected_body_points[15][2] if 15 in detected_body_points else 0
            current_rh = detected_body_points[16][2] if 16 in detected_body_points else 0
            
            lh_data.append(current_lh)
            rh_data.append(current_rh)
            lh_data.pop(0)
            rh_data.pop(0)
            
            line_lh.set_ydata(lh_data)
            line_rh.set_ydata(rh_data)

        cv2.imshow('MediaPipe Demo', image)
        
        # Handle Plot Events
        plt.pause(0.001)

        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    mp_model.close()

if __name__ == "__main__":
    main()