# leglance_classifier.py

import os
import cv2
import matplotlib
import numpy as np
import pandas as pd
import mediapipe as mp
from matplotlib import pyplot as plt
matplotlib.use('Agg')

def classify_leg_glance_type(highlight_clip_path, output_folder):
    """
    Further classify a leg glance/flick into front foot or back foot by analyzing:
    1. Ankle movement patterns to determine footwork
    2. Body orientation to confirm it's a leg glance/flick

    Args:
        highlight_clip_path (str): Path to the highlight video clip
        output_folder (str): Directory where analysis results will be saved

    Returns:
        str: Final classification result
    """
    # Create a subfolder for leg glance analysis
    glance_folder = os.path.join(output_folder, "leg_glance_analysis")
    os.makedirs(glance_folder, exist_ok=True)

    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # Get video properties
    cap = cv2.VideoCapture(highlight_clip_path)
    if not cap.isOpened():
        print("Error", f"Could not open video file: {highlight_clip_path}")
        return "Classification failed: Could not open video"

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    mid_frame = total_frames // 2

    # Landmark indices
    left_shoulder = mp_pose.PoseLandmark.LEFT_SHOULDER.value
    right_shoulder = mp_pose.PoseLandmark.RIGHT_SHOULDER.value
    left_ankle = mp_pose.PoseLandmark.LEFT_ANKLE.value
    right_ankle = mp_pose.PoseLandmark.RIGHT_ANKLE.value
    left_hip = mp_pose.PoseLandmark.LEFT_HIP.value
    right_hip = mp_pose.PoseLandmark.RIGHT_HIP.value

    # Data collection
    ankle_distances = []
    shoulder_angles = []
    frames_data = []

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        time = frame_count / fps

        # Process frame with MediaPipe
        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        frame_data = {
            'frame': frame_count,
            'time': time,
            'is_glance_candidate': False,
            'ankle_distance': 0,
            'shoulder_angle': 0
        }

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Calculate ankle distance
            left_ankle_pos = np.array([landmarks[left_ankle].x, landmarks[left_ankle].y])
            right_ankle_pos = np.array([landmarks[right_ankle].x, landmarks[right_ankle].y])
            ankle_distance = np.linalg.norm(left_ankle_pos - right_ankle_pos)
            frame_data['ankle_distance'] = ankle_distance
            ankle_distances.append(ankle_distance)

            # Calculate shoulder angle to determine body orientation
            left_shoulder_pos = np.array([landmarks[left_shoulder].x, landmarks[left_shoulder].y])
            right_shoulder_pos = np.array([landmarks[right_shoulder].x, landmarks[right_shoulder].y])
            left_hip_pos = np.array([landmarks[left_hip].x, landmarks[left_hip].y])
            right_hip_pos = np.array([landmarks[right_hip].x, landmarks[right_hip].y])

            # Vector from right shoulder to left shoulder
            shoulder_vec = left_shoulder_pos - right_shoulder_pos
            # Vector from right hip to right shoulder
            hip_shoulder_vec = right_shoulder_pos - right_hip_pos

            # Calculate angle between vectors
            angle = np.degrees(np.arccos(
                np.dot(shoulder_vec, hip_shoulder_vec) /
                (np.linalg.norm(shoulder_vec) * np.linalg.norm(hip_shoulder_vec))
            ))

            frame_data['shoulder_angle'] = angle
            shoulder_angles.append(angle)

            # Check leg glance condition: body is somewhat side-on (angle between 30-60 degrees)
            if 30 < angle < 60:
                frame_data['is_glance_candidate'] = True

            frames_data.append(frame_data)

            # Release resources
            cap.release()
            pose.close()

            # Convert to DataFrame
            df = pd.DataFrame(frames_data)

            # Step 1: Confirm it's a leg glance/flick
            glance_frames = df[df['is_glance_candidate']].shape[0]
            is_glance = glance_frames > 5  # At least 5 frames meet glance condition

            if not is_glance:
                return "Leg Glance/Flick (body orientation not typical for this shot)"

    # Step 2: Determine footwork for leg glance/flick
    before_mid = df[df['frame'] <= mid_frame]['ankle_distance'].sum()
    after_mid = df[df['frame'] > mid_frame]['ankle_distance'].sum()

    # Calculate difference
    diff = after_mid - before_mid

    # Determine classification
    if diff > 0.1:  # Significant threshold to avoid false positives
        result = "Front Foot Leg Glance/Flick"
    elif diff < -0.1:
        result = "Back Foot Leg Glance/Flick"
    else:
        result = "Leg Glance/Flick (Footwork unclear - ankle movement patterns inconclusive)"

    # Save analysis data
    analysis_data = {
        'total_frames': total_frames,
        'glance_frames': glance_frames,
        'before_mid_sum': before_mid,
        'after_mid_sum': after_mid,
        'difference': diff,
        'classification': result
    }

    # Create report
    report = (
        f"Leg Glance/Flick Classification Analysis:\n"
        f"======================================\n"
        f"Total frames: {total_frames}\n"
        f"Frames meeting glance criteria: {glance_frames}\n"
        f"Ankle distance sum (first half): {before_mid:.4f}\n"
        f"Ankle distance sum (second half): {after_mid:.4f}\n"
        f"Difference (after - before): {diff:.4f}\n"
        f"\nFinal Classification: {result}\n"
    )

    # Save report
    report_path = os.path.join(glance_folder, "leg_glance_classification_report.txt")
    with open(report_path, 'w') as f:
        f.write(report)

    # Visualize ankle distances
    plt.figure(figsize=(10, 6))
    plt.plot(df['time'], df['ankle_distance'], label='Ankle Distance')
    plt.axvline(x=df.iloc[mid_frame]['time'], color='r', linestyle='--', label='Mid Point')
    plt.title('Ankle Distance Over Time for Leg Glance/Flick')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Normalized Distance')
    plt.legend()
    plt.grid(True)

    plot_path = os.path.join(glance_folder, "ankle_distance_plot.png")
    plt.savefig(plot_path)
    plt.close()

    print(
        "Leg Glance Classification Complete",
        f"Analysis complete!\n\n{report}\n\nResults saved to:\n{glance_folder}"
    )

    return result


def process_leg_glance_classification(highlight_clip_path, output_folder):
    """
    Main function to process leg glance/flick classification.

    Args:
        highlight_clip_path (str): Path to the highlight clip
        output_folder (str): Main output folder for all results

    Returns:
        str: Final classification message
    """
    try:
        return classify_leg_glance_type(highlight_clip_path, output_folder)
    except Exception as e:
        error_msg = f"Error during leg glance classification: {str(e)}"
        print("Classification Error", error_msg)
        return error_msg