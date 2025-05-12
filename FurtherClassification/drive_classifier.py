# drive_classifier.py

import os
import cv2
import matplotlib
import numpy as np
import pandas as pd
import mediapipe as mp
from matplotlib import pyplot as plt
matplotlib.use('Agg')


def classify_drive_type(highlight_clip_path, output_folder):
    """
    Further classify a drive into front foot or back foot drive by analyzing:
    1. Elbow positions to confirm it's a drive (not defence)
    2. Ankle movement patterns to determine footwork

    Args:
        highlight_clip_path (str): Path to the highlight video clip
        output_folder (str): Directory where analysis results will be saved

    Returns:
        str: Final classification result
    """
    # Create a subfolder for drive analysis
    drive_folder = os.path.join(output_folder, "drive_analysis")
    os.makedirs(drive_folder, exist_ok=True)

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
    left_elbow = mp_pose.PoseLandmark.LEFT_ELBOW.value
    right_elbow = mp_pose.PoseLandmark.RIGHT_ELBOW.value
    left_ankle = mp_pose.PoseLandmark.LEFT_ANKLE.value
    right_ankle = mp_pose.PoseLandmark.RIGHT_ANKLE.value

    # Data collection
    elbow_positions = []
    ankle_distances = []
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
            'is_drive_candidate': False,
            'ankle_distance': 0
        }

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Get elbow positions
            left_elbow_x = landmarks[left_elbow].x
            right_elbow_x = landmarks[right_elbow].x
            elbow_positions.append((left_elbow_x, right_elbow_x))

            # Check drive condition: right elbow is to the right of left elbow
            if right_elbow_x > left_elbow_x:
                frame_data['is_drive_candidate'] = True

            # Calculate ankle distance
            left_ankle_pos = np.array([landmarks[left_ankle].x, landmarks[left_ankle].y])
            right_ankle_pos = np.array([landmarks[right_ankle].x, landmarks[right_ankle].y])
            ankle_distance = np.linalg.norm(left_ankle_pos - right_ankle_pos)
            frame_data['ankle_distance'] = ankle_distance
            ankle_distances.append(ankle_distance)

        frames_data.append(frame_data)

    # Release resources
    cap.release()
    pose.close()

    # Convert to DataFrame
    df = pd.DataFrame(frames_data)

    # Step 1: Confirm it's a drive (not defence)
    drive_frames = df[df['is_drive_candidate']].shape[0]
    is_drive = drive_frames > 5  # At least 5 frames meet drive condition

    if not is_drive:
        # Classify as defence and determine footwork
        return classify_defence_type(df, mid_frame, output_folder)

    # Step 2: Determine footwork for drive
    before_mid = df[df['frame'] <= mid_frame]['ankle_distance'].sum()
    after_mid = df[df['frame'] > mid_frame]['ankle_distance'].sum()

    # Calculate difference
    diff = after_mid - before_mid

    # Determine classification
    if diff > 0.1:  # Significant threshold to avoid false positives
        result = "Front Foot Drive"
    elif diff < -0.1:
        result = "Back Foot Drive"
    else:
        result = "Drive (Footwork unclear - ankle movement patterns inconclusive)"

    # Save analysis data
    analysis_data = {
        'total_frames': total_frames,
        'drive_frames': drive_frames,
        'before_mid_sum': before_mid,
        'after_mid_sum': after_mid,
        'difference': diff,
        'classification': result
    }

    # Create report
    report = (
        f"Drive Classification Analysis:\n"
        f"============================\n"
        f"Total frames: {total_frames}\n"
        f"Frames meeting drive criteria: {drive_frames}\n"
        f"Ankle distance sum (first half): {before_mid:.4f}\n"
        f"Ankle distance sum (second half): {after_mid:.4f}\n"
        f"Difference (after - before): {diff:.4f}\n"
        f"\nFinal Classification: {result}\n"
    )

    # Save report
    report_path = os.path.join(drive_folder, "drive_classification_report.txt")
    with open(report_path, 'w') as f:
        f.write(report)

    # Visualize ankle distances
    plt.figure(figsize=(10, 6))
    plt.plot(df['time'], df['ankle_distance'], label='Ankle Distance')
    plt.axvline(x=df.iloc[mid_frame]['time'], color='r', linestyle='--', label='Mid Point')
    plt.title('Ankle Distance Over Time')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Normalized Distance')
    plt.legend()
    plt.grid(True)

    plot_path = os.path.join(drive_folder, "ankle_distance_plot.png")
    plt.savefig(plot_path)
    plt.close()

    print(
        "Drive Classification Complete",
        f"Analysis complete!\n\n{report}\n\nResults saved to:\n{drive_folder}"
    )

    return result


def classify_defence_type(df, mid_frame, output_folder):
    """
    Classify defence into front foot or back foot defence based on ankle movement.

    Args:
        df (DataFrame): Frame data with ankle distances
        mid_frame (int): Middle frame number of the clip
        output_folder (str): Output directory path

    Returns:
        str: Defence classification result
    """
    # Calculate ankle distance sums
    before_mid = df[df['frame'] <= mid_frame]['ankle_distance'].sum()
    after_mid = df[df['frame'] > mid_frame]['ankle_distance'].sum()
    diff = after_mid - before_mid

    # Determine classification
    if diff > 0.1:  # Significant threshold
        result = "Front Foot Defence"
    elif diff < -0.1:
        result = "Back Foot Defence"
    else:
        result = "Defence (Footwork unclear - ankle movement patterns inconclusive)"

    # Save analysis data
    defence_folder = os.path.join(output_folder, "defence_analysis")
    os.makedirs(defence_folder, exist_ok=True)

    analysis_data = {
        'before_mid_sum': before_mid,
        'after_mid_sum': after_mid,
        'difference': diff,
        'classification': result
    }

    # Create report
    report = (
        f"Defence Classification Analysis:\n"
        f"===============================\n"
        f"Ankle distance sum (first half): {before_mid:.4f}\n"
        f"Ankle distance sum (second half): {after_mid:.4f}\n"
        f"Difference (after - before): {diff:.4f}\n"
        f"\nFinal Classification: {result}\n"
    )

    # Save report
    report_path = os.path.join(defence_folder, "defence_classification_report.txt")
    with open(report_path, 'w') as f:
        f.write(report)

    print(
        "Defence Classification",
        f"Shot was classified as defence.\n\n{report}"
    )

    return result


def process_drive_classification(highlight_clip_path, output_folder):
    """
    Main function to process drive classification.

    Args:
        highlight_clip_path (str): Path to the highlight clip
        output_folder (str): Main output folder for all results

    Returns:
        str: Final classification message
    """
    try:
        return classify_drive_type(highlight_clip_path, output_folder)
    except Exception as e:
        error_msg = f"Error during drive classification: {str(e)}"
        print("Classification Error", error_msg)
        return error_msg