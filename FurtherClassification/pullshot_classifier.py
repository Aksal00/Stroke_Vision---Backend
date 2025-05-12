# pullshot_classifier.py

import os
import cv2
import matplotlib
import numpy as np
import pandas as pd
import mediapipe as mp
from matplotlib import pyplot as plt
matplotlib.use('Agg')

def classify_pullshot_type(highlight_clip_path, output_folder):
    """
    Differentiate between pull shot and hook shot by analyzing:
    1. The moment when right wrist has maximum x coordinate
    2. Compare right wrist y coordinate with right shoulder y coordinate at that moment

    Args:
        highlight_clip_path (str): Path to the highlight video clip
        output_folder (str): Directory where analysis results will be saved

    Returns:
        str: Final classification result ("Pull Shot" or "Hook Shot")
    """
    # Create a subfolder for pull shot analysis
    pullshot_folder = os.path.join(output_folder, "pullshot_analysis")
    os.makedirs(pullshot_folder, exist_ok=True)

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

    # Landmark indices
    right_wrist = mp_pose.PoseLandmark.RIGHT_WRIST.value
    right_shoulder = mp_pose.PoseLandmark.RIGHT_SHOULDER.value

    # Data collection
    wrist_positions = []
    shoulder_positions = []
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
            'wrist_x': 0,
            'wrist_y': 0,
            'shoulder_y': 0,
            'is_shot_moment': False
        }

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Get wrist and shoulder positions
            wrist_x = landmarks[right_wrist].x
            wrist_y = landmarks[right_wrist].y
            shoulder_y = landmarks[right_shoulder].y

            frame_data['wrist_x'] = wrist_x
            frame_data['wrist_y'] = wrist_y
            frame_data['shoulder_y'] = shoulder_y

            wrist_positions.append((wrist_x, wrist_y))
            shoulder_positions.append(shoulder_y)

        frames_data.append(frame_data)

    # Release resources
    cap.release()
    pose.close()

    # Convert to DataFrame
    df = pd.DataFrame(frames_data)

    # Find the moment when wrist has maximum x coordinate (most extended position)
    max_x_idx = df['wrist_x'].idxmax()
    max_x_frame = df.loc[max_x_idx]

    # Get the y coordinates at this moment
    wrist_y_at_max = max_x_frame['wrist_y']
    shoulder_y_at_max = max_x_frame['shoulder_y']

    # Determine shot type
    if wrist_y_at_max < shoulder_y_at_max:
        result = "Pull Shot"
        reason = "Wrist is below shoulder at maximum extension"
    else:
        result = "Hook Shot"
        reason = "Wrist is at or above shoulder at maximum extension"

    # Save analysis data
    analysis_data = {
        'total_frames': total_frames,
        'max_x_frame': int(max_x_frame['frame']),
        'max_x_time': max_x_frame['time'],
        'wrist_y_at_max': wrist_y_at_max,
        'shoulder_y_at_max': shoulder_y_at_max,
        'classification': result,
        'reason': reason
    }

    # Create report
    report = (
        f"Pull Shot vs Hook Shot Analysis:\n"
        f"==============================\n"
        f"Total frames analyzed: {total_frames}\n"
        f"Key moment frame: {int(max_x_frame['frame'])} (at {max_x_frame['time']:.2f} seconds)\n"
        f"Right wrist y coordinate at max extension: {wrist_y_at_max:.4f}\n"
        f"Right shoulder y coordinate at same moment: {shoulder_y_at_max:.4f}\n"
        f"\nKey Observation: {reason}\n"
        f"\nFinal Classification: {result}\n"
    )

    # Save report
    report_path = os.path.join(pullshot_folder, "pullshot_classification_report.txt")
    with open(report_path, 'w') as f:
        f.write(report)

    # Visualize wrist and shoulder positions
    plt.figure(figsize=(10, 6))
    plt.plot(df['time'], df['wrist_x'], label='Right Wrist X Position')
    plt.plot(df['time'], df['wrist_y'], label='Right Wrist Y Position')
    plt.plot(df['time'], df['shoulder_y'], label='Right Shoulder Y Position')
    plt.axvline(x=max_x_frame['time'], color='r', linestyle='--', label='Key Moment')
    plt.title('Wrist and Shoulder Positions Over Time')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Normalized Position')
    plt.legend()
    plt.grid(True)

    plot_path = os.path.join(pullshot_folder, "position_plot.png")
    plt.savefig(plot_path)
    plt.close()

    print(
        "Pull Shot Classification Complete",
        f"Analysis complete!\n\n{report}\n\nResults saved to:\n{pullshot_folder}"
    )

    return result


def process_pullshot_classification(highlight_clip_path, output_folder):
    """
    Main function to process pull shot classification.

    Args:
        highlight_clip_path (str): Path to the highlight clip
        output_folder (str): Main output folder for all results

    Returns:
        str: Final classification message
    """
    try:
        return classify_pullshot_type(highlight_clip_path, output_folder)
    except Exception as e:
        error_msg = f"Error during pull shot classification: {str(e)}"
        print("Classification Error", error_msg)
        return error_msg