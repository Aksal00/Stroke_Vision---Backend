#pose_analyzer.py

import cv2
import mediapipe as mp
import numpy as np
import os
from tkinter import messagebox
from visualization import visualize_landmark_data
from highlight_generator import export_highlight_clips


def analyze_cricket_landmarks(video_path, output_path):
    """
    Main function to analyze cricket player's pose and movements from a video.

    Args:
        video_path (str): Path to the input video file.
        output_path (str): Path where the analyzed video will be saved.

    Returns:
        bool: True if analysis completed successfully, False otherwise.
    """
    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        smooth_segmentation=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    mp_drawing = mp.solutions.drawing_utils

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        messagebox.showerror("Error", "Could not open video file")
        return False

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create VideoWriter for output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Create a progress window
    cv2.namedWindow('Processing Video - Press Q to cancel', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Processing Video - Press Q to cancel', 800, 600)

    # Data collection for landmarks
    landmark_data = []
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    timestamps = []

    # Store original frames for highlight clips
    original_frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        current_time = frame_count / fps
        timestamps.append(current_time)

        # Store original frame for highlight clips
        original_frames.append(frame.copy())

        # Process pose estimation
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Initialize frame data
        frame_data = {'frame': frame_count, 'time': current_time}

        # Draw pose annotations and collect data
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
            )

            # Highlight important cricket joints and collect their positions
            landmarks = results.pose_landmarks.landmark
            for idx, landmark in enumerate(landmarks):
                x = int(landmark.x * width)
                y = int(landmark.y * height)

                # Store normalized coordinates (0-1 range)
                frame_data[f'landmark_{idx}_x'] = landmark.x
                frame_data[f'landmark_{idx}_y'] = landmark.y
                frame_data[f'landmark_{idx}_z'] = landmark.z
                frame_data[f'landmark_{idx}_visibility'] = landmark.visibility

                # Draw important landmarks
                if idx in [mp_pose.PoseLandmark.LEFT_SHOULDER.value,
                           mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
                           mp_pose.PoseLandmark.LEFT_ELBOW.value,
                           mp_pose.PoseLandmark.RIGHT_ELBOW.value,
                           mp_pose.PoseLandmark.LEFT_WRIST.value,
                           mp_pose.PoseLandmark.RIGHT_WRIST.value,
                           mp_pose.PoseLandmark.LEFT_HIP.value,
                           mp_pose.PoseLandmark.RIGHT_HIP.value,
                           mp_pose.PoseLandmark.LEFT_KNEE.value,
                           mp_pose.PoseLandmark.RIGHT_KNEE.value]:
                    cv2.circle(image, (x, y), 5, (255, 0, 0), -1)

        # Add frame data to collection
        landmark_data.append(frame_data)

        # Add progress text
        progress_text = f"Processing: {frame_count}/{total_frames} frames ({int(frame_count / total_frames * 100)}%)"
        cv2.putText(image, progress_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Write frame to output video
        out.write(image)

        # Display the frame with progress
        cv2.imshow('Processing Video - Press Q to cancel', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    pose.close()

    # Process and visualize landmark data
    if landmark_data:
        # Get peaks for highlight extraction
        peaks_info = visualize_landmark_data(landmark_data, os.path.dirname(output_path))

        # Create highlight clips around peaks
        if peaks_info is not None and peaks_info.shape[0] > 0:
            export_highlight_clips(video_path, original_frames, peaks_info, fps, os.path.dirname(output_path))

    return True