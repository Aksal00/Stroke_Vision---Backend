import cv2
import mediapipe as mp
import numpy as np
import os
import logging
from typing import List, Dict, Any

# Import visualization modules
from visualization import visualize_landmark_data
from highlight_generator import export_highlight_clips

# Set up logging
logger = logging.getLogger(__name__)


def analyze_cricket_landmarks(
        model_path: str,
        video_path: str,
        output_path: str,
        batch_size: int = 100
) -> bool:
    """
    Analyze cricket player's pose and movements from a video.

    Args:
        model_path: Path to the shot classification model
        video_path: Path to the input video file
        output_path: Path where analyzed video will be saved
        batch_size: Number of frames to process at a time

    Returns:
        bool: True if analysis completed successfully
    """
    logger.info(f"Starting video analysis: {video_path}")

    # Initialize MediaPipe Pose
    try:
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
    except Exception as e:
        logger.error(f"Failed to initialize MediaPipe: {e}")
        return False

    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Could not open video file: {video_path}")
        return False

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    logger.info(f"Video properties: {width}x{height} @ {fps}fps, {total_frames} frames")

    # Prepare output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not out.isOpened():
        logger.error(f"Could not create output video: {output_path}")
        cap.release()
        return False

    # Data collection
    landmark_data: List[Dict[str, Any]] = []
    original_frames: List[np.ndarray] = []
    frame_count = 0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            current_time = frame_count / fps

            # Store original frame
            original_frames.append(frame.copy())
            if len(original_frames) > batch_size:
                original_frames.pop(0)  # Keep only the most recent frames

            # Process pose estimation
            try:
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

                    # Store landmark data (fixed indentation issue)
                    landmarks = results.pose_landmarks.landmark
                    for idx, landmark in enumerate(landmarks):
                        frame_data[f'landmark_{idx}_x'] = landmark.x
                        frame_data[f'landmark_{idx}_y'] = landmark.y
                        frame_data[f'landmark_{idx}_z'] = landmark.z
                        frame_data[f'landmark_{idx}_visibility'] = landmark.visibility

                        # Highlight important cricket joints
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
                            x = int(landmark.x * width)
                            y = int(landmark.y * height)
                            cv2.circle(image, (x, y), 5, (255, 0, 0), -1)

                    # Add progress text
                    progress_text = f"Processing: {frame_count}/{total_frames} frames ({int(frame_count / total_frames * 100)}%)"
                    cv2.putText(image, progress_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

                    # Write frame
                    out.write(image)

                    # Add frame data
                    landmark_data.append(frame_data)

                    # Log progress
                    if frame_count % 100 == 0 or frame_count == total_frames:
                        logger.info(f"Processed {frame_count}/{total_frames} frames")

            except Exception as e:
                logger.error(f"Error processing frame {frame_count}: {e}")
                continue

    except Exception as e:
        logger.error(f"Error during video processing: {e}")
        return False

    finally:
        # Release resources
        cap.release()
        out.release()
        pose.close()
        logger.info("Released video resources")

    # Process landmark data if we have results
    if landmark_data:
        logger.info("Analyzing landmark data")
        peaks_info = visualize_landmark_data(landmark_data, os.path.dirname(output_path))

        if peaks_info is not None and not peaks_info.empty:
            logger.info("Exporting highlight clips")
            export_success = export_highlight_clips(
                model_path, video_path, original_frames, peaks_info, fps, os.path.dirname(output_path))

            if not export_success:
                logger.warning("Highlight clip export failed")
        else:
            logger.warning("No significant movements detected")

    logger.info("Analysis complete")
    return True