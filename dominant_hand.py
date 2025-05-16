import cv2
import mediapipe as mp
import numpy as np
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


def determine_dominant_hand(video_path: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Determine the dominant hand of the batsman by analyzing the video frames.

    Args:
        video_path: Path to the input video file

    Returns:
        Tuple: (dominant_hand, mirrored_video_path) where:
            - dominant_hand: 'left' or 'right'
            - mirrored_video_path: Path to mirrored video if needed, None otherwise
    """
    try:
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

        # Open video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Could not open video file: {video_path}")
            return None, None

        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Counters for left/right dominant frames
        left_count = 0
        right_count = 0
        total_frames = 0

        # Process frames to determine dominant hand
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            total_frames += 1

            # Process pose estimation
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            if results.pose_landmarks:
                # Get nose and both ears landmarks
                nose = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
                left_ear = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EAR]
                right_ear = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EAR]

                # Calculate distances from nose to each ear
                dist_left = abs(nose.x - left_ear.x)
                dist_right = abs(nose.x - right_ear.x)

                # Determine which ear is the "back ear" (farther from nose)
                if dist_left > dist_right:
                    back_ear_x = left_ear.x
                else:
                    back_ear_x = right_ear.x

                # Check nose position relative to back ear
                if nose.x < back_ear_x:
                    left_count += 1
                else:
                    right_count += 1

        cap.release()
        pose.close()

        # Determine dominant hand
        if left_count > right_count:
            dominant_hand = 'left'
            # Create mirrored video
            mirrored_path = video_path.replace('.mp4', '_mirrored.mp4')
            if not mirror_video(video_path, mirrored_path):
                logger.error("Failed to create mirrored video")
                return None, None
            return dominant_hand, mirrored_path
        else:
            dominant_hand = 'right'
            return dominant_hand, None

    except Exception as e:
        logger.error(f"Error determining dominant hand: {e}")
        return None, None


def mirror_video(input_path: str, output_path: str) -> bool:
    """
    Create a horizontally mirrored version of the input video.

    Args:
        input_path: Path to input video
        output_path: Path to save mirrored video

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            return False

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            mirrored_frame = cv2.flip(frame, 1)
            out.write(mirrored_frame)

        cap.release()
        out.release()
        return True

    except Exception as e:
        logger.error(f"Error mirroring video: {e}")
        return False