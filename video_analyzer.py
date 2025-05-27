import cv2
import numpy as np
from typing import List
import mediapipe as mp


def verify_pose_detection(frame: np.ndarray) -> bool:
    """
    Verify if MediaPipe can detect pose in a given frame

    Args:
        frame: Input frame in BGR format

    Returns:
        bool: True if pose is detected, False otherwise
    """
    mp_pose = mp.solutions.pose
    with mp_pose.Pose(
            static_image_mode=True,
            model_complexity=1,
            min_detection_confidence=0.5
    ) as pose:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)
        return results.pose_landmarks is not None

def video_analyzer(video_path: str) -> tuple[List[np.ndarray], float]:
    """
    Extract frames and FPS from a video file.

    Args:
        video_path: Path to the video file

    Returns:
        tuple: (list of frames as numpy arrays, video fps)
    """
    frames = []
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    return frames, fps


import cv2
import mediapipe as mp
import numpy as np
from typing import List, Tuple


def verify_pose_detection(frame: np.ndarray) -> bool:
    """
    Verify if MediaPipe can detect pose in a given frame

    Args:
        frame: Input frame in BGR format

    Returns:
        bool: True if pose is detected, False otherwise
    """
    mp_pose = mp.solutions.pose
    with mp_pose.Pose(
            static_image_mode=True,
            model_complexity=1,
            min_detection_confidence=0.5
    ) as pose:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)
        return results.pose_landmarks is not None


def select_valid_frames(
        frames: List[np.ndarray],
        frame_indices: List[int],
        candidate_indices: List[int],
        num_frames: int = 3
) -> Tuple[List[int], List[np.ndarray]]:
    """
    Select frames where pose detection is successful

    Args:
        frames: List of all video frames
        frame_indices: List of original frame indices
        candidate_indices: Initial candidate indices to check
        num_frames: Number of valid frames to select

    Returns:
        Tuple of (selected_indices, selected_frames)
    """
    selected_indices = []
    selected_frames = []

    # Check initial candidates
    for idx in candidate_indices:
        if len(selected_indices) >= num_frames:
            break

        frame_idx = frame_indices[idx]
        if verify_pose_detection(frames[frame_idx]):
            selected_indices.append(frame_idx)
            selected_frames.append(frames[frame_idx])

    # If we still need more frames, look at nearby frames
    if len(selected_indices) < num_frames:
        remaining = num_frames - len(selected_indices)
        # Create a list of all possible indices sorted by distance to candidates
        all_indices = sorted(
            range(len(frame_indices)),
            key=lambda x: min(abs(x - c) for c in candidate_indices)
        )

        for idx in all_indices:
            if len(selected_indices) >= num_frames:
                break

            frame_idx = frame_indices[idx]
            if frame_idx not in selected_indices and verify_pose_detection(frames[frame_idx]):
                selected_indices.append(frame_idx)
                selected_frames.append(frames[frame_idx])

    return selected_indices[:num_frames], selected_frames[:num_frames]
