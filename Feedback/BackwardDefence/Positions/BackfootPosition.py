import os
import cv2
import numpy as np
import mediapipe as mp
from typing import Dict, Any, Tuple, List, Optional
from ...image_utils import calculate_distance
from ...image_utils import crop_and_save_image
from ...ref_images import BackwardDefence_Backfoot_Heel_ref_images
import uuid


def get_mediapipe_landmarks(frame_path: str, pose) -> Optional[Dict[str, Any]]:
    """
    Get MediaPipe landmarks from an image frame
    Returns landmarks for pose analysis with pixel coordinates
    """
    try:
        image = cv2.imread(frame_path)
        if image is None:
            print(f"[ERROR] Could not read image: {frame_path}")
            return None

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image_rgb.shape[:2]

        results = pose.process(image_rgb)
        if not results.pose_landmarks:
            print(f"[INFO] No pose landmarks detected in: {frame_path}")
            return None

        landmarks = results.pose_landmarks.landmark

        return {
            'right_heel': (landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HEEL.value].x * w,
                           landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HEEL.value].y * h),
            'right_ankle': (landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ANKLE.value].x * w,
                            landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ANKLE.value].y * h),
            'right_knee': (landmarks[mp.solutions.pose.PoseLandmark.RIGHT_KNEE.value].x * w,
                           landmarks[mp.solutions.pose.PoseLandmark.RIGHT_KNEE.value].y * h),
            'right_hip': (landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value].x * w,
                          landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value].y * h),
            'left_heel': (landmarks[mp.solutions.pose.PoseLandmark.LEFT_HEEL.value].x * w,
                          landmarks[mp.solutions.pose.PoseLandmark.LEFT_HEEL.value].y * h),
            'left_ankle': (landmarks[mp.solutions.pose.PoseLandmark.LEFT_ANKLE.value].x * w,
                           landmarks[mp.solutions.pose.PoseLandmark.LEFT_ANKLE.value].y * h),
            'frame_width': w,
            'frame_height': h
        }

    except Exception as e:
        print(f"[ERROR] Failed to get landmarks: {e}")
        return None


def analyze_backfoot_heel_movement(
        highlights_folder: str,
        frame_files: List[str],
        pose
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """
    Analyze backfoot heel movement across frames to find peak movement time
    Returns:
        - Dictionary of frame analysis data
        - Dictionary of peak movement frame data
    """
    frame_data = {}
    max_movement = 0
    peak_frame = None

    for frame_file in frame_files:
        frame_path = os.path.join(highlights_folder, frame_file)
        landmarks = get_mediapipe_landmarks(frame_path, pose)
        if not landmarks:
            continue

        # Calculate movement metrics
        heel_x = landmarks['right_heel'][0]
        ankle_x = landmarks['right_ankle'][0]
        knee_x = landmarks['right_knee'][0]
        hip_x = landmarks['right_hip'][0]

        # Check if right foot is actually the backfoot
        if heel_x > landmarks['left_heel'][0]:
            continue  # Skip if right foot is front foot

        movement = abs(heel_x - ankle_x) + abs(ankle_x - knee_x)

        frame_data[frame_file] = {
            'heel_x': heel_x,
            'ankle_x': ankle_x,
            'knee_x': knee_x,
            'hip_x': hip_x,
            'movement': movement,
            'landmarks': landmarks
        }

        # Track peak movement frame
        if movement > max_movement:
            max_movement = movement
            peak_frame = frame_data[frame_file]
            peak_frame['frame_file'] = frame_file

    return frame_data, peak_frame


def calculate_movement_averages(
        frame_data: Dict[str, Any],
        peak_frame: Dict[str, Any]
) -> Tuple[float, float]:
    """
    Calculate average heel x position before and after peak movement
    Returns:
        - before_peak_avg: Average x position before peak
        - after_peak_avg: Average x position after peak
    """
    if not frame_data or not peak_frame:
        return 0, 0

    peak_file = peak_frame['frame_file']
    frame_files_sorted = sorted(frame_data.keys())
    peak_index = frame_files_sorted.index(peak_file)

    # Get frames before and after peak
    before_frames = frame_files_sorted[:peak_index]
    after_frames = frame_files_sorted[peak_index + 1:]

    # Calculate averages
    before_avg = np.mean([frame_data[f]['heel_x'] for f in before_frames]) if before_frames else 0
    after_avg = np.mean([frame_data[f]['heel_x'] for f in after_frames]) if after_frames else 0

    return before_avg, after_avg


def find_highlight_clip(highlights_folder: str) -> Optional[str]:
    """
    Find the highlight clip in the highlights folder
    Returns path to the highlight clip or None if not found
    """
    highlight_clips = [f for f in os.listdir(highlights_folder)
                       if f.endswith('.mp4') and 'highest_peak' in f]

    if highlight_clips:
        return os.path.join(highlights_folder, highlight_clips[0])
    return None


def process_backfoot_position(
        highlights_folder: str,
        frame_files: List[str],
        pose,
        feedback_images_dir: str,
        is_left_handed: bool = False
) -> Dict[str, Any]:
    """
    Main function to process backfoot position analysis for backward defence
    Returns feedback dictionary with analysis results
    """
    try:
        # Find the highlight clip path
        highlight_clip_path = find_highlight_clip(highlights_folder)
        if not highlight_clip_path:
            print("[WARNING] No highlight clip found in the highlights folder")

        # Analyze all frames for backfoot movement
        frame_data, peak_frame = analyze_backfoot_heel_movement(highlights_folder, frame_files, pose)

        if not frame_data or not peak_frame:
            return {
                "feedback_no": 3,
                "title": "Backfoot Movement Analysis",
                "image_filename": "",
                "feedback_text": "Could not analyze backfoot movement. Ensure proper posture with visible backfoot.",
                "ref-images": BackwardDefence_Backfoot_Heel_ref_images,
                "is_ideal": False,
                "highlight_clip": highlight_clip_path
            }

        # Calculate movement averages
        before_avg, after_avg = calculate_movement_averages(frame_data, peak_frame)

        # Determine feedback based on movement pattern
        if after_avg < before_avg:
            feedback_text = (
                "Your backfoot movement is correct for backward defence. "
                "The backfoot should move slightly backward as you transfer weight to the front foot. "
                "This movement is crucial for maintaining balance and generating power in your shot."
            )
            is_ideal = True
        else:
            feedback_text = (
                "Check your backfoot movement. In backward defence, your backfoot should move slightly backward, "
                "not forward. Proper backfoot movement helps maintain balance and prepares you for quick footwork "
                "against short-pitched deliveries."
            )
            is_ideal = False

        # Crop the peak frame for visual feedback
        peak_frame_path = os.path.join(highlights_folder, peak_frame['frame_file'])
        cropped_image = crop_and_save_image(
            original_image_path=peak_frame_path,
            center_coords=(int(peak_frame['landmarks']['right_heel'][0]),
                           int(peak_frame['landmarks']['right_heel'][1])),
            crop_size=300,
            output_dir=feedback_images_dir,
            mirror_if_left_handed=is_left_handed
        )

        return {
            "feedback_no": 3,
            "title": "Backfoot Movement Analysis",
            "image_filename": cropped_image,
            "feedback_text": feedback_text,
            "ref-images": BackwardDefence_Backfoot_Heel_ref_images,
            "is_ideal": is_ideal,
            "analysis_data": {
                "before_peak_avg": float(before_avg),
                "after_peak_avg": float(after_avg),
                "peak_frame": peak_frame['frame_file']
            },
            "highlight_clip": highlight_clip_path
        }

    except Exception as e:
        print(f"[ERROR] Failed to process backfoot position: {e}")
        return {
            "feedback_no": 3,
            "title": "Backfoot Movement Analysis",
            "image_filename": "",
            "feedback_text": "Error analyzing backfoot movement. Please try again.",
            "ref-images": BackwardDefence_Backfoot_Heel_ref_images,
            "is_ideal": False,
            "highlight_clip": highlight_clip_path if 'highlight_clip_path' in locals() else None
        }