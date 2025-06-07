import os
import cv2
import numpy as np
import mediapipe as mp
from typing import Dict, Any, Tuple, List
import uuid
from ...image_utils import crop_and_save_image
from ...ref_images import ForwardDefence_Backfoot_Knee_ref_images


def get_mediapipe_landmarks(frame_path: str, pose) -> Dict[str, Any]:
    """
    Get MediaPipe landmarks from an image frame
    Returns landmarks for pose analysis
    """
    try:
        # Read the image
        image = cv2.imread(frame_path)
        if image is None:
            print(f"[ERROR] Could not read image: {frame_path}")
            return None

        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image_rgb.shape[:2]

        # Process image with MediaPipe
        results = pose.process(image_rgb)
        if not results.pose_landmarks:
            print(f"[INFO] No pose landmarks detected in: {frame_path}")
            return None

        # Extract key landmarks
        landmarks = results.pose_landmarks.landmark

        # Define foot and leg landmarks
        left_leg = {
            'ankle': (landmarks[mp.solutions.pose.PoseLandmark.LEFT_ANKLE.value].x * w,
                      landmarks[mp.solutions.pose.PoseLandmark.LEFT_ANKLE.value].y * h),
            'knee': (landmarks[mp.solutions.pose.PoseLandmark.LEFT_KNEE.value].x * w,
                     landmarks[mp.solutions.pose.PoseLandmark.LEFT_KNEE.value].y * h),
            'hip': (landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value].x * w,
                    landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value].y * h),
            'heel': (landmarks[mp.solutions.pose.PoseLandmark.LEFT_HEEL.value].x * w,
                     landmarks[mp.solutions.pose.PoseLandmark.LEFT_HEEL.value].y * h)
        }

        right_leg = {
            'ankle': (landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ANKLE.value].x * w,
                      landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ANKLE.value].y * h),
            'knee': (landmarks[mp.solutions.pose.PoseLandmark.RIGHT_KNEE.value].x * w,
                     landmarks[mp.solutions.pose.PoseLandmark.RIGHT_KNEE.value].y * h),
            'hip': (landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value].x * w,
                    landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value].y * h),
            'heel': (landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HEEL.value].x * w,
                     landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HEEL.value].y * h)
        }

        return {
            'left_leg': left_leg,
            'right_leg': right_leg
        }

    except Exception as e:
        print(f"[ERROR] Failed to get landmarks: {e}")
        return None


def analyze_backfoot_knee(frame_path: str, pose) -> Tuple[Dict[str, Any], Dict[str, Any], bool]:
    """
    Analyze backfoot knee position for a single frame
    Returns:
        - backfoot data
        - frame data
        - is_backfoot_left (True if left foot is backfoot)
    """
    frame_data = get_mediapipe_landmarks(frame_path, pose)
    if not frame_data:
        return None, None, None

    # Determine which foot is backfoot (left side in image)
    left_foot_heel_x = frame_data['left_leg']['heel'][0]
    right_foot_heel_x = frame_data['right_leg']['heel'][0]
    is_left_backfoot = left_foot_heel_x < right_foot_heel_x

    backfoot = frame_data['left_leg'] if is_left_backfoot else frame_data['right_leg']
    frontfoot = frame_data['right_leg'] if is_left_backfoot else frame_data['left_leg']

    # Check landmark validity
    backfoot_valid = backfoot['knee'][0] < backfoot['hip'][0]
    frontfoot_valid = frontfoot['knee'][0] > frontfoot['hip'][0]

    if not (backfoot_valid and frontfoot_valid):
        return None, None, None

    return backfoot, frame_data, is_left_backfoot


def create_backfoot_knee_feedback_image(
        highlights_folder: str,
        frame_file: str,
        backfoot: Dict[str, Any],
        feedback_images_dir: str,
        is_left_handed: bool = False
) -> str:
    """
    Create feedback image for backfoot knee analysis using the original frame with background
    and return its filename
    Args:
        highlights_folder: Path to folder containing the processed frame
        frame_file: Filename of the frame
        backfoot: Dictionary containing backfoot landmarks (from processed frame)
        feedback_images_dir: Directory to save feedback images
        is_left_handed: Whether player is left-handed (for mirroring)
    Returns:
        Filename of the saved feedback image
    """
    try:
        # Get the path to the original frame with background in the for_feedback_output folder
        feedback_output_folder = os.path.join(os.path.dirname(os.path.dirname(highlights_folder)),
                                              "for_feedback_output")
        frame_path = os.path.join(feedback_output_folder, frame_file)

        if not os.path.exists(frame_path):
            # Fallback to the processed frame if original not found
            frame_path = os.path.join(highlights_folder, frame_file)
            print(f"[WARNING] Original frame with background not found, using processed frame: {frame_file}")

        # Get coordinates in pixels (from processed frame analysis)
        ankle_x, ankle_y = backfoot['ankle']
        knee_x, knee_y = backfoot['knee']

        # Calculate distance between knee and ankle
        knee_ankle_dist = np.sqrt((knee_x - ankle_x) ** 2 + (knee_y - ankle_y) ** 2)
        crop_size = int(2 * knee_ankle_dist)

        # Ensure crop_size is at least 100px to avoid too small crops
        crop_size = max(crop_size, 100)

        # Use the utility function to crop and save
        return crop_and_save_image(
            original_image_path=frame_path,
            center_coords=(int(knee_x), int(knee_y)),
            crop_size=crop_size,
            output_dir=feedback_images_dir,
            mirror_if_left_handed=is_left_handed
        )

    except Exception as e:
        print(f"[ERROR] Failed to create feedback image: {e}")
        return ""


def process_backfoot_knee_position(
        highlights_folder: str,
        frame_files: List[str],
        pose,
        feedback_images_dir: str,
        is_left_handed: bool = False
) -> Dict[str, Any]:
    """
    Main function to process backfoot knee position analysis
    Returns feedback dictionary with analysis results
    """
    try:
        valid_backfoot_data = []

        # Process frames without background for calculations
        for frame_file in frame_files:
            frame_path = os.path.join(highlights_folder, frame_file)
            backfoot, frame_data, is_left_backfoot = analyze_backfoot_knee(frame_path, pose)

            if backfoot and frame_data:
                hip_x = backfoot['hip'][0]
                valid_backfoot_data.append({
                    'frame_file': frame_file,
                    'backfoot': backfoot,
                    'frame_data': frame_data,
                    'hip_x': hip_x,
                    'is_left_backfoot': is_left_backfoot
                })

        if not valid_backfoot_data:
            return {
                "feedback_no": 2,
                "title": "Backfoot Knee Position Analysis",
                "image_filename": "",
                "feedback_text": "Player knee position didn't recognize correctly. Please ensure proper posture for accurate analysis.",
                "ref-images": ForwardDefence_Backfoot_Knee_ref_images,
                "is_ideal": False
            }

        # Select frame where hip has highest x coordinate (most forward position)
        selected_frame = max(valid_backfoot_data, key=lambda x: x['hip_x'])

        # Extract coordinates
        backfoot = selected_frame['backfoot']
        hip_x = backfoot['hip'][0]
        knee_x = backfoot['knee'][0]
        ankle_x = backfoot['ankle'][0]

        # Calculate distances
        hip_knee_diff = hip_x - knee_x
        knee_ankle_diff = knee_x - ankle_x

        # Generate base feedback
        feedback_text = "Your back knee should be slightly bent, not locked straight. This gives you balance and helps transfer your weight smoothly to the front foot."

        # Determine if position is ideal
        is_ideal = False

        # Add specific feedback based on knee position
        if hip_knee_diff < knee_ankle_diff / 3:
            feedback_text += " It seems like your knee has bent too much."
            is_ideal = False
        elif hip_knee_diff >= knee_ankle_diff / 3 and hip_knee_diff < knee_ankle_diff:
            feedback_text += " It seems like your backfoot knee is technically okay. Keep it up!"
            is_ideal = True
        else:
            feedback_text += " It seems like your backfoot knee is locked straight."
            is_ideal = False

        # Create feedback image using the original frame with background
        image_filename = create_backfoot_knee_feedback_image(
            highlights_folder,
            selected_frame['frame_file'],
            selected_frame['backfoot'],
            feedback_images_dir,
            is_left_handed
        )

        return {
            "feedback_no": 2,
            "title": "Backfoot Knee Position Analysis",
            "image_filename": image_filename,
            "feedback_text": feedback_text,
            "ref-images": ForwardDefence_Backfoot_Knee_ref_images,
            "is_ideal": is_ideal
        }

    except Exception as e:
        print(f"[ERROR] Failed to process backfoot knee position: {e}")
        return None