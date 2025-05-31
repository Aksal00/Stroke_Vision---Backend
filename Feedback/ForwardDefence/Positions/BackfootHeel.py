import os
import cv2
import numpy as np
import mediapipe as mp
from typing import Dict, Any, Tuple, List
import uuid
from ...image_utils import crop_and_save_image
from ...ref_images import ForwardDefence_Backfoot_Heel_ref_images

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

        # Define foot landmarks
        left_foot = {
            'heel': (landmarks[mp.solutions.pose.PoseLandmark.LEFT_HEEL.value].x * w,
                     landmarks[mp.solutions.pose.PoseLandmark.LEFT_HEEL.value].y * h),
            'ankle': (landmarks[mp.solutions.pose.PoseLandmark.LEFT_ANKLE.value].x * w,
                      landmarks[mp.solutions.pose.PoseLandmark.LEFT_ANKLE.value].y * h),
            'index': (landmarks[mp.solutions.pose.PoseLandmark.LEFT_FOOT_INDEX.value].x * w,
                      landmarks[mp.solutions.pose.PoseLandmark.LEFT_FOOT_INDEX.value].y * h),
            'knee': (landmarks[mp.solutions.pose.PoseLandmark.LEFT_KNEE.value].x * w,
                     landmarks[mp.solutions.pose.PoseLandmark.LEFT_KNEE.value].y * h)
        }

        right_foot = {
            'heel': (landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HEEL.value].x * w,
                     landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HEEL.value].y * h),
            'ankle': (landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ANKLE.value].x * w,
                      landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ANKLE.value].y * h),
            'index': (landmarks[mp.solutions.pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x * w,
                      landmarks[mp.solutions.pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y * h),
            'knee': (landmarks[mp.solutions.pose.PoseLandmark.RIGHT_KNEE.value].x * w,
                     landmarks[mp.solutions.pose.PoseLandmark.RIGHT_KNEE.value].y * h)
        }

        return {
            'left_foot': left_foot,
            'right_foot': right_foot
        }

    except Exception as e:
        print(f"[ERROR] Failed to get landmarks: {e}")
        return None


def analyze_backfoot_heel(frame_path: str, pose) -> Tuple[Dict[str, Any], Dict[str, Any], bool]:
    """
    Analyze backfoot position for a single frame
    Returns:
        - backfoot data
        - frame data
        - heel_lifted status
    """
    frame_data = get_mediapipe_landmarks(frame_path, pose)
    if not frame_data:
        return None, None, None

    # Determine which foot is backfoot (left side in image)
    left_foot_heel_x = frame_data['left_foot']['heel'][0]
    right_foot_heel_x = frame_data['right_foot']['heel'][0]
    is_left_backfoot = left_foot_heel_x < right_foot_heel_x

    backfoot = frame_data['left_foot'] if is_left_backfoot else frame_data['right_foot']

    # Validate foot position landmarks
    heel_x, heel_y = backfoot['heel']
    ankle_x, ankle_y = backfoot['ankle']
    knee_x, knee_y = backfoot['knee']

    # Check if heel and ankle are to the left of knee (x coordinate) and below knee (y coordinate)
    if not (heel_x < knee_x and ankle_x < knee_x and heel_y > knee_y and ankle_y > knee_y):
        return None, None, None

    # Calculate heel and ankle positions
    heel_y = backfoot['heel'][1]
    ankle_y = backfoot['ankle'][1]
    index_y = backfoot['index'][1]

    # Improved logic for checking heel lift
    heel_lifted = (ankle_y - heel_y) > (heel_y - index_y) * 0.5  # Adjusted threshold

    return backfoot, frame_data, heel_lifted


def create_backfoot_heel_feedback_image(
        highlights_folder: str,
        frame_file: str,
        backfoot: Dict[str, Any],
        feedback_images_dir: str,
        is_left_handed: bool = False
) -> str:
    """
    Create feedback image for backfoot analysis and return its filename
    """
    try:
        frame_path = os.path.join(highlights_folder, frame_file)

        # Get coordinates in pixels
        ankle_x, ankle_y = backfoot['ankle']
        knee_x, knee_y = backfoot['knee']

        # Calculate distance between knee and ankle
        knee_ankle_dist = np.sqrt((knee_x - ankle_x) ** 2 + (knee_y - ankle_y) ** 2)
        crop_size = int(2 * knee_ankle_dist)

        # Use the utility function to crop and save
        return crop_and_save_image(
            original_image_path=frame_path,
            center_coords=(int(ankle_x), int(ankle_y)),
            crop_size=crop_size,
            output_dir=feedback_images_dir,
            mirror_if_left_handed=is_left_handed
        )

    except Exception as e:
        print(f"[ERROR] Failed to create feedback image: {e}")
        return ""


def process_backfoot_heel_position(
        highlights_folder: str,
        frame_files: List[str],
        pose,
        feedback_images_dir: str,
        is_left_handed: bool = False
) -> Dict[str, Any]:
    """
    Main function to process backfoot position analysis
    Returns feedback dictionary with analysis results
    """
    try:
        valid_backfoot_data = []

        for frame_file in frame_files:
            frame_path = os.path.join(highlights_folder, frame_file)
            backfoot, frame_data, heel_lifted = analyze_backfoot_heel(frame_path, pose)

            if backfoot and frame_data:
                valid_backfoot_data.append({
                    'frame_file': frame_file,
                    'heel_lifted': heel_lifted,
                    'backfoot': backfoot,
                    'frame_data': frame_data
                })

        # If no valid frames found
        if not valid_backfoot_data:
            return {
                "feedback_no": 1,
                "title": "Backfoot Heel Position Analysis",
                "image_filename": "",
                "feedback_text": "Player heel position didn't recognize correctly. Please ensure proper posture for accurate analysis.",
                "ref-images": ForwardDefence_Backfoot_Heel_ref_images,
                "is_ideal": False
            }

        # Select frame where heel is most lifted (or least lifted if no lift detected)
        if any(item['heel_lifted'] for item in valid_backfoot_data):
            selected_frame = max(
                [item for item in valid_backfoot_data if item['heel_lifted']],
                key=lambda x: x['backfoot']['ankle'][1] - x['backfoot']['heel'][1]
            )
            is_ideal = True
        else:
            selected_frame = min(
                valid_backfoot_data,
                key=lambda x: x['backfoot']['ankle'][1] - x['backfoot']['heel'][1]
            )
            is_ideal = False

        # Create feedback image
        image_filename = create_backfoot_heel_feedback_image(
            highlights_folder,
            selected_frame['frame_file'],
            selected_frame['backfoot'],
            feedback_images_dir,
            is_left_handed
        )

        # Generate feedback text
        feedback_text = "When playing forward defence the backfoot heel should lift to maintain a proper balance."
        if selected_frame['heel_lifted']:
            feedback_text += " When considering your back foot, it seems the heel is lifted. Nice work!"
        else:
            feedback_text += " But when considering your backfoot, it doesn't seem like the heel is lifting enough."

        return {
            "feedback_no": 1,
            "title": "Backfoot Heel Position Analysis",
            "image_filename": image_filename,
            "feedback_text": feedback_text,
            "ref-images": ForwardDefence_Backfoot_Heel_ref_images,
            "is_ideal": is_ideal
        }

    except Exception as e:
        print(f"[ERROR] Failed to process backfoot position: {e}")
        return None