import os
import cv2
import numpy as np
import mediapipe as mp
from typing import Dict, Any, Tuple, List
import uuid
from ...image_utils import crop_and_save_image, calculate_anticlockwise_angle
from ...ref_images import BackwardDefence_Frontfoot_Knee_ref_images

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

def analyze_frontfoot_knee_angle(
        frame_path: str,
        pose
) -> Tuple[Dict[str, Any], Dict[str, Any], bool, float]:
    """
    Analyze frontfoot knee angle for a single frame in backward defence
    Returns:
        - frontfoot data
        - frame data
        - is_frontfoot_left (True if left foot is frontfoot)
        - knee angle in degrees
    """
    frame_data = get_mediapipe_landmarks(frame_path, pose)
    if not frame_data:
        return None, None, None, 0

    # Determine which foot is frontfoot (right side in image)
    left_foot_heel_x = frame_data['left_leg']['heel'][0]
    right_foot_heel_x = frame_data['right_leg']['heel'][0]
    is_frontfoot_left = left_foot_heel_x > right_foot_heel_x

    frontfoot = frame_data['left_leg'] if is_frontfoot_left else frame_data['right_leg']
    backfoot = frame_data['right_leg'] if is_frontfoot_left else frame_data['left_leg']

    # Check landmark validity
    frontfoot_valid = frontfoot['knee'][0] > frontfoot['hip'][0]
    backfoot_valid = backfoot['knee'][0] < backfoot['hip'][0]

    if not (frontfoot_valid and backfoot_valid):
        return None, None, None, 0

    # Calculate anticlockwise knee angle (ankle-knee-hip)
    point_a = frontfoot['ankle']
    point_b = frontfoot['knee']
    point_c = frontfoot['hip']
    knee_angle = calculate_anticlockwise_angle(point_a, point_b, point_c)

    return frontfoot, frame_data, is_frontfoot_left, knee_angle

def create_frontfoot_knee_feedback_image(
        highlights_folder: str,
        frame_file: str,
        frontfoot: Dict[str, Any],
        feedback_images_dir: str,
        is_left_handed: bool = False
) -> str:
    """
    Create feedback image for frontfoot knee analysis and return its filename
    Args:
        highlights_folder: Path to folder containing the frame
        frame_file: Filename of the frame
        frontfoot: Dictionary containing frontfoot landmarks
        feedback_images_dir: Directory to save feedback images
        is_left_handed: Whether player is left-handed (for mirroring)
    Returns:
        Filename of the saved feedback image
    """
    try:
        frame_path = os.path.join(highlights_folder, frame_file)

        # Get coordinates in pixels
        ankle_x, ankle_y = frontfoot['ankle']
        knee_x, knee_y = frontfoot['knee']

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

def process_frontfoot_knee_position(
        highlights_folder: str,
        frame_files: List[str],
        pose,
        feedback_images_dir: str,
        is_left_handed: bool = False
) -> Dict[str, Any]:
    """
    Main function to process frontfoot knee position analysis for backward defence
    Returns feedback dictionary with analysis results
    """
    try:
        valid_frontfoot_data = []

        for frame_file in frame_files:
            frame_path = os.path.join(highlights_folder, frame_file)
            frontfoot, frame_data, is_frontfoot_left, knee_angle = analyze_frontfoot_knee_angle(frame_path, pose)

            if frontfoot and frame_data and knee_angle > 0:
                hip_x = frontfoot['hip'][0]
                valid_frontfoot_data.append({
                    'frame_file': frame_file,
                    'frontfoot': frontfoot,
                    'frame_data': frame_data,
                    'hip_x': hip_x,
                    'is_frontfoot_left': is_frontfoot_left,
                    'knee_angle': knee_angle
                })

        if not valid_frontfoot_data:
            return {
                "feedback_no": 3,
                "title": "Frontfoot Knee Position Analysis",
                "image_filename": "",
                "feedback_text": "Player knee position didn't recognize correctly. Please ensure proper posture for accurate analysis.",
                "ref-images": BackwardDefence_Frontfoot_Knee_ref_images,
                "is_ideal": False
            }

        # Select frame with maximum hip extension (most forward position)
        selected_frame = max(valid_frontfoot_data, key=lambda x: x['hip_x'])

        # Extract data
        frontfoot = selected_frame['frontfoot']
        knee_angle = selected_frame['knee_angle']

        # Generate feedback based on knee angle
        if knee_angle > 150:
            feedback_text = (
                "Your frontfoot knee position is ideal for backward defence. "
                "This position provides the right balance between stability and mobility."
            )
            is_ideal = True
        elif knee_angle > 130:
            feedback_text = (
                "Your frontfoot knee could be slightly straighter. "
                "While some bend is acceptable, a straighter front leg provides better stability "
                "against short-pitched deliveries in backward defence."
            )
            is_ideal = False
        else:
            feedback_text = (
                "Your frontfoot knee is bent too much. "
                "In backward defence, your front leg should be nearly straight to provide a solid base. "
                "Practice keeping your front leg straighter while maintaining balance."
            )
            is_ideal = False

        # Add general advice
        feedback_text += (
            "\n\nIn backward defence, your frontfoot helps control your weight transfer. "
            "The knee should be nearly straight to maintain stability while allowing "
            "controlled movement against short-pitched deliveries."
        )

        # Create feedback image
        image_filename = create_frontfoot_knee_feedback_image(
            highlights_folder,
            selected_frame['frame_file'],
            selected_frame['frontfoot'],
            feedback_images_dir,
            is_left_handed
        )

        return {
            "feedback_no": 3,
            "title": "Frontfoot Knee Position Analysis",
            "image_filename": image_filename,
            "feedback_text": feedback_text,
            "ref-images": BackwardDefence_Frontfoot_Knee_ref_images,
            "is_ideal": is_ideal
        }

    except Exception as e:
        print(f"[ERROR] Failed to process frontfoot knee position: {e}")
        return None