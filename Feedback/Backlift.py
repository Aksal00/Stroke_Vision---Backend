import os
import cv2
import numpy as np
import mediapipe as mp
from typing import Dict, Any, Tuple, List, Optional
import uuid
import math
import logging
from .image_utils import crop_and_save_image

# Configure logging
logger = logging.getLogger(__name__)


def get_all_landmarks(frame_path: str, pose) -> Optional[Dict[str, Any]]:
    """
    Get all relevant MediaPipe landmarks for backlift analysis
    Returns dictionary of all needed landmarks in pixel coordinates
    """
    try:
        image = cv2.imread(frame_path)
        if image is None:
            logger.error(f"Could not read image: {frame_path}")
            return None

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image_rgb.shape[:2]

        results = pose.process(image_rgb)
        if not results.pose_landmarks:
            logger.info(f"No pose landmarks detected in: {frame_path}")
            return None

        landmarks = results.pose_landmarks.landmark
        mp_pose = mp.solutions.pose.PoseLandmark

        # Extract all needed landmarks
        return {
            'nose': (landmarks[mp_pose.NOSE.value].x * w, landmarks[mp_pose.NOSE.value].y * h),
            'left_ear': (landmarks[mp_pose.LEFT_EAR.value].x * w, landmarks[mp_pose.LEFT_EAR.value].y * h),
            'right_ear': (landmarks[mp_pose.RIGHT_EAR.value].x * w, landmarks[mp_pose.RIGHT_EAR.value].y * h),
            'left_shoulder': (
            landmarks[mp_pose.LEFT_SHOULDER.value].x * w, landmarks[mp_pose.LEFT_SHOULDER.value].y * h),
            'right_shoulder': (
            landmarks[mp_pose.RIGHT_SHOULDER.value].x * w, landmarks[mp_pose.RIGHT_SHOULDER.value].y * h),
            'left_elbow': (landmarks[mp_pose.LEFT_ELBOW.value].x * w, landmarks[mp_pose.LEFT_ELBOW.value].y * h),
            'right_elbow': (landmarks[mp_pose.RIGHT_ELBOW.value].x * w, landmarks[mp_pose.RIGHT_ELBOW.value].y * h),
            'left_wrist': (landmarks[mp_pose.LEFT_WRIST.value].x * w, landmarks[mp_pose.LEFT_WRIST.value].y * h),
            'right_wrist': (landmarks[mp_pose.RIGHT_WRIST.value].x * w, landmarks[mp_pose.RIGHT_WRIST.value].y * h),
            'left_hip': (landmarks[mp_pose.LEFT_HIP.value].x * w, landmarks[mp_pose.LEFT_HIP.value].y * h),
            'right_hip': (landmarks[mp_pose.RIGHT_HIP.value].x * w, landmarks[mp_pose.RIGHT_HIP.value].y * h),
            'left_index': (landmarks[mp_pose.LEFT_INDEX.value].x * w, landmarks[mp_pose.LEFT_INDEX.value].y * h),
            'right_index': (landmarks[mp_pose.RIGHT_INDEX.value].x * w, landmarks[mp_pose.RIGHT_INDEX.value].y * h)
        }
    except Exception as e:
        logger.error(f"Failed to get landmarks: {e}")
        return None


def is_proper_stance(landmarks: Dict[str, Any]) -> bool:
    """
    Check if the stance meets all required conditions for backlift analysis
    Returns False with specific reason if any condition fails
    """
    try:
        # Nose should be right side from right ear (image coordinates)
        nose_x = landmarks['nose'][0]
        right_ear_x = landmarks['right_ear'][0]

        if not (nose_x > right_ear_x):
            logger.debug("STANCE CHECK FAILED: Nose not to the right of right ear")
            logger.debug(f"Right ear X: {right_ear_x:.2f}, Nose X: {nose_x:.2f}")
            return False

        # Right arm conditions (player's right arm - appears on left in image)
        right_shoulder = landmarks['right_shoulder']
        right_elbow = landmarks['right_elbow']
        right_wrist = landmarks['right_wrist']

        # Right elbow should be left and below right shoulder
        if not (right_elbow[0] < right_shoulder[0]):
            logger.debug("STANCE CHECK FAILED: Right elbow not left of right shoulder (X position)")
            logger.debug(f"Elbow X: {right_elbow[0]:.2f}, Shoulder X: {right_shoulder[0]:.2f}")
            return False
        if not (right_elbow[1] > right_shoulder[1]):
            logger.debug("STANCE CHECK FAILED: Right elbow not below right shoulder (Y position)")
            logger.debug(f"Elbow Y: {right_elbow[1]:.2f}, Shoulder Y: {right_shoulder[1]:.2f}")
            return False

        # Right wrist should be right and below right elbow
        if not (right_wrist[0] > right_elbow[0]):
            logger.debug("STANCE CHECK FAILED: Right wrist not right of right elbow (X position)")
            logger.debug(f"Wrist X: {right_wrist[0]:.2f}, Elbow X: {right_elbow[0]:.2f}")
            return False
        if not (right_wrist[1] > right_elbow[1]):
            logger.debug("STANCE CHECK FAILED: Right wrist not below right elbow (Y position)")
            logger.debug(f"Wrist Y: {right_wrist[1]:.2f}, Elbow Y: {right_elbow[1]:.2f}")
            return False

        # Left arm conditions (player's left arm - appears on right in image)
        left_shoulder = landmarks['left_shoulder']
        left_elbow = landmarks['left_elbow']
        left_wrist = landmarks['left_wrist']

        # Left elbow should be below left shoulder
        if not (left_elbow[1] > left_shoulder[1]):
            logger.debug("STANCE CHECK FAILED: Left elbow not below left shoulder (Y position)")
            logger.debug(f"Elbow Y: {left_elbow[1]:.2f}, Shoulder Y: {left_shoulder[1]:.2f}")
            return False

        # Left wrist should be left and below left elbow
        if not (left_wrist[0] < left_elbow[0]):
            logger.debug("STANCE CHECK FAILED: Left wrist not left of left elbow (X position)")
            logger.debug(f"Wrist X: {left_wrist[0]:.2f}, Elbow X: {left_elbow[0]:.2f}")
            return False
        if not (left_wrist[1] > left_elbow[1]):
            logger.debug("STANCE CHECK FAILED: Left wrist not below left elbow (Y position)")
            logger.debug(f"Wrist Y: {left_wrist[1]:.2f}, Elbow Y: {left_elbow[1]:.2f}")
            return False

        # Relative position between wrists
        if not (right_wrist[0] < left_wrist[0]):
            logger.debug("STANCE CHECK FAILED: Right wrist not left of left wrist (X position)")
            logger.debug(f"Right wrist X: {right_wrist[0]:.2f}, Left wrist X: {left_wrist[0]:.2f}")
            return False
        if not (right_wrist[1] < left_wrist[1]):
            logger.debug("STANCE CHECK FAILED: Right wrist not above left wrist (Y position)")
            logger.debug(f"Right wrist Y: {right_wrist[1]:.2f}, Left wrist Y: {left_wrist[1]:.2f}")
            return False

        logger.debug("STANCE CHECK PASSED: All conditions met")
        return True

    except Exception as e:
        logger.error(f"Failed to check stance conditions: {e}")
        return False


def calculate_backlift_angle(left_wrist: Tuple[float, float], right_wrist: Tuple[float, float]) -> float:
    """
    Calculate the backlift angle as the anti-clockwise angle between:
    1. The line connecting left wrist to right wrist
    2. The vertical line going down from left wrist

    Returns angle in degrees (0-180)
    """
    # Create wrist-to-wrist vector (left to right)
    wrist_vec = (right_wrist[0] - left_wrist[0], right_wrist[1] - left_wrist[1])

    # Create vertical vector (pointing down from left wrist)
    vertical_vec = (0, 1)  # Positive Y is downward in image coordinates

    # Calculate angle between the two vectors
    dot_product = wrist_vec[0] * vertical_vec[0] + wrist_vec[1] * vertical_vec[1]
    mag_wrist = math.sqrt(wrist_vec[0] ** 2 + wrist_vec[1] ** 2)
    mag_vertical = math.sqrt(vertical_vec[0] ** 2 + vertical_vec[1] ** 2)

    # Avoid division by zero
    if mag_wrist * mag_vertical == 0:
        return 0.0

    # Calculate the angle in radians
    angle_rad = math.acos(dot_product / (mag_wrist * mag_vertical))
    angle_deg = math.degrees(angle_rad)

    # Determine if we need to adjust the angle (since acos gives the smallest angle)
    # We want the anti-clockwise angle from vertical to wrist line
    if wrist_vec[0] < 0:  # If the wrist line is pointing left
        angle_deg = 360 - angle_deg

    # Return angle between 0-180 degrees
    return angle_deg if angle_deg <= 180 else 360 - angle_deg


def create_backlift_feedback_image(
        frame_path: str,
        landmarks: Dict[str, Any],
        feedback_images_dir: str,
        is_left_handed: bool = False
) -> str:
    """
    Create clean feedback image for backlift analysis centered around right hip
    Shows only the wrist-to-wrist line without landmark annotations
    """
    try:
        right_hip = landmarks['right_hip']
        right_index = landmarks['right_index']

        # Calculate crop size (distance from hip to index)
        crop_radius = math.sqrt((right_hip[0] - right_index[0]) ** 2 + (right_hip[1] - right_index[1]) ** 2)
        crop_size = int(2 * crop_radius)
        crop_size = max(crop_size, 200)  # Ensure minimum size

        # Use the utility function to crop and save
        return crop_and_save_image(
            original_image_path=frame_path,
            center_coords=(int(right_hip[0]), int(right_hip[1])),
            crop_size=crop_size,
            output_dir=feedback_images_dir,
            mirror_if_left_handed=is_left_handed
        )

    except Exception as e:
        logger.error(f"Failed to create feedback image: {e}")
        return ""


def process_backlift_angle(
        highlights_folder: str,
        frame_files: List[str],
        pose,
        feedback_images_dir: str,
        is_left_handed: bool = False
) -> Dict[str, Any]:
    """
    Process backlift angle analysis for cricket batting stance
    Now checks every frame in the highlight video clip
    Returns clean feedback without frame numbers
    """
    try:
        valid_frames = []

        # Find the highlight clip
        clip_files = [f for f in os.listdir(highlights_folder) if f.endswith('.mp4')]
        if not clip_files:
            return {
                "feedback_no": 0,
                "title": "Backlift Angle Analysis",
                "image_filename": "",
                "feedback_text": "No highlight clip found for analysis",
                "ref-images": ["Backlift.png", "BatGrip.png"],
                "is_ideal": False
            }

        clip_path = os.path.join(highlights_folder, clip_files[0])
        cap = cv2.VideoCapture(clip_path)

        if not cap.isOpened():
            return {
                "feedback_no": 0,
                "title": "Backlift Angle Analysis",
                "image_filename": "",
                "feedback_text": "Could not open highlight clip",
                "ref-images": ["Backlift.png", "BatGrip.png"],
                "is_ideal": False
            }

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            temp_frame_path = os.path.join(highlights_folder, f"temp_frame_{frame_count}.jpg")
            cv2.imwrite(temp_frame_path, frame)

            landmarks = get_all_landmarks(temp_frame_path, pose)
            os.remove(temp_frame_path)

            if landmarks and is_proper_stance(landmarks):
                y_distance = abs(landmarks['left_wrist'][1] - landmarks['right_wrist'][1])
                valid_frames.append({
                    'landmarks': landmarks,
                    'y_distance': y_distance,
                    'frame': frame.copy()
                })

        cap.release()

        if not valid_frames:
            return {
                "feedback_no": 0,
                "title": "Backlift Angle Analysis",
                "image_filename": "",
                "feedback_text": "Could not detect proper batting stance. Please ensure proper posture.",
                "ref-images": ["Backlift.png", "BatGrip.png"],
                "is_ideal": False
            }

        # Select best frame (maximum vertical distance between wrists)
        selected_frame = max(valid_frames, key=lambda x: x['y_distance'])
        angle = calculate_backlift_angle(
            selected_frame['landmarks']['left_wrist'],
            selected_frame['landmarks']['right_wrist']
        )

        # Determine if angle is ideal (between 90-150 degrees)
        is_ideal = 90 <= angle <= 150

        # Create clean feedback image
        temp_frame_path = os.path.join(highlights_folder, "temp_selected_frame.jpg")
        cv2.imwrite(temp_frame_path, selected_frame['frame'])
        image_filename = create_backlift_feedback_image(
            temp_frame_path,
            selected_frame['landmarks'],
            feedback_images_dir,
            is_left_handed
        )
        os.remove(temp_frame_path)

        # Generate feedback text based on angle
        feedback_text = f"Your backlift angle is {angle:.1f}°.\n"
        if angle < 90:
            feedback_text += "This is quite low. A higher backlift (around 120°) can help generate more power."
        elif is_ideal:
            feedback_text += "This is a good backlift angle, allowing for both control and power."
        else:
            feedback_text += "This is quite high. While it can generate power, it might affect your timing and control."

        return {
            "feedback_no": 0,
            "title": "Backlift Angle Analysis",
            "image_filename": image_filename,
            "feedback_text": feedback_text,
            "ref-images": ["Backlift.png", "BatGrip.png"],
            "is_ideal": is_ideal,
            "angle": angle
        }

    except Exception as e:
        logger.error(f"Error in backlift angle analysis: {e}")
        return {
            "feedback_no": 0,
            "title": "Backlift Angle Analysis",
            "image_filename": "",
            "feedback_text": "An error occurred during backlift angle analysis.",
            "ref-images": ["Backlift.png", "BatGrip.png"],
            "is_ideal": False
        }