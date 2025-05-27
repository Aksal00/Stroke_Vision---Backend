import os
import cv2
import numpy as np
import mediapipe as mp
from typing import Dict, Any, Tuple, List
import uuid
import math
from ...image_utils import crop_and_save_image, calculate_anticlockwise_angle, calculate_distance
from ...ref_images import BackwardDefence_BottomArm_ref_images


def get_bottom_arm_landmarks(frame_path: str, pose) -> Dict[str, Any]:
    """
    Get MediaPipe landmarks for bottom arm analysis in backfoot defence
    Returns landmarks for shoulder, elbow, and wrist of both arms
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

        # Get coordinates for both arms
        left_arm = {
            'shoulder': (landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value].x * w,
                         landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value].y * h),
            'elbow': (landmarks[mp.solutions.pose.PoseLandmark.LEFT_ELBOW.value].x * w,
                      landmarks[mp.solutions.pose.PoseLandmark.LEFT_ELBOW.value].y * h),
            'wrist': (landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST.value].x * w,
                      landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST.value].y * h)
        }

        right_arm = {
            'shoulder': (landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value].x * w,
                         landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value].y * h),
            'elbow': (landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW.value].x * w,
                      landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW.value].y * h),
            'wrist': (landmarks[mp.solutions.pose.PoseLandmark.RIGHT_WRIST.value].x * w,
                      landmarks[mp.solutions.pose.PoseLandmark.RIGHT_WRIST.value].y * h)
        }

        return {
            'left_arm': left_arm,
            'right_arm': right_arm
        }

    except Exception as e:
        print(f"[ERROR] Failed to get arm landmarks: {e}")
        return None


def calculate_arm_bend_angle(shoulder: Tuple[float, float],
                             elbow: Tuple[float, float],
                             wrist: Tuple[float, float]) -> float:
    """
    Calculate the angle between shoulder-elbow-wrist using the new utility function
    Returns angle in degrees (0-360)
    """
    # Use the new utility function for consistent angle calculation
    angle = calculate_anticlockwise_angle(shoulder, elbow, wrist)
    return angle


def analyze_bottom_arm_bend(frame_path: str, pose) -> Tuple[Dict[str, Any], Dict[str, Any], bool]:
    """
    Analyze bottom arm bend for a single frame in backfoot defence
    Returns:
        - bottom_arm data (shoulder, elbow, wrist)
        - frame data
        - is_bottom_arm_right (True if right arm is bottom arm)
    """
    frame_data = get_bottom_arm_landmarks(frame_path, pose)
    if not frame_data:
        return None, None, None

    # Determine which arm is bottom arm (which elbow is lower in the image - higher y-coordinate)
    left_elbow_y = frame_data['left_arm']['elbow'][1]
    right_elbow_y = frame_data['right_arm']['elbow'][1]
    is_bottom_arm_right = right_elbow_y > left_elbow_y

    bottom_arm = frame_data['right_arm'] if is_bottom_arm_right else frame_data['left_arm']
    top_arm = frame_data['left_arm'] if is_bottom_arm_right else frame_data['right_arm']

    # Check landmark validity
    bottom_arm_valid = all(
        coord is not None for coord in bottom_arm['shoulder'] + bottom_arm['elbow'] + bottom_arm['wrist'])

    if not bottom_arm_valid:
        return None, None, None

    # Calculate distances for validation using the new utility function
    bottom_elbow_wrist_dist = calculate_distance(bottom_arm['elbow'], bottom_arm['wrist'])
    wrist_dist = calculate_distance(frame_data['left_arm']['wrist'], frame_data['right_arm']['wrist'])

    # Validate that bottom arm elbow-wrist distance is reasonable relative to wrist distance
    if bottom_elbow_wrist_dist <= wrist_dist * 0.8:  # Adjusted threshold for backfoot defence
        return None, None, None

    return bottom_arm, frame_data, is_bottom_arm_right


def create_bottom_arm_feedback_image(
        highlights_folder: str,
        frame_file: str,
        bottom_arm: Dict[str, Any],
        feedback_images_dir: str,
        is_left_handed: bool = False
) -> str:
    """
    Create feedback image for bottom arm analysis centered around the elbow
    """
    try:
        frame_path = os.path.join(highlights_folder, frame_file)

        # Get coordinates in pixels
        shoulder_x, shoulder_y = bottom_arm['shoulder']
        elbow_x, elbow_y = bottom_arm['elbow']

        # Calculate distance between elbow and shoulder using utility function
        elbow_shoulder_dist = calculate_distance((elbow_x, elbow_y), (shoulder_x, shoulder_y))
        crop_size = int(2 * elbow_shoulder_dist)
        crop_size = max(crop_size, 100)  # Ensure minimum crop size

        # Use the utility function to crop and save
        return crop_and_save_image(
            original_image_path=frame_path,
            center_coords=(int(elbow_x), int(elbow_y)),
            crop_size=crop_size,
            output_dir=feedback_images_dir,
            mirror_if_left_handed=is_left_handed
        )

    except Exception as e:
        print(f"[ERROR] Failed to create feedback image: {e}")
        return ""


def process_bottom_arm_bend(
        highlights_folder: str,
        frame_files: List[str],
        pose,
        feedback_images_dir: str,
        is_left_handed: bool = False
) -> Dict[str, Any]:
    """
    Main function to process bottom arm bend analysis for backfoot defence
    Returns feedback dictionary with analysis results
    """
    try:
        valid_bottom_arm_data = []

        for frame_file in frame_files:
            frame_path = os.path.join(highlights_folder, frame_file)
            bottom_arm, frame_data, is_bottom_arm_right = analyze_bottom_arm_bend(frame_path, pose)

            if bottom_arm and frame_data:
                angle = calculate_arm_bend_angle(
                    bottom_arm['shoulder'],
                    bottom_arm['elbow'],
                    bottom_arm['wrist']
                )
                valid_bottom_arm_data.append({
                    'frame_file': frame_file,
                    'bottom_arm': bottom_arm,
                    'frame_data': frame_data,
                    'angle': angle,
                    'is_bottom_arm_right': is_bottom_arm_right
                })

        if not valid_bottom_arm_data:
            return {
                "feedback_no": 6,
                "title": "Bottom Arm Bend Analysis (Backfoot Defence)",
                "image_filename": "",
                "feedback_text": "Player's bottom arm position wasn't recognized correctly. Please ensure proper backfoot defence posture for accurate analysis.",
                "ref-images": BackwardDefence_BottomArm_ref_images,
                "is_ideal": False
            }

        # Select frame with most clear bottom arm position (middle of sequence if multiple valid frames)
        selected_frame = None
        is_ideal = False

        # Adjusted ideal range for backfoot defence (typically more bent than forward defence)
        ideal_frames = [f for f in valid_bottom_arm_data if 80 <= f['angle'] <= 140]
        if ideal_frames:
            selected_frame = ideal_frames[len(ideal_frames) // 2]
            is_ideal = True
        else:
            # If no ideal frames, just take the middle of all valid frames
            selected_frame = valid_bottom_arm_data[len(valid_bottom_arm_data) // 2]
            is_ideal = False

        # Extract angle
        angle = selected_frame['angle']

        # Add specific feedback based on angle for backfoot defence
        if angle < 80:
            feedback_text = "Your bottom arm is bending too much for backfoot defence. This can limit your reach and control. Try to extend it slightly while maintaining flexibility.\n"
        elif 80 <= angle <= 140:
            feedback_text = "Your bottom arm bend is in the ideal range for backfoot defence. This provides excellent control and stability.\n"
        else:
            feedback_text = "Your bottom arm is too straight for backfoot defence. This can make it difficult to adjust to the ball's line and length. Try to relax it slightly.\n"

        # Generate base feedback specific to backfoot defence
        feedback_text += "In backfoot defence, the bottom arm should be relaxed but firm, providing a stable base for the shot. " \
                         "A slight bend allows for better control when getting on top of the ball and helps absorb impact. " \
                         "The bottom arm works with the top arm to guide the ball safely into the ground, close to your body. " \
                         "It should be flexible enough to make late adjustments but firm enough to provide control. " \
                         "Proper bottom arm positioning helps ensure soft hands and prevents the ball from popping up to close fielders."

        # Create feedback image
        image_filename = create_bottom_arm_feedback_image(
            highlights_folder,
            selected_frame['frame_file'],
            selected_frame['bottom_arm'],
            feedback_images_dir,
            is_left_handed
        )

        return {
            "feedback_no": 6,
            "title": "Bottom Arm Bend Analysis (Backfoot Defence)",
            "image_filename": image_filename,
            "feedback_text": feedback_text,
            "ref-images": BackwardDefence_BottomArm_ref_images,
            "is_ideal": is_ideal,
            "angle": angle
        }

    except Exception as e:
        print(f"[ERROR] Failed to process bottom arm bend: {e}")
        return None