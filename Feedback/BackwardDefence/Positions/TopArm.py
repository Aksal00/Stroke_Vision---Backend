import os
import cv2
import numpy as np
import mediapipe as mp
from typing import Dict, Any, Tuple, List
from ...image_utils import crop_and_save_image, calculate_anticlockwise_angle
from ...ref_images import BackwardDefence_TopArm_ref_images


def get_top_arm_landmarks(frame_path: str, pose) -> Dict[str, Any]:
    """
    Get MediaPipe landmarks for top arm analysis in backfoot defence
    Returns landmarks for shoulder, elbow, and wrist of both arms
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

        # Determine which arm is the top arm (higher elbow for backfoot defence)
        is_top_arm_right = right_arm['elbow'][1] < left_arm['elbow'][1]
        top_arm = right_arm if is_top_arm_right else left_arm

        return {
            'top_arm': top_arm,
            'is_top_arm_right': is_top_arm_right,
            'left_arm': left_arm,
            'right_arm': right_arm
        }

    except Exception as e:
        print(f"[ERROR] Failed to get top arm landmarks: {e}")
        return None


def calculate_top_arm_angle(top_arm: Dict[str, Any]) -> float:
    """
    Calculate the top arm angle using the new image_utils function
    Returns angle between shoulder-elbow-wrist (anticlockwise)
    """
    shoulder = top_arm['shoulder']
    elbow = top_arm['elbow']
    wrist = top_arm['wrist']

    # Calculate angle using the new utility function
    angle = calculate_anticlockwise_angle(shoulder, elbow, wrist)

    return angle


def create_top_arm_feedback_image(
        highlights_folder: str,
        frame_file: str,
        top_arm: Dict[str, Any],
        feedback_images_dir: str,
        is_left_handed: bool = False
) -> str:
    """
    Create feedback image for top arm analysis centered around the elbow
    """
    try:
        frame_path = os.path.join(highlights_folder, frame_file)
        elbow_x, elbow_y = top_arm['elbow']
        shoulder_x, shoulder_y = top_arm['shoulder']

        elbow_shoulder_dist = np.sqrt((elbow_x - shoulder_x) ** 2 + (elbow_y - shoulder_y) ** 2)
        crop_size = int(2 * elbow_shoulder_dist)
        crop_size = max(crop_size, 100)

        return crop_and_save_image(
            original_image_path=frame_path,
            center_coords=(int(elbow_x), int(elbow_y)),
            crop_size=crop_size,
            output_dir=feedback_images_dir,
            mirror_if_left_handed=is_left_handed
        )

    except Exception as e:
        print(f"[ERROR] Failed to create top arm feedback image: {e}")
        return ""


def is_valid_top_arm_position(landmarks: Dict[str, Any]) -> bool:
    """
    Validate if the detected top arm position meets all required conditions for backfoot defence
    """
    try:
        top_arm = landmarks['top_arm']
        is_top_arm_right = landmarks['is_top_arm_right']
        bottom_arm = landmarks['right_arm'] if not is_top_arm_right else landmarks['left_arm']

        # 1. Check top arm shoulder is higher than bottom arm shoulder
        if top_arm['shoulder'][1] >= bottom_arm['shoulder'][1]:
            return False

        # 2. Check top arm elbow is higher than bottom arm elbow
        if top_arm['elbow'][1] >= bottom_arm['elbow'][1]:
            return False

        # 3. Check top arm wrist is higher than bottom arm wrist
        if top_arm['wrist'][1] >= bottom_arm['wrist'][1]:
            return False

        # 4. For backfoot defence, check arm positioning relative to body
        if is_top_arm_right:
            # Right arm as top arm - elbow should be positioned correctly for backfoot stance
            if top_arm['elbow'][0] >= top_arm['shoulder'][0]:
                return False
            # Bottom arm (left) positioning check
            if bottom_arm['wrist'][0] >= bottom_arm['elbow'][0]:
                return False
        else:
            # Left arm as top arm
            if top_arm['elbow'][0] <= top_arm['shoulder'][0]:
                return False
            # Bottom arm (right) positioning check
            if bottom_arm['wrist'][0] <= bottom_arm['elbow'][0]:
                return False

        return True

    except Exception as e:
        print(f"[ERROR] Failed to validate top arm position: {e}")
        return False


def process_top_arm_position(
        highlights_folder: str,
        frame_files: List[str],
        pose,
        feedback_images_dir: str,
        is_left_handed: bool = False
) -> Dict[str, Any]:
    """
    Main function to process top arm position analysis for backfoot defence
    Returns feedback dictionary with analysis results
    """
    try:
        valid_top_arm_data = []
        valid_frames = []

        for frame_file in frame_files:
            frame_path = os.path.join(highlights_folder, frame_file)
            landmarks = get_top_arm_landmarks(frame_path, pose)

            if landmarks and 'top_arm' in landmarks and is_valid_top_arm_position(landmarks):
                valid_top_arm_data.append({
                    'frame_file': frame_file,
                    'landmarks': landmarks
                })
                valid_frames.append(frame_file)

        if not valid_top_arm_data:
            return {
                "feedback_no": 7,
                "title": "Top Arm Position Analysis (Backfoot Defence)",
                "image_filename": "",
                "feedback_text": "Player top arm position wasn't recognized correctly in any of the frames. Please ensure proper backfoot defence posture for accurate analysis.",
                "ref-images": BackwardDefence_TopArm_ref_images,
                "is_ideal": False,
                "valid_frames": valid_frames
            }

        # Select the frame with the most pronounced top arm position
        selected_frame = max(valid_top_arm_data, key=lambda x: abs(x['landmarks']['top_arm']['elbow'][1] -
                                                                   x['landmarks']['top_arm']['shoulder'][1]))

        top_arm = selected_frame['landmarks']['top_arm']
        angle = calculate_top_arm_angle(top_arm)

        # Adjusted ideal range for backfoot defence (typically more bent than forward defence)
        is_ideal = 45 <= angle <= 135

        if is_ideal:
            feedback_text = "Excellent! Your top arm bend is technically correct for backfoot defence!"
        elif angle < 45:
            feedback_text = "\n\nYour top arm is too bent for backfoot defence. This can restrict your ability to get on top of the ball and may lead to catching practice to the slip cordon. Try to extend it slightly while maintaining control."
        else:
            feedback_text = "\n\nYour top arm is too straight for backfoot defence. This reduces your ability to control the ball and get on top of it. A slight bend provides better control and allows for quick adjustments."

        feedback_text += " In backfoot defence, the top arm should be firm but flexible, allowing you to get on top of the ball. "
        feedback_text += "It should guide the bat down at an angle to smother the ball into the ground. "
        feedback_text += "The top arm controls the bat face and helps you play the ball close to your body with soft hands."

        image_filename = create_top_arm_feedback_image(
            highlights_folder,
            selected_frame['frame_file'],
            selected_frame['landmarks']['top_arm'],
            feedback_images_dir,
            is_left_handed
        )

        return {
            "feedback_no": 7,
            "title": "Top Arm Position Analysis (Backfoot Defence)",
            "image_filename": image_filename,
            "feedback_text": feedback_text,
            "ref-images": BackwardDefence_TopArm_ref_images,
            "is_ideal": is_ideal,
            "angle": angle,
            "valid_frames": valid_frames
        }

    except Exception as e:
        print(f"[ERROR] Failed to process top arm position: {e}")
        return None