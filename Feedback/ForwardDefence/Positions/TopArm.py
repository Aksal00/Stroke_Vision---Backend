import os
import cv2
import numpy as np
import mediapipe as mp
from typing import Dict, Any, Tuple, List
from ...image_utils import crop_and_save_image
from ...ref_images import ForwardDefence_TopArm_ref_images


def get_top_arm_landmarks(frame_path: str, pose) -> Dict[str, Any]:
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

        # Determine which arm is the top arm (higher elbow)
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
    shoulder = np.array(top_arm['shoulder'])
    elbow = np.array(top_arm['elbow'])
    wrist = np.array(top_arm['wrist'])

    # Vector from elbow to shoulder
    shoulder_vec = shoulder - elbow
    # Vector from elbow to wrist
    wrist_vec = wrist - elbow

    # Calculate angle between vectors (counter-clockwise from shoulder to wrist)
    angle = np.degrees(np.arctan2(
        shoulder_vec[0] * wrist_vec[1] - shoulder_vec[1] * wrist_vec[0],
        shoulder_vec[0] * wrist_vec[1] + shoulder_vec[1] * wrist_vec[0]
    ))

    # Ensure angle is positive (0-360)
    angle = angle % 360
    # Get the smaller angle (if >180, return 360-angle)
    if angle > 180:
        angle = 360 - angle

    return angle


def create_top_arm_feedback_image(
        highlights_folder: str,
        frame_file: str,
        top_arm: Dict[str, Any],
        feedback_images_dir: str,
        is_left_handed: bool = False
) -> str:
    try:
        # First try to get the original frame with background from for_feedback_output folder
        feedback_output_folder = os.path.join(os.path.dirname(os.path.dirname(highlights_folder)),
                                              "for_feedback_output")
        frame_path = os.path.join(feedback_output_folder, frame_file)

        if not os.path.exists(frame_path):
            # Fallback to the processed frame if original not found
            frame_path = os.path.join(highlights_folder, frame_file)
            print(f"[WARNING] Original frame with background not found, using processed frame: {frame_file}")

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
    Validate if the detected top arm position meets all required conditions.
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

        # 4. For right-handed (top arm is right), elbow.x should be larger than shoulder.x
        #    For left-handed (top arm is left), elbow.x should be smaller than shoulder.x
        if is_top_arm_right:
            if top_arm['elbow'][0] >= top_arm['shoulder'][0]:
                return False
            # 5. Bottom arm (left) wrist.x should be larger than elbow.x
            if bottom_arm['wrist'][0] >= bottom_arm['elbow'][0]:
                return False
        else:
            if top_arm['elbow'][0] <= top_arm['shoulder'][0]:
                return False
            # 5. Bottom arm (right) wrist.x should be smaller than elbow.x
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
                "title": "Top Arm Position Analysis",
                "image_filename": "",
                "feedback_text": "Player top arm position wasn't recognized correctly in any of the frames. Please ensure proper posture for accurate analysis.",
                "ref-images": ForwardDefence_TopArm_ref_images,
                "is_ideal": False,
                "valid_frames": valid_frames
            }

        # Select the frame with the most pronounced top arm position
        selected_frame = max(valid_top_arm_data, key=lambda x: abs(x['landmarks']['top_arm']['elbow'][1] -
                                                                   x['landmarks']['top_arm']['shoulder'][1]))

        top_arm = selected_frame['landmarks']['top_arm']
        angle = calculate_top_arm_angle(top_arm)
        is_ideal = 55 <= angle <= 165

        if is_ideal:
            feedback_text = "Nice work, your top arm bend is technically correct!"
        elif angle < 55:
            feedback_text = "\n\nYour top arm is too bent. This reduces control over the bat face, may cause a crooked bat or an uncontrolled defensive push, and makes it less likely to keep the ball down."
        else:
            feedback_text = "\n\nYour top arm is too straight. This leads to rigid movement, poor timing, limits adjustment to late movement or spin, and provides less shock absorption from the ball impact."

        feedback_text += " Top arm should have a slight bend at the elbow - not too straight, not overly bent. "
        feedback_text += "The top arm should be firm but relaxed, guiding the bat down vertically. "
        feedback_text += "It controls the bat face, ensuring it's angled correctly to defend the ball with a straight bat."

        image_filename = create_top_arm_feedback_image(
            highlights_folder,
            selected_frame['frame_file'],
            selected_frame['landmarks']['top_arm'],
            feedback_images_dir,
            is_left_handed
        )

        return {
            "feedback_no": 7,
            "title": "Top Arm Position Analysis",
            "image_filename": image_filename,
            "feedback_text": feedback_text,
            "ref-images": ForwardDefence_TopArm_ref_images,
            "is_ideal": is_ideal,
            "angle": angle,
            "valid_frames": valid_frames
        }

    except Exception as e:
        print(f"[ERROR] Failed to process top arm position: {e}")
        return None