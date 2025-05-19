import os
import cv2
import numpy as np
import mediapipe as mp
from typing import Dict, Any, Tuple, List
import uuid
import math
from ...image_utils import crop_and_save_image
import sys
from ...ref_images import ForwardDefence_Frontfoot_Knee_ref_images

def get_mediapipe_landmarks(frame_path: str, pose) -> Dict[str, Any]:
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


def analyze_frontfoot_knee(frame_path: str, pose) -> Tuple[Dict[str, Any], Dict[str, Any], bool]:
    frame_data = get_mediapipe_landmarks(frame_path, pose)
    if not frame_data:
        return None, None, None

    left_knee_x = frame_data['left_leg']['knee'][0]
    right_knee_x = frame_data['right_leg']['knee'][0]
    is_frontfoot_right = right_knee_x > left_knee_x

    frontfoot = frame_data['right_leg'] if is_frontfoot_right else frame_data['left_leg']
    backfoot = frame_data['left_leg'] if is_frontfoot_right else frame_data['right_leg']

    frontfoot_valid = (frontfoot['hip'][0] < frontfoot['ankle'][0] and
                       frontfoot['knee'][0] > frontfoot['ankle'][0] and
                       frontfoot['knee'][0] > frontfoot['hip'][0])

    if not frontfoot_valid:
        return None, None, None

    return frontfoot, frame_data, is_frontfoot_right


def calculate_knee_bend_angle(frontfoot: Dict[str, Any]) -> float:
    hip = np.array(frontfoot['hip'])
    knee = np.array(frontfoot['knee'])
    ankle = np.array(frontfoot['ankle'])

    hip_knee = knee - hip
    knee_ankle = ankle - knee

    dot_product = np.dot(hip_knee, knee_ankle)
    magnitude_product = np.linalg.norm(hip_knee) * np.linalg.norm(knee_ankle)

    if magnitude_product == 0:
        return 180.0

    angle = np.degrees(np.arccos(dot_product / magnitude_product))
    return angle


def create_frontfoot_knee_feedback_image(
        highlights_folder: str,
        frame_file: str,
        frontfoot: Dict[str, Any],
        feedback_images_dir: str,
        is_left_handed: bool = False
) -> str:
    try:
        frame_path = os.path.join(highlights_folder, frame_file)
        ankle_x, ankle_y = frontfoot['ankle']
        knee_x, knee_y = frontfoot['knee']

        knee_ankle_dist = np.sqrt((knee_x - ankle_x) ** 2 + (knee_y - ankle_y) ** 2)
        crop_size = int(2 * knee_ankle_dist)
        crop_size = max(crop_size, 100)

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
    try:
        valid_frontfoot_data = []

        for frame_file in frame_files:
            frame_path = os.path.join(highlights_folder, frame_file)
            frontfoot, frame_data, is_frontfoot_right = analyze_frontfoot_knee(frame_path, pose)

            if frontfoot and frame_data:
                ankle_x = frontfoot['ankle'][0]
                valid_frontfoot_data.append({
                    'frame_file': frame_file,
                    'frontfoot': frontfoot,
                    'frame_data': frame_data,
                    'ankle_x': ankle_x,
                    'is_frontfoot_right': is_frontfoot_right
                })

        if not valid_frontfoot_data:
            return {
                "feedback_no": 3,
                "title": "Frontfoot Knee Position Analysis",
                "image_filename": "",
                "feedback_text": "Player front knee position didn't recognize correctly. Please ensure proper posture for accurate analysis.",
                "ref-images": ForwardDefence_Frontfoot_Knee_ref_images,
                "is_ideal": False
            }

        selected_frame = max(valid_frontfoot_data, key=lambda x: x['ankle_x'])
        frontfoot = selected_frame['frontfoot']

        angle = calculate_knee_bend_angle(frontfoot)
        is_ideal = 40 <= angle <= 120

        feedback_text = "Your front knee should be bent to lower your body and get your head over the ball. "
        feedback_text += "This helps with control and makes it easier to smother any bounce or movement off the pitch. "

        if is_ideal:
            feedback_text += "Your front knee bend is ideal - excellent technique!"
        elif angle > 120:
            feedback_text += "Your knee is bent too much. While some bend is good, too much can affect your balance."
        else:
            feedback_text += "Your knee is too straight. Try bending it more for better control."

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
            "ref-images": ForwardDefence_Frontfoot_Knee_ref_images,
            "is_ideal": is_ideal,
            "angle": angle
        }

    except Exception as e:
        print(f"[ERROR] Failed to process frontfoot knee position: {e}")
        return None
