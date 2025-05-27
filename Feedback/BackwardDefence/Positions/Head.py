import os
import cv2
import numpy as np
import mediapipe as mp
from typing import Dict, Any, Tuple, List
import uuid
from ...image_utils import crop_and_save_image
from ...ref_images import BackwardDefence_Head_ref_images  # Updated reference images

def get_mediapipe_head_landmarks(frame_path: str, pose) -> Dict[str, Any]:
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

        head_data = {
            'nose': (landmarks[mp.solutions.pose.PoseLandmark.NOSE.value].x * w,
                     landmarks[mp.solutions.pose.PoseLandmark.NOSE.value].y * h),
            'left_ear': (landmarks[mp.solutions.pose.PoseLandmark.LEFT_EAR.value].x * w,
                         landmarks[mp.solutions.pose.PoseLandmark.LEFT_EAR.value].y * h),
            'right_ear': (landmarks[mp.solutions.pose.PoseLandmark.RIGHT_EAR.value].x * w,
                          landmarks[mp.solutions.pose.PoseLandmark.RIGHT_EAR.value].y * h),
            'left_shoulder': (landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value].x * w,
                              landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value].y * h),
            'left_ankle': (landmarks[mp.solutions.pose.PoseLandmark.LEFT_ANKLE.value].x * w,
                           landmarks[mp.solutions.pose.PoseLandmark.LEFT_ANKLE.value].y * h),
            'left_knee': (landmarks[mp.solutions.pose.PoseLandmark.LEFT_KNEE.value].x * w,
                          landmarks[mp.solutions.pose.PoseLandmark.LEFT_KNEE.value].y * h),
            'right_knee': (landmarks[mp.solutions.pose.PoseLandmark.RIGHT_KNEE.value].x * w,
                           landmarks[mp.solutions.pose.PoseLandmark.RIGHT_KNEE.value].y * h)
        }

        return head_data

    except Exception as e:
        print(f"[ERROR] Failed to get head landmarks: {e}")
        return None


def analyze_head_position(frame_path: str, pose) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    frame_data = get_mediapipe_head_landmarks(frame_path, pose)
    if not frame_data:
        return None, None

    left_knee_x = frame_data['left_knee'][0]
    right_knee_x = frame_data['right_knee'][0]
    back_knee_x = min(left_knee_x, right_knee_x)

    head_data = {
        'nose_position': frame_data['nose'],
        'back_knee_x': back_knee_x,
        'ears_y': (frame_data['left_ear'][1], frame_data['right_ear'][1]),
        'left_shoulder': frame_data['left_shoulder'],
        'left_ankle': frame_data['left_ankle']
    }

    return head_data, frame_data


def create_head_position_feedback_image(
        highlights_folder: str,
        frame_file: str,
        left_shoulder: Tuple[float, float],
        left_ankle: Tuple[float, float],
        feedback_images_dir: str,
        is_left_handed: bool = False
) -> str:
    try:
        frame_path = os.path.join(highlights_folder, frame_file)

        shoulder_x, shoulder_y = left_shoulder
        ankle_x, ankle_y = left_ankle
        shoulder_ankle_dist = np.sqrt((shoulder_x - ankle_x) ** 2 + (shoulder_y - ankle_y) ** 2)
        crop_size = max(int(2 * shoulder_ankle_dist), 200)

        return crop_and_save_image(
            original_image_path=frame_path,
            center_coords=(int(shoulder_x), int(shoulder_y)),
            crop_size=crop_size,
            output_dir=feedback_images_dir,
            mirror_if_left_handed=is_left_handed
        )

    except Exception as e:
        print(f"[ERROR] Failed to create feedback image: {e}")
        return ""


def process_head_position(
        highlights_folder: str,
        frame_files: List[str],
        pose,
        feedback_images_dir: str,
        is_left_handed: bool = False
) -> Dict[str, Any]:
    try:
        head_data_list = []

        for frame_file in frame_files:
            frame_path = os.path.join(highlights_folder, frame_file)
            head_data, frame_data = analyze_head_position(frame_path, pose)

            if head_data and frame_data:
                head_data_list.append({
                    'frame_file': frame_file,
                    'head_data': head_data,
                    'frame_data': frame_data,
                    'nose_x': head_data['nose_position'][0]
                })

        if not head_data_list:
            return {
                "feedback_no": 6,
                "title": "Backfoot Defence Head Position",
                "image_filename": "",
                "feedback_text": "Could not detect proper head position in the backfoot defence stance. Please ensure you're captured clearly during the stroke.",
                "ref-images": BackwardDefence_Head_ref_images,
                "is_ideal": False
            }

        selected_frame = min(head_data_list, key=lambda x: x['nose_x'])  # more backward = lower x
        head_data = selected_frame['head_data']

        nose_x = head_data['nose_position'][0]
        nose_y = head_data['nose_position'][1]
        left_ear_y, right_ear_y = head_data['ears_y']

        is_behind_back_knee = nose_x < head_data['back_knee_x']
        is_looking_down = nose_y > right_ear_y
        is_ideal = is_behind_back_knee and is_looking_down

        feedback_text = "Maintaining correct head position is essential when playing a backfoot defence shot."

        if is_behind_back_knee:
            feedback_text += (" Your head is well aligned behind your back knee. This helps maintain balance "
                              "and improves your ability to handle bounce and late movement.\n")
        else:
            feedback_text += (" Try to keep your head slightly behind the back knee while defending on the backfoot. "
                              "This ensures better control, stability, and shot execution.\n")

        if is_looking_down:
            feedback_text += " You're keeping your eyes on the ball well. Good job!"
        else:
            feedback_text += " You may not be watching the ball till the end. Focus on tracking it all the way."

        image_filename = create_head_position_feedback_image(
            highlights_folder,
            selected_frame['frame_file'],
            head_data['left_shoulder'],
            head_data['left_ankle'],
            feedback_images_dir,
            is_left_handed
        )

        return {
            "feedback_no": 6,
            "title": "Backfoot Defence Head Position",
            "image_filename": image_filename,
            "feedback_text": feedback_text,
            "ref-images": ["BackfootHead.png", "BackfootStance.png"],
            "is_ideal": is_ideal
        }

    except Exception as e:
        print(f"[ERROR] Failed to process backfoot head position: {e}")
        return None
