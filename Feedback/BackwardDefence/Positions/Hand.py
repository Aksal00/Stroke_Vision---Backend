import os
import cv2
import numpy as np
import mediapipe as mp
from typing import Dict, Any, Tuple, List
import uuid
from ...image_utils import crop_and_save_image
from ...ref_images import BackwardDefence_Hand_ref_images


def get_mediapipe_hand_landmarks(frame_path: str, pose) -> Dict[str, Any]:
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

        hands = {
            'left_wrist': (landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST.value].x * w,
                           landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST.value].y * h),
            'right_wrist': (landmarks[mp.solutions.pose.PoseLandmark.RIGHT_WRIST.value].x * w,
                            landmarks[mp.solutions.pose.PoseLandmark.RIGHT_WRIST.value].y * h),
            'left_elbow': (landmarks[mp.solutions.pose.PoseLandmark.LEFT_ELBOW.value].x * w,
                           landmarks[mp.solutions.pose.PoseLandmark.LEFT_ELBOW.value].y * h),
            'right_elbow': (landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW.value].x * w,
                            landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW.value].y * h)
        }

        return hands

    except Exception as e:
        print(f"[ERROR] Failed to get hand landmarks: {e}")
        return None


def analyze_hand_positions(frame_path: str, pose) -> Tuple[Dict[str, Any], Dict[str, Any], bool]:
    frame_data = get_mediapipe_hand_landmarks(frame_path, pose)
    if not frame_data:
        return None, None, None

    left_elbow_y = frame_data['left_elbow'][1]
    right_elbow_y = frame_data['right_elbow'][1]

    is_left_top_hand = left_elbow_y < right_elbow_y

    hand_data = {
        'top_hand': {
            'wrist': frame_data['left_wrist'] if is_left_top_hand else frame_data['right_wrist'],
            'elbow': frame_data['left_elbow'] if is_left_top_hand else frame_data['right_elbow']
        },
        'bottom_hand': {
            'wrist': frame_data['right_wrist'] if is_left_top_hand else frame_data['left_wrist'],
            'elbow': frame_data['right_elbow'] if is_left_top_hand else frame_data['left_elbow']
        }
    }

    return hand_data, frame_data, is_left_top_hand


def create_hand_position_feedback_image(
        highlights_folder: str,
        frame_file: str,
        bottom_hand: Dict[str, Any],
        feedback_images_dir: str,
        is_left_handed: bool = False
) -> str:
    try:
        frame_path = os.path.join(highlights_folder, frame_file)

        wrist_x, wrist_y = bottom_hand['wrist']
        elbow_x, elbow_y = bottom_hand['elbow']

        wrist_elbow_dist = np.sqrt((wrist_x - elbow_x) ** 2 + (wrist_y - elbow_y) ** 2)
        crop_size = int(2 * wrist_elbow_dist)
        crop_size = max(crop_size, 150)

        return crop_and_save_image(
            original_image_path=frame_path,
            center_coords=(int(wrist_x), int(wrist_y)),
            crop_size=crop_size,
            output_dir=feedback_images_dir,
            mirror_if_left_handed=is_left_handed
        )

    except Exception as e:
        print(f"[ERROR] Failed to create hand position feedback image: {e}")
        return ""


def process_hand_positions(
        highlights_folder: str,
        frame_files: List[str],
        pose,
        feedback_images_dir: str,
        is_left_handed: bool = False
) -> Dict[str, Any]:
    try:
        valid_hand_data = []

        for frame_file in frame_files:
            frame_path = os.path.join(highlights_folder, frame_file)
            hand_data, frame_data, is_left_top_hand = analyze_hand_positions(frame_path, pose)

            if hand_data and frame_data:
                valid_hand_data.append({
                    'frame_file': frame_file,
                    'hand_data': hand_data,
                    'frame_data': frame_data,
                    'is_left_top_hand': is_left_top_hand
                })

        if not valid_hand_data:
            return {
                "feedback_no": 5,
                "title": "Backward Defence - Hand Position Analysis",
                "image_filename": "",
                "feedback_text": "Player hand positions couldn't be recognized. Ensure a proper stance for accurate analysis.",
                "ref-images": BackwardDefence_Hand_ref_images,
                "is_ideal": False
            }

        selected_frame = valid_hand_data[0]
        hand_data = selected_frame['hand_data']

        top_wrist_y = hand_data['top_hand']['wrist'][1]
        bottom_wrist_y = hand_data['bottom_hand']['wrist'][1]
        top_wrist_x = hand_data['top_hand']['wrist'][0]
        bottom_wrist_x = hand_data['bottom_hand']['wrist'][0]

        is_ideal = (top_wrist_y < bottom_wrist_y) and (top_wrist_x > bottom_wrist_x)

        feedback_text = (
            "In backward defence, the top hand (Left Hand for Right-Handers) should control the bat with a firm grip, "
            "while the bottom hand guides without applying too much pressure. "
            "This allows better control and prevents edges from flying off the bat. "
            "A stable bat angle and good hand coordination are essential for defending short-pitched deliveries."
        )

        if top_wrist_y < bottom_wrist_y:
            feedback_text += " Your top and bottom hands are well aligned for a stable grip."
        else:
            feedback_text += " Your hand grip may need improvement — the top hand should be more dominant in backward defence."

        if top_wrist_x > bottom_wrist_x:
            feedback_text += " Your bat is angled correctly, helping in keeping the ball down."
        else:
            feedback_text += " Try angling your bat downward more to reduce the chance of edges or catches."

        image_filename = create_hand_position_feedback_image(
            highlights_folder=highlights_folder,
            frame_file=selected_frame['frame_file'],
            bottom_hand=hand_data['bottom_hand'],
            feedback_images_dir=feedback_images_dir,
            is_left_handed=is_left_handed
        )

        return {
            "feedback_no": 5,
            "title": "Backward Defence - Hand Position Analysis",
            "image_filename": image_filename,
            "feedback_text": feedback_text,
            "ref-images": ["BackwardHandGrip.png", "BackwardBatAngle.png"],
            "is_ideal": is_ideal
        }

    except Exception as e:
        print(f"[ERROR] Failed to process hand positions: {e}")
        return None
