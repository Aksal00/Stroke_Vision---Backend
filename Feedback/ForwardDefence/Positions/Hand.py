import os
import cv2
import numpy as np
import mediapipe as mp
from typing import Dict, Any, Tuple, List
import uuid


def get_mediapipe_hand_landmarks(frame_path: str, pose) -> Dict[str, Any]:
    """
    Get MediaPipe landmarks for hands from an image frame
    Returns landmarks for hand position analysis
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


def analyze_hand_positions(frame_path: str, pose) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Analyze hand positions for a single frame
    Returns:
        - hand_data (with top/bottom hand info)
        - frame_data (raw landmarks)
    """
    frame_data = get_mediapipe_hand_landmarks(frame_path, pose)
    if not frame_data:
        return None, None

    # Determine which hand is top hand (higher elbow)
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

    return hand_data, frame_data


def create_hand_position_feedback_image(highlights_folder: str, frame_file: str,
                                        bottom_hand: Dict[str, Any], feedback_images_dir: str) -> str:
    """
    Create feedback image for hand position analysis
    """
    try:
        frame_path = os.path.join(highlights_folder, frame_file)
        img = cv2.imread(frame_path)
        if img is None:
            return ""

        h, w = img.shape[:2]

        wrist_x, wrist_y = bottom_hand['wrist']
        elbow_x, elbow_y = bottom_hand['elbow']

        # Calculate distance between wrist and elbow
        wrist_elbow_dist = np.sqrt((wrist_x - elbow_x) ** 2 + (wrist_y - elbow_y) ** 2)
        crop_size = int(2 * wrist_elbow_dist)

        # Ensure minimum crop size
        crop_size = max(crop_size, 150)

        # Calculate crop coordinates centered on wrist
        x1 = max(0, int(wrist_x - crop_size / 2))
        y1 = max(0, int(wrist_y - crop_size / 2))
        x2 = min(w, int(wrist_x + crop_size / 2))
        y2 = min(h, int(wrist_y + crop_size / 2))

        # Adjust if near edges
        if x2 - x1 < crop_size:
            if x1 == 0:
                x2 = min(w, x1 + crop_size)
            else:
                x1 = max(0, x2 - crop_size)

        if y2 - y1 < crop_size:
            if y1 == 0:
                y2 = min(h, y1 + crop_size)
            else:
                y1 = max(0, y2 - crop_size)

        cropped_img = img[y1:y2, x1:x2]

        unique_id = uuid.uuid4().hex
        feedback_filename = f"hand_position_{unique_id}.jpg"
        feedback_path = os.path.join(feedback_images_dir, feedback_filename)
        cv2.imwrite(feedback_path, cropped_img)

        return feedback_filename

    except Exception as e:
        print(f"[ERROR] Failed to create hand position feedback image: {e}")
        return ""


def process_hand_positions(highlights_folder: str, frame_files: List[str], pose, feedback_images_dir: str) -> Dict[
    str, Any]:
    """
    Main function to process hand position analysis
    Returns feedback dictionary with analysis results
    """
    try:
        valid_hand_data = []

        for frame_file in frame_files:
            frame_path = os.path.join(highlights_folder, frame_file)
            hand_data, frame_data = analyze_hand_positions(frame_path, pose)

            if hand_data and frame_data:
                valid_hand_data.append({
                    'frame_file': frame_file,
                    'hand_data': hand_data,
                    'frame_data': frame_data
                })

        if not valid_hand_data:
            return None

        # Use the first valid frame (or could implement selection logic)
        selected_frame = valid_hand_data[0]
        hand_data = selected_frame['hand_data']

        # Generate base feedback
        feedback_text = (
            "When playing the forward defence, the top hand (Left Hand for Right-Handers) controls the bat. "
            "It should be firm but not rigid, ensuring the bat face stays straight and presents a full face to the ball. "
            "The bottom hand (Right Hand for Right-Handers) is relaxed and acts as a guide. "
            "Too much pressure with the bottom hand can result in hard hands and the ball flying off the bat, "
            "increasing the risk of being caught.")

        # Check hand positions
        top_wrist_y = hand_data['top_hand']['wrist'][1]
        bottom_wrist_y = hand_data['bottom_hand']['wrist'][1]

        if top_wrist_y < bottom_wrist_y:
            feedback_text += " It seems like your hands are in correct positions on the bat."
        else:
            feedback_text += " It seems like something is wrong with your hand gripping positions. Please check your hand grip."

        # Check bat angle (wrist x positions)
        top_wrist_x = hand_data['top_hand']['wrist'][0]
        bottom_wrist_x = hand_data['bottom_hand']['wrist'][0]

        if top_wrist_x > bottom_wrist_x:
            feedback_text += " You are correctly angling down the bat which will help avoid getting caught."
        else:
            feedback_text += " When playing forward defence, your bat should angle down to reduce chances of being caught by near fielders. It seems your bat is not angling down."

        # Create feedback image
        image_filename = create_hand_position_feedback_image(
            highlights_folder,
            selected_frame['frame_file'],
            hand_data['bottom_hand'],
            feedback_images_dir
        )

        return {
            "feedback_no": 4,  # Assuming foot analyses are 1 and 2
            "title": "Hand Position Analysis",
            "image_filename": image_filename,
            "feedback_text": feedback_text
        }

    except Exception as e:
        print(f"[ERROR] Failed to process hand positions: {e}")
        return None