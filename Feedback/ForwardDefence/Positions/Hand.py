import os
import cv2
import numpy as np
import mediapipe as mp
from typing import Dict, Any, Tuple, List
import uuid
from ...image_utils import crop_and_save_image
from ...ref_images import ForwardDefence_Hand_ref_images
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


def analyze_hand_positions(frame_path: str, pose) -> Tuple[Dict[str, Any], Dict[str, Any], bool]:
    """
    Analyze hand positions for a single frame
    Returns:
        - hand_data (with top/bottom hand info)
        - frame_data (raw landmarks)
        - is_left_top_hand (True if left hand is top hand)
    """
    frame_data = get_mediapipe_hand_landmarks(frame_path, pose)
    if not frame_data:
        return None, None, None

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

    return hand_data, frame_data, is_left_top_hand


def create_hand_position_feedback_image(
        highlights_folder: str,
        frame_file: str,
        bottom_hand: Dict[str, Any],
        feedback_images_dir: str,
        is_left_handed: bool = False
) -> str:
    """
    Create feedback image for hand position analysis and return its filename
    Args:
        highlights_folder: Path to folder containing the frame
        frame_file: Filename of the frame
        bottom_hand: Dictionary containing bottom hand landmarks
        feedback_images_dir: Directory to save feedback images
        is_left_handed: Whether player is left-handed (for mirroring)
    Returns:
        Filename of the saved feedback image
    """
    try:
        frame_path = os.path.join(highlights_folder, frame_file)

        # Get coordinates in pixels
        wrist_x, wrist_y = bottom_hand['wrist']
        elbow_x, elbow_y = bottom_hand['elbow']

        # Calculate distance between wrist and elbow
        wrist_elbow_dist = np.sqrt((wrist_x - elbow_x) ** 2 + (wrist_y - elbow_y) ** 2)
        crop_size = int(2 * wrist_elbow_dist)

        # Ensure crop_size is at least 150px to avoid too small crops
        crop_size = max(crop_size, 150)

        # Use the utility function to crop and save
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
    """
    Main function to process hand position analysis
    Returns feedback dictionary with analysis results
    """
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
                "feedback_no": 4,
                "title": "Hand Position Analysis",
                "image_filename": "",
                "feedback_text": "Player hand positions didn't recognize correctly. Please ensure proper posture for accurate analysis.",
                "ref-images": ForwardDefence_Hand_ref_images,  # Updated to use imported array
                "is_ideal": False
            }

        # Use the first valid frame (or could implement selection logic)
        selected_frame = valid_hand_data[0]
        hand_data = selected_frame['hand_data']

        # Check hand positions
        top_wrist_y = hand_data['top_hand']['wrist'][1]
        bottom_wrist_y = hand_data['bottom_hand']['wrist'][1]
        top_wrist_x = hand_data['top_hand']['wrist'][0]
        bottom_wrist_x = hand_data['bottom_hand']['wrist'][0]

        # Determine if position is ideal
        is_ideal = (top_wrist_y < bottom_wrist_y) and (top_wrist_x > bottom_wrist_x)

        # Generate base feedback
        feedback_text = (
            "When playing the forward defence, the top hand (Left Hand for Right-Handers) controls the bat. "
            "It should be firm but not rigid, ensuring the bat face stays straight and presents a full face to the ball. "
            "The bottom hand (Right Hand for Right-Handers) is relaxed and acts as a guide. "
            "Too much pressure with the bottom hand can result in hard hands and the ball flying off the bat, "
            "increasing the risk of being caught.")

        # Add specific feedback
        if top_wrist_y < bottom_wrist_y:
            feedback_text += " It seems like your hands are in correct positions on the bat."
        else:
            feedback_text += " It seems like something is wrong with your hand gripping positions. Please check your hand grip."

        if top_wrist_x > bottom_wrist_x:
            feedback_text += " You are correctly angling down the bat which will help avoid getting caught."
        else:
            feedback_text += " When playing forward defence, your bat should angle down to reduce chances of being caught by near fielders. It seems your bat is not angling down."

        # Create feedback image
        image_filename = create_hand_position_feedback_image(
            highlights_folder=highlights_folder,
            frame_file=selected_frame['frame_file'],
            bottom_hand=hand_data['bottom_hand'],
            feedback_images_dir=feedback_images_dir,
            is_left_handed=is_left_handed
        )

        return {
            "feedback_no": 4,
            "title": "Hand Position Analysis",
            "image_filename": image_filename,
            "feedback_text": feedback_text,
            "ref-images": ["HandGrip.png", "BatAngle.png"],
            "is_ideal": is_ideal
        }

    except Exception as e:
        print(f"[ERROR] Failed to process hand positions: {e}")
        return None