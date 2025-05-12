import os
import cv2
import numpy as np
import mediapipe as mp
from typing import Dict, Any, Tuple, List
import uuid


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

        # Define foot landmarks
        left_foot = {
            'heel': (landmarks[mp.solutions.pose.PoseLandmark.LEFT_HEEL.value].x * w,
                     landmarks[mp.solutions.pose.PoseLandmark.LEFT_HEEL.value].y * h),
            'ankle': (landmarks[mp.solutions.pose.PoseLandmark.LEFT_ANKLE.value].x * w,
                      landmarks[mp.solutions.pose.PoseLandmark.LEFT_ANKLE.value].y * h),
            'index': (landmarks[mp.solutions.pose.PoseLandmark.LEFT_FOOT_INDEX.value].x * w,
                      landmarks[mp.solutions.pose.PoseLandmark.LEFT_FOOT_INDEX.value].y * h),
            'knee': (landmarks[mp.solutions.pose.PoseLandmark.LEFT_KNEE.value].x * w,
                     landmarks[mp.solutions.pose.PoseLandmark.LEFT_KNEE.value].y * h)
        }

        right_foot = {
            'heel': (landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HEEL.value].x * w,
                     landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HEEL.value].y * h),
            'ankle': (landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ANKLE.value].x * w,
                      landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ANKLE.value].y * h),
            'index': (landmarks[mp.solutions.pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x * w,
                      landmarks[mp.solutions.pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y * h),
            'knee': (landmarks[mp.solutions.pose.PoseLandmark.RIGHT_KNEE.value].x * w,
                     landmarks[mp.solutions.pose.PoseLandmark.RIGHT_KNEE.value].y * h)
        }

        return {
            'left_foot': left_foot,
            'right_foot': right_foot
        }

    except Exception as e:
        print(f"[ERROR] Failed to get landmarks: {e}")
        return None


def analyze_backfoot_position(frame_path: str, pose) -> Tuple[Dict[str, Any], Dict[str, Any], bool]:
    """
    Analyze backfoot position for a single frame
    Returns:
        - backfoot data
        - frame data
        - heel_lifted status
    """
    frame_data = get_mediapipe_landmarks(frame_path, pose)
    if not frame_data:
        return None, None, None

    # Determine which foot is backfoot (left side in image)
    left_foot_heel_x = frame_data['left_foot']['heel'][0]
    right_foot_heel_x = frame_data['right_foot']['heel'][0]
    is_left_backfoot = left_foot_heel_x < right_foot_heel_x

    backfoot = frame_data['left_foot'] if is_left_backfoot else frame_data['right_foot']

    # Calculate heel and ankle positions
    heel_y = backfoot['heel'][1]
    ankle_y = backfoot['ankle'][1]
    index_y = backfoot['index'][1]

    # Improved logic for checking heel lift
    heel_lifted = (ankle_y - heel_y) > (heel_y - index_y) * 0.5  # Adjusted threshold

    return backfoot, frame_data, heel_lifted


def create_backfoot_feedback_image(highlights_folder: str, frame_file: str,
                                   backfoot: Dict[str, Any], frame_data: Dict[str, Any],
                                   feedback_images_dir: str) -> str:
    """
    Create feedback image for backfoot analysis and return its filename
    """
    try:
        frame_path = os.path.join(highlights_folder, frame_file)
        img = cv2.imread(frame_path)
        if img is None:
            return ""

        # Get image dimensions
        h, w = img.shape[:2]

        # Get coordinates in pixels
        ankle_x, ankle_y = backfoot['ankle']
        knee_x, knee_y = backfoot['knee']

        # Calculate crop parameters
        ankle_to_knee_dist = np.sqrt((ankle_x - knee_x) ** 2 + (ankle_y - knee_y) ** 2)
        crop_size = int(2 * ankle_to_knee_dist)

        # Calculate crop coordinates
        x1 = max(0, int(ankle_x - crop_size / 2))
        y1 = max(0, int(ankle_y - crop_size / 2))
        x2 = min(w, int(ankle_x + crop_size / 2))
        y2 = min(h, int(ankle_y + crop_size / 2))

        # Crop the original image
        cropped_img = img[y1:y2, x1:x2]

        # Generate unique filename and save
        unique_id = uuid.uuid4().hex
        feedback_filename = f"backfoot_feedback_{unique_id}.jpg"
        feedback_path = os.path.join(feedback_images_dir, feedback_filename)
        cv2.imwrite(feedback_path, cropped_img)

        return feedback_filename

    except Exception as e:
        print(f"[ERROR] Failed to create feedback image: {e}")
        return ""


def process_backfoot_position(highlights_folder: str, frame_files: List[str], pose, feedback_images_dir: str) -> Dict[
    str, Any]:
    """
    Main function to process backfoot position analysis
    Returns feedback dictionary with analysis results
    """
    try:
        backfoot_data = []

        for frame_file in frame_files:
            frame_path = os.path.join(highlights_folder, frame_file)
            backfoot, frame_data, heel_lifted = analyze_backfoot_position(frame_path, pose)

            if backfoot and frame_data:
                backfoot_data.append({
                    'frame_file': frame_file,
                    'heel_lifted': heel_lifted,
                    'backfoot': backfoot,
                    'frame_data': frame_data
                })

        if not backfoot_data:
            return None

        # Select frame where heel is most lifted (or least lifted if no lift detected)
        if any(item['heel_lifted'] for item in backfoot_data):
            selected_frame = max(
                [item for item in backfoot_data if item['heel_lifted']],
                key=lambda x: x['backfoot']['ankle'][1] - x['backfoot']['heel'][1]
            )
        else:
            selected_frame = min(
                backfoot_data,
                key=lambda x: x['backfoot']['ankle'][1] - x['backfoot']['heel'][1]
            )

        # Create feedback image
        image_filename = create_backfoot_feedback_image(
            highlights_folder,
            selected_frame['frame_file'],
            selected_frame['backfoot'],
            selected_frame['frame_data'],
            feedback_images_dir
        )

        # Generate feedback text
        feedback_text = "When playing forward defence the backfoot heel should lift to maintain a proper balance."
        if selected_frame['heel_lifted']:
            feedback_text += " When considering your backfoot it seems the heel is lifted. Nice work!"
        else:
            feedback_text += " But when considering your backfoot, it doesn't seem like the heel is lifting enough."

        return {
            "feedback_no": 1,
            "title": "Backfoot Position Analysis",
            "image_filename": image_filename,
            "feedback_text": feedback_text
        }

    except Exception as e:
        print(f"[ERROR] Failed to process backfoot position: {e}")
        return None