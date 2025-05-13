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

        # Define foot and leg landmarks
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


def analyze_backfoot_knee(frame_path: str, pose) -> Tuple[Dict[str, Any], Dict[str, Any], bool]:
    """
    Analyze backfoot knee position for a single frame
    Returns:
        - backfoot data
        - frame data
        - is_backfoot_left (True if left foot is backfoot)
    """
    frame_data = get_mediapipe_landmarks(frame_path, pose)
    if not frame_data:
        return None, None, None

    # Determine which foot is backfoot (left side in image)
    left_foot_heel_x = frame_data['left_leg']['heel'][0]
    right_foot_heel_x = frame_data['right_leg']['heel'][0]
    is_left_backfoot = left_foot_heel_x < right_foot_heel_x

    backfoot = frame_data['left_leg'] if is_left_backfoot else frame_data['right_leg']
    frontfoot = frame_data['right_leg'] if is_left_backfoot else frame_data['left_leg']

    # Check landmark validity
    backfoot_valid = backfoot['knee'][0] < backfoot['hip'][0]
    frontfoot_valid = frontfoot['knee'][0] > frontfoot['hip'][0]

    if not (backfoot_valid and frontfoot_valid):
        return None, None, None

    return backfoot, frame_data, is_left_backfoot


def create_backfoot_knee_feedback_image(highlights_folder: str, frame_file: str,
                                        backfoot: Dict[str, Any], feedback_images_dir: str) -> str:
    """
    Create feedback image for backfoot knee analysis and return its filename
    Ensures the cropped image is perfectly centered around the knee landmark
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

        # Calculate distance between knee and ankle
        knee_ankle_dist = np.sqrt((knee_x - ankle_x) ** 2 + (knee_y - ankle_y) ** 2)
        crop_size = int(2 * knee_ankle_dist)

        # Ensure crop_size is at least 100px to avoid too small crops
        crop_size = max(crop_size, 100)

        # Calculate exact center coordinates
        center_x = int(knee_x)
        center_y = int(knee_y)

        # Calculate crop boundaries ensuring we stay within image dimensions
        half_size = crop_size // 2

        # Adjust boundaries if they go beyond image dimensions
        x1 = max(0, center_x - half_size)
        y1 = max(0, center_y - half_size)
        x2 = min(w, center_x + half_size)
        y2 = min(h, center_y + half_size)

        # If we're at the edge of the image, adjust the other side to maintain crop size
        if x2 - x1 < crop_size:
            if x1 == 0:  # At left edge
                x2 = min(w, x1 + crop_size)
            else:  # At right edge
                x1 = max(0, x2 - crop_size)

        if y2 - y1 < crop_size:
            if y1 == 0:  # At top edge
                y2 = min(h, y1 + crop_size)
            else:  # At bottom edge
                y1 = max(0, y2 - crop_size)

        # Crop the original image
        cropped_img = img[y1:y2, x1:x2]

        # Generate unique filename and save
        unique_id = uuid.uuid4().hex
        feedback_filename = f"backfoot_knee_feedback_{unique_id}.jpg"
        feedback_path = os.path.join(feedback_images_dir, feedback_filename)
        cv2.imwrite(feedback_path, cropped_img)

        return feedback_filename

    except Exception as e:
        print(f"[ERROR] Failed to create feedback image: {e}")
        return ""


def process_backfoot_knee_position(highlights_folder: str, frame_files: List[str], pose, feedback_images_dir: str) -> \
Dict[str, Any]:
    """
    Main function to process backfoot knee position analysis
    Returns feedback dictionary with analysis results
    """
    try:
        valid_backfoot_data = []

        for frame_file in frame_files:
            frame_path = os.path.join(highlights_folder, frame_file)
            backfoot, frame_data, is_left_backfoot = analyze_backfoot_knee(frame_path, pose)

            if backfoot and frame_data:
                # Get the hip x coordinate (same side as backfoot)
                hip_x = backfoot['hip'][0]
                valid_backfoot_data.append({
                    'frame_file': frame_file,
                    'backfoot': backfoot,
                    'frame_data': frame_data,
                    'hip_x': hip_x,
                    'is_left_backfoot': is_left_backfoot
                })

        if not valid_backfoot_data:
            return {
                "feedback_no": 2,
                "title": "Backfoot Knee Position Analysis",
                "image_filename": "Couldn't Process",
                "feedback_text": "Player knee position didn't recognize correctly. Please ensure proper posture for accurate analysis."
            }

        # Select frame where hip has highest x coordinate (most forward position)
        selected_frame = max(valid_backfoot_data, key=lambda x: x['hip_x'])

        # Extract coordinates
        backfoot = selected_frame['backfoot']
        hip_x = backfoot['hip'][0]
        knee_x = backfoot['knee'][0]
        ankle_x = backfoot['ankle'][0]

        # Calculate distances
        hip_knee_diff = hip_x - knee_x
        knee_ankle_diff = knee_x - ankle_x

        # Generate base feedback
        feedback_text = "Your back knee should be slightly bent, not locked straight. This gives you balance and helps transfer your weight smoothly to the front foot."

        # Add specific feedback based on knee position
        if hip_knee_diff < knee_ankle_diff / 3:
            feedback_text += " It seems like your knee has bent too much."
        elif hip_knee_diff >= knee_ankle_diff / 3 and hip_knee_diff < knee_ankle_diff:
            feedback_text += " It seems like your backfoot knee is technically okay. Keep it up!"
        else:
            feedback_text += " It seems like your backfoot knee is locked straight."

        # Create feedback image
        image_filename = create_backfoot_knee_feedback_image(
            highlights_folder,
            selected_frame['frame_file'],
            selected_frame['backfoot'],
            feedback_images_dir
        )

        return {
            "feedback_no": 2,
            "title": "Backfoot Knee Position Analysis",
            "image_filename": image_filename,
            "feedback_text": feedback_text
        }

    except Exception as e:
        print(f"[ERROR] Failed to process backfoot knee position: {e}")
        return None