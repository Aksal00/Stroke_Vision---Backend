import os
import cv2
import numpy as np
import mediapipe as mp
from typing import Dict, Any, Tuple, List
import uuid
from ...image_utils import crop_and_save_image
from ...ref_images import ForwardDefence_Head_ref_images

def get_mediapipe_head_landmarks(frame_path: str, pose) -> Dict[str, Any]:
    """
    Get MediaPipe landmarks for head position analysis from background-removed image
    Returns landmarks for head, ears, nose, and relevant body parts
    """
    try:
        # Read the background-removed image for calculations
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
    """
    Analyze head position for a single frame using background-removed image
    Returns:
        - head_data (with analysis results)
        - frame_data (raw landmarks)
    """
    frame_data = get_mediapipe_head_landmarks(frame_path, pose)
    if not frame_data:
        return None, None

    # Determine front knee (more right in image)
    left_knee_x = frame_data['left_knee'][0]
    right_knee_x = frame_data['right_knee'][0]
    front_knee_x = max(left_knee_x, right_knee_x)

    head_data = {
        'nose_position': frame_data['nose'],
        'front_knee_x': front_knee_x,
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
    """
    Create feedback image for head position analysis using original frame with background
    """
    try:
        # Get the path to the original frame with background in the for_feedback_output folder
        feedback_output_folder = os.path.join(os.path.dirname(os.path.dirname(highlights_folder)),
                                            "for_feedback_output")
        frame_path = os.path.join(feedback_output_folder, frame_file)

        if not os.path.exists(frame_path):
            # Fallback to the processed frame if original not found
            frame_path = os.path.join(highlights_folder, frame_file)
            print(f"[WARNING] Original frame with background not found, using processed frame: {frame_file}")

        # Calculate distance between shoulder and ankle (using coordinates from analysis)
        shoulder_x, shoulder_y = left_shoulder
        ankle_x, ankle_y = left_ankle
        shoulder_ankle_dist = np.sqrt((shoulder_x - ankle_x) ** 2 + (shoulder_y - ankle_y) ** 2)
        crop_size = int(2 * shoulder_ankle_dist)

        # Ensure minimum crop size
        crop_size = max(crop_size, 200)

        # Use the utility function to crop and save
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
    """
    Main function to process head position analysis
    Uses background-removed images for calculations but original images for feedback
    Returns feedback dictionary with analysis results
    """
    try:
        head_data_list = []

        # Process frames without background for calculations
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
                "feedback_no": 5,
                "title": "Head Position Analysis",
                "image_filename": "",
                "feedback_text": "Player head position didn't recognize correctly. Please ensure proper posture for accurate analysis.",
                "ref-images": ForwardDefence_Head_ref_images,
                "is_ideal": False
            }

        # Select frame with most right-side nose position
        selected_frame = max(head_data_list, key=lambda x: x['nose_x'])
        head_data = selected_frame['head_data']

        # Determine if position is ideal
        nose_x = head_data['nose_position'][0]
        nose_y = head_data['nose_position'][1]
        left_ear_y, right_ear_y = head_data['ears_y']

        is_over_front_knee = nose_x > head_data['front_knee_x']
        is_looking_down = nose_y > right_ear_y
        is_ideal = is_over_front_knee and is_looking_down

        # Generate base feedback
        feedback_text = "Proper head position is crucial for a good forward defence."

        # Check head position relative to front knee
        if is_over_front_knee:
            feedback_text += (" Your head is correctly positioned over your front knee when defending. "
                            "This puts your weight over the ball, helping you: \n"
                            "1. Control the shot better\n"
                            "2. Judge the line and length more accurately\n"
                            "3. Maintain balance\n"
                            "4. Prevent edges\n")
        else:
            feedback_text += (" Your head should come over or slightly ahead of your front knee when defending. "
                            "This helps with:\n"
                            "1. Putting weight over the ball for better control\n"
                            "2. Judging line and length more accurately\n"
                            "3. Maintaining proper balance\n"
                            "4. Reducing chances of edging the ball\n")

        # Check if player is looking down
        if is_looking_down:
            feedback_text += " You appear to be watching the ball closely with your head down. Nice work!"
        else:
            feedback_text += " It seems you're not looking down at the ball. Focus on watching the ball closely until the moment of contact."

        # Create feedback image using original frame with background
        image_filename = create_head_position_feedback_image(
            highlights_folder,
            selected_frame['frame_file'],
            head_data['left_shoulder'],
            head_data['left_ankle'],
            feedback_images_dir,
            is_left_handed
        )

        return {
            "feedback_no": 5,
            "title": "Head Position Analysis",
            "image_filename": image_filename,
            "feedback_text": feedback_text,
            "ref-images": ForwardDefence_Head_ref_images,
            "is_ideal": is_ideal
        }

    except Exception as e:
        print(f"[ERROR] Failed to process head positions: {e}")
        return None