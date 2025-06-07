import os
import cv2
import numpy as np
import mediapipe as mp
from typing import Dict, Any, Tuple, List
import uuid
import math
from ...image_utils import crop_and_save_image
from ...ref_images import ForwardDefence_BottomArm_ref_images

def get_bottom_arm_landmarks(frame_path: str, pose) -> Dict[str, Any]:
    """
    Get MediaPipe landmarks for bottom arm analysis from background-removed image
    Returns landmarks for shoulder, elbow, and wrist of the bottom arm
    """
    try:
        # Read the image (background removed version)
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

        # Get coordinates for both arms
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

        return {
            'left_arm': left_arm,
            'right_arm': right_arm
        }

    except Exception as e:
        print(f"[ERROR] Failed to get arm landmarks: {e}")
        return None

def calculate_arm_bend_angle(shoulder: Tuple[float, float],
                             elbow: Tuple[float, float],
                             wrist: Tuple[float, float]) -> float:
    """
    Calculate the clockwise angle between shoulder-elbow and elbow-wrist lines
    Returns angle in degrees (0-360)
    """
    # Vector from elbow to shoulder
    vec1 = (shoulder[0] - elbow[0], shoulder[1] - elbow[1])
    # Vector from elbow to wrist
    vec2 = (wrist[0] - elbow[0], wrist[1] - elbow[1])

    # Calculate dot product and magnitudes
    dot_product = vec1[0] * vec2[0] + vec1[1] * vec2[1]
    mag1 = math.sqrt(vec1[0] ** 2 + vec1[1] ** 2)
    mag2 = math.sqrt(vec2[0] ** 2 + vec2[1] ** 2)

    # Calculate angle in radians
    angle_rad = math.acos(dot_product / (mag1 * mag2))

    # Convert to degrees
    angle_deg = math.degrees(angle_rad)

    # Determine if angle is clockwise or counter-clockwise
    cross_product = vec1[0] * vec2[1] - vec1[1] * vec2[0]
    if cross_product < 0:
        angle_deg = 360 - angle_deg

    return angle_deg

def analyze_bottom_arm_bend(frame_path: str, pose) -> Tuple[Dict[str, Any], Dict[str, Any], bool]:
    """
    Analyze bottom arm bend for a single frame using background-removed image
    Returns:
        - bottom_arm data (shoulder, elbow, wrist)
        - frame data
        - is_bottom_arm_right (True if right arm is bottom arm)
    """
    frame_data = get_bottom_arm_landmarks(frame_path, pose)
    if not frame_data:
        return None, None, None

    # Determine which arm is bottom arm (which elbow is lower in the image - higher y-coordinate)
    left_elbow_y = frame_data['left_arm']['elbow'][1]
    right_elbow_y = frame_data['right_arm']['elbow'][1]
    is_bottom_arm_right = right_elbow_y > left_elbow_y

    bottom_arm = frame_data['right_arm'] if is_bottom_arm_right else frame_data['left_arm']
    top_arm = frame_data['left_arm'] if is_bottom_arm_right else frame_data['right_arm']

    # Check landmark validity
    bottom_arm_valid = all(
        coord is not None for coord in bottom_arm['shoulder'] + bottom_arm['elbow'] + bottom_arm['wrist'])

    if not bottom_arm_valid:
        return None, None, None

    # Calculate distances for validation
    bottom_elbow_wrist_dist = math.sqrt(
        (bottom_arm['elbow'][0] - bottom_arm['wrist'][0]) ** 2 +
        (bottom_arm['elbow'][1] - bottom_arm['wrist'][1]) ** 2
    )

    wrist_dist = math.sqrt(
        (frame_data['left_arm']['wrist'][0] - frame_data['right_arm']['wrist'][0]) ** 2 +
        (frame_data['left_arm']['wrist'][1] - frame_data['right_arm']['wrist'][1]) ** 2
    )

    # Validate that bottom arm elbow-wrist distance is larger than inter-wrist distance
    if bottom_elbow_wrist_dist <= wrist_dist:
        return None, None, None

    return bottom_arm, frame_data, is_bottom_arm_right

def create_bottom_arm_feedback_image(
        highlights_folder: str,
        frame_file: str,
        bottom_arm: Dict[str, Any],
        feedback_images_dir: str,
        is_left_handed: bool = False
) -> str:
    """
    Create feedback image for bottom arm analysis centered around the elbow
    Uses the original frame with background from for_feedback_output folder
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

        # Get coordinates in pixels
        shoulder_x, shoulder_y = bottom_arm['shoulder']
        elbow_x, elbow_y = bottom_arm['elbow']

        # Calculate distance between elbow and shoulder
        elbow_shoulder_dist = np.sqrt((elbow_x - shoulder_x) ** 2 + (elbow_y - shoulder_y) ** 2)
        crop_size = int(2 * elbow_shoulder_dist)

        # Use the utility function to crop and save
        return crop_and_save_image(
            original_image_path=frame_path,
            center_coords=(int(elbow_x), int(elbow_y)),
            crop_size=crop_size,
            output_dir=feedback_images_dir,
            mirror_if_left_handed=is_left_handed
        )

    except Exception as e:
        print(f"[ERROR] Failed to create feedback image: {e}")
        return ""

def process_bottom_arm_bend(
        highlights_folder: str,
        frame_files: List[str],
        pose,
        feedback_images_dir: str,
        is_left_handed: bool = False
) -> Dict[str, Any]:
    """
    Main function to process bottom arm bend analysis
    Uses background-removed images for calculations but original images for feedback
    Returns feedback dictionary with analysis results
    """
    try:
        valid_bottom_arm_data = []

        for frame_file in frame_files:
            frame_path = os.path.join(highlights_folder, frame_file)
            bottom_arm, frame_data, is_bottom_arm_right = analyze_bottom_arm_bend(frame_path, pose)

            if bottom_arm and frame_data:
                angle = calculate_arm_bend_angle(
                    bottom_arm['shoulder'],
                    bottom_arm['elbow'],
                    bottom_arm['wrist']
                )
                valid_bottom_arm_data.append({
                    'frame_file': frame_file,
                    'bottom_arm': bottom_arm,
                    'frame_data': frame_data,
                    'angle': angle,
                    'is_bottom_arm_right': is_bottom_arm_right
                })

        if not valid_bottom_arm_data:
            return {
                "feedback_no": 6,
                "title": "Bottom Arm Bend Analysis",
                "image_filename": "",
                "feedback_text": "Player's bottom arm position wasn't recognized correctly. Please ensure proper posture for accurate analysis.",
                "ref-images": ForwardDefence_BottomArm_ref_images,
                "is_ideal": False
            }

        # Select frame with most clear bottom arm position (middle of sequence if multiple valid frames)
        selected_frame = None
        is_ideal = False

        # Try to find a frame with angle between 90-150 degrees first (ideal range)
        ideal_frames = [f for f in valid_bottom_arm_data if 90 <= f['angle'] <= 150]
        if ideal_frames:
            selected_frame = ideal_frames[len(ideal_frames) // 2]
            is_ideal = True
        else:
            # If no ideal frames, just take the middle of all valid frames
            selected_frame = valid_bottom_arm_data[len(valid_bottom_arm_data) // 2]
            is_ideal = False

        # Extract angle
        angle = selected_frame['angle']

        # Add specific feedback based on angle
        if angle < 90:
            feedback_text = "Your bottom arm is bending too much. Try to keep it more relaxed and only slightly bent.\n"
        elif 90 <= angle <= 150:
            feedback_text = "Your bottom arm bend is in the ideal range. This is perfect for controlled forward defense.\n"
        else:
            feedback_text = "Your bottom arm isn't bending enough. Try to relax it slightly for better control.\n"

        # Generate base feedback
        feedback_text += "Bottom arm should be slightly bent and relaxed during forward defense, " \
                         "ready to firm up slightly at the point of contact if needed. " \
                         "A slight bend gives you more control over the bat face, allowing subtle adjustments " \
                         "as the ball approaches. A rigid, fully straight bottom arm can cause the ball to rebound " \
                         "or the bat to jar, while a properly bent arm helps absorb the ball's energy and keeps " \
                         "the shot soft and defensive. Slightly bent arms also promote playing with relaxed hands, " \
                         "which is key to avoiding edges carrying to slip fielders."

        # Create feedback image using original frame with background
        image_filename = create_bottom_arm_feedback_image(
            highlights_folder,
            selected_frame['frame_file'],
            selected_frame['bottom_arm'],
            feedback_images_dir,
            is_left_handed
        )

        return {
            "feedback_no": 6,
            "title": "Bottom Arm Bend Analysis",
            "image_filename": image_filename,
            "feedback_text": feedback_text,
            "ref-images": ForwardDefence_BottomArm_ref_images,
            "is_ideal": is_ideal,
            "angle": angle
        }

    except Exception as e:
        print(f"[ERROR] Failed to process bottom arm bend: {e}")
        return None