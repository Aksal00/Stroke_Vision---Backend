import cv2
import numpy as np
import os
import uuid
from typing import Tuple, List
import mediapipe as mp


def crop_and_save_image(
        original_image_path: str,
        center_coords: Tuple[int, int],
        crop_size: int,
        output_dir: str,
        mirror_if_left_handed: bool = False
) -> str:
    """
    Draw MediaPipe body landmarks and lines on image, then crop around center coordinates and save it.

    Args:
        original_image_path: Path to the original image
        center_coords: (x, y) coordinates of the center point
        crop_size: Size of the square crop area
        output_dir: Directory to save the cropped image
        mirror_if_left_handed: Whether to mirror the image for left-handed players

    Returns:
        Filename of the saved cropped image with landmarks
    """
    try:
        # Initialize MediaPipe Pose
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
        mp_drawing = mp.solutions.drawing_utils

        # Read the image
        img = cv2.imread(original_image_path)
        if img is None:
            return ""

        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Process the image and detect pose landmarks
        results = pose.process(img_rgb)

        # Draw landmarks and connections on the image
        if results.pose_landmarks:
            # Draw landmarks with custom styling
            mp_drawing.draw_landmarks(
                img,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(
                    color=(0, 255, 0), thickness=3, circle_radius=3),
                connection_drawing_spec=mp_drawing.DrawingSpec(
                    color=(255, 0, 0), thickness=2)
            )

        # Get image dimensions
        h, w = img.shape[:2]

        # Calculate exact center coordinates
        center_x, center_y = center_coords

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

        # Crop the image with landmarks
        cropped_img = img[y1:y2, x1:x2]

        # Mirror if needed for left-handed players
        if mirror_if_left_handed:
            cropped_img = cv2.flip(cropped_img, 1)

        # Generate unique filename and save
        os.makedirs(output_dir, exist_ok=True)
        unique_id = uuid.uuid4().hex
        feedback_filename = f"cropped_feedback_{unique_id}.jpg"
        feedback_path = os.path.join(output_dir, feedback_filename)
        cv2.imwrite(feedback_path, cropped_img)

        return feedback_filename

    except Exception as e:
        print(f"[ERROR] Failed to create feedback image with landmarks: {e}")
        return ""
    finally:
        # Release MediaPipe resources
        if 'pose' in locals():
            pose.close()


def calculate_anticlockwise_angle(
        point_a: Tuple[float, float],
        point_b: Tuple[float, float],
        point_c: Tuple[float, float]
) -> float:
    """
    Calculate the anticlockwise angle between three points (A-B-C).

    Args:
        point_a: First point (x, y)
        point_b: Center point (x, y)
        point_c: Third point (x, y)

    Returns:
        Angle in degrees (0-360) measured anticlockwise from BA to BC
    """
    # Convert points to numpy arrays
    a = np.array(point_a)
    b = np.array(point_b)
    c = np.array(point_c)

    # Calculate vectors
    ba = a - b
    bc = c - b

    # Calculate dot product and magnitudes
    dot_product = np.dot(ba, bc)
    magnitude_ba = np.linalg.norm(ba)
    magnitude_bc = np.linalg.norm(bc)

    # Calculate cosine of the angle
    cosine_angle = dot_product / (magnitude_ba * magnitude_bc)

    # Handle floating point precision issues
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)

    # Calculate angle in radians
    angle_rad = np.arccos(cosine_angle)

    # Calculate cross product to determine direction
    cross_product = np.cross(ba, bc)

    # Convert to degrees (0-360)
    angle_deg = np.degrees(angle_rad)
    if cross_product < 0:
        angle_deg = 360 - angle_deg

    return angle_deg


def calculate_clockwise_angle(
        point_a: Tuple[float, float],
        point_b: Tuple[float, float],
        point_c: Tuple[float, float]
) -> float:
    """
    Calculate the clockwise angle between three points (A-B-C).

    Args:
        point_a: First point (x, y)
        point_b: Center point (x, y)
        point_c: Third point (x, y)

    Returns:
        Angle in degrees (0-360) measured clockwise from BA to BC
    """
    # Calculate anticlockwise angle first
    angle = calculate_anticlockwise_angle(point_a, point_b, point_c)

    # Convert to clockwise angle
    clockwise_angle = 360 - angle if angle != 0 else 0

    return clockwise_angle


def calculate_distance(
        point_a: Tuple[float, float],
        point_b: Tuple[float, float]
) -> float:
    """
    Calculate the Euclidean distance between two points.

    Args:
        point_a: First point (x, y)
        point_b: Second point (x, y)

    Returns:
        Distance between the two points
    """
    # Convert points to numpy arrays
    a = np.array(point_a)
    b = np.array(point_b)

    # Calculate Euclidean distance
    distance = np.linalg.norm(a - b)

    return distance


# Example usage of the new functions
if __name__ == "__main__":
    # Example points
    A = (0, 0)
    B = (1, 0)
    C = (1, 1)

    # Calculate angles
    acw_angle = calculate_anticlockwise_angle(A, B, C)
    cw_angle = calculate_clockwise_angle(A, B, C)
    dist = calculate_distance(A, B)

    print(f"Anticlockwise angle between A-B-C: {acw_angle:.2f} degrees")
    print(f"Clockwise angle between A-B-C: {cw_angle:.2f} degrees")
    print(f"Distance between A and B: {dist:.2f} units")