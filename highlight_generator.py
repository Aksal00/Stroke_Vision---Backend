import shutil
import logging
import os
import cv2
import mediapipe as mp
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from scipy.spatial import distance
from rembg import remove
from PIL import Image
from background_remover import process_frame_with_grey_background
from video_analyzer import select_valid_frames

# Configure logging
logger = logging.getLogger(__name__)


def export_highlight_frames(
        video_path: str,
        original_frames: List[np.ndarray],
        fps: float,
        output_folder: str,
        dominant_hand: str
) -> bool:
    """
    Export highlight frames based on wrist movement analysis with background removal and grey background.
    Also stores original frames with background in a separate folder for feedback.

    Args:
        video_path: Original video path
        original_frames: List of video frames
        fps: Video frames per second
        output_folder: Root output folder
        dominant_hand: 'left' or 'right' indicating batsman's dominant hand

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create highlights folder
        highlights_folder = os.path.join(output_folder, "movement_highlights")
        os.makedirs(highlights_folder, exist_ok=True)

        # Create folder for extracted frames
        frames_folder = os.path.join(highlights_folder, "extracted_frames")
        os.makedirs(frames_folder, exist_ok=True)

        # Create folder for feedback output (original frames with background)
        feedback_folder = os.path.join(output_folder, "for_feedback_output")
        os.makedirs(feedback_folder, exist_ok=True)

        # Create temp folder for processing
        temp_folder = os.path.join(highlights_folder, "temp_processing")
        os.makedirs(temp_folder, exist_ok=True)

        # Initialize MediaPipe Pose
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            smooth_segmentation=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Initialize lists to store positions
        left_wrist_normalized = []
        right_wrist_normalized = []
        shoulder_widths_normalized = []
        wrist_distances_normalized = []
        frame_indices = []

        # Process each frame to get normalized wrist positions
        for frame_idx, frame in enumerate(original_frames):
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                # Extract keypoints and flip Y coordinate
                keypoints = np.array([[lm.x, 1 - lm.y, lm.z] for lm in landmarks])

                # Hip-centered normalization
                left_hip = keypoints[23]
                right_hip = keypoints[24]
                mid_hip = (left_hip + right_hip) / 2
                keypoints_normalized = keypoints - mid_hip

                # Scale by torso length
                neck = keypoints[11]
                torso_length = np.linalg.norm(neck - mid_hip)
                if torso_length > 0:
                    keypoints_normalized /= torso_length

                # Get wrist landmarks
                left_wrist_norm = keypoints_normalized[15]
                right_wrist_norm = keypoints_normalized[16]

                # Calculate normalized distances
                shoulder_width_norm = distance.euclidean(
                    keypoints_normalized[11][:2],
                    keypoints_normalized[12][:2])
                wrist_distance_norm = distance.euclidean(
                    left_wrist_norm[:2],
                    right_wrist_norm[:2])

                # Store normalized values
                shoulder_widths_normalized.append(shoulder_width_norm)
                wrist_distances_normalized.append(wrist_distance_norm)
                left_wrist_normalized.append(left_wrist_norm[:2])
                right_wrist_normalized.append(right_wrist_norm[:2])
                frame_indices.append(frame_idx)

        pose.close()

        if not frame_indices:
            logger.warning("No pose landmarks detected in any frame")
            return False

        # Convert to numpy arrays
        left_wrist_normalized = np.array(left_wrist_normalized)
        right_wrist_normalized = np.array(right_wrist_normalized)
        shoulder_widths_normalized = np.array(shoulder_widths_normalized)
        wrist_distances_normalized = np.array(wrist_distances_normalized)
        frame_indices = np.array(frame_indices)

        # Filter positions where wrist distance exceeds max shoulder width
        max_shoulder_width = np.max(shoulder_widths_normalized)
        valid_frames = wrist_distances_normalized <= max_shoulder_width

        if not np.any(valid_frames):
            logger.warning("No valid frames after shoulder width filtering")
            return False

        filtered_left = left_wrist_normalized[valid_frames]
        filtered_right = right_wrist_normalized[valid_frames]
        filtered_frame_indices = frame_indices[valid_frames]

        # Analyze motion direction
        def analyze_direction(positions):
            initial_x = positions[0][0]
            right_distance = 0  # Motion towards positive x (front/right)
            left_distance = 0  # Motion towards negative x (back/left)

            for i in range(1, len(positions)):
                current_x = positions[i][0]
                x_displacement = current_x - initial_x

                if x_displacement > 0:
                    right_distance += abs(x_displacement)
                else:
                    left_distance += abs(x_displacement)

            total_distance = right_distance + left_distance
            if total_distance > 0:
                return right_distance / total_distance
            return 0.5

        left_ratio = analyze_direction(filtered_left)
        right_ratio = analyze_direction(filtered_right)
        overall_ratio = (left_ratio + right_ratio) / 2

        # Determine dominant direction
        if overall_ratio > 0.5:
            dominant_direction = "front"
            wrist_x_avg = (filtered_left[:, 0] + filtered_right[:, 0]) / 2
            top_candidate_indices = np.argsort(wrist_x_avg)[-5:][::-1]  # Get top 5 candidates
        else:
            dominant_direction = "back"
            wrist_x_avg = (filtered_left[:, 0] + filtered_right[:, 0]) / 2
            top_candidate_indices = np.argsort(wrist_x_avg)[:5]  # Get top 5 candidates

        # Select frames with valid pose detection
        highlight_frame_indices, highlight_frames = select_valid_frames(
            original_frames,
            filtered_frame_indices,
            top_candidate_indices,
            num_frames=3
        )

        if len(highlight_frame_indices) < 3:
            logger.warning(f"Only found {len(highlight_frame_indices)} frames with valid pose detection")
            if len(highlight_frame_indices) == 0:
                return False

        # Process and save the highlight frames with background removal
        for i, frame_idx in enumerate(highlight_frame_indices):
            if frame_idx < len(original_frames):
                frame = original_frames[frame_idx].copy()

                # Add annotation
                time_sec = frame_idx / fps
                cv2.putText(frame, f"Time: {time_sec:.2f}s", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.putText(frame, f"Dominant: {dominant_hand}", (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.putText(frame, f"Direction: {dominant_direction}", (20, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

                # Save original frame to temp location
                img_filename = f"highlight_{i + 1}_{dominant_direction}_{dominant_hand}.jpg"
                temp_path = os.path.join(temp_folder, img_filename)
                final_path = os.path.join(frames_folder, img_filename)
                feedback_path = os.path.join(feedback_folder, img_filename)

                # Save original to temp
                cv2.imwrite(temp_path, frame)

                # Save original frame with background to feedback folder
                cv2.imwrite(feedback_path, frame)
                logger.info(f"Saved original frame with background for feedback: {feedback_path}")

                # Process with background removal and grey background
                if process_frame_with_grey_background(temp_path, final_path):
                    logger.info(f"Successfully processed and saved highlight frame: {final_path}")
                else:
                    # Fallback to original if processing fails
                    cv2.imwrite(final_path, frame)
                    logger.warning(f"Used original frame due to processing error: {final_path}")

        # Clean up temp folder
        try:
            shutil.rmtree(temp_folder)
        except Exception as e:
            logger.warning(f"Could not clean up temp folder: {e}")

        # Save motion analysis results
        analysis_path = os.path.join(highlights_folder, "motion_analysis.txt")
        with open(analysis_path, 'w') as f:
            f.write(f"Dominant Hand: {dominant_hand}\n")
            f.write(f"Dominant Direction: {dominant_direction}\n")
            f.write(f"Front Ratio: {overall_ratio:.2f}\n")
            f.write(f"Highlight Frames: {highlight_frame_indices}\n")
            f.write(f"Frame Times: {highlight_frame_indices / fps}\n")

        return True

    except Exception as e:
        logger.error(f"Error generating highlight frames: {e}")
        return False