import os
import cv2
import pandas as pd
import mediapipe as mp
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

# Configure logging
logger = logging.getLogger(__name__)


def export_highlight_clips(
        model_path: str,
        video_path: str,
        original_frames: List[np.ndarray],
        peaks_info: pd.DataFrame,
        fps: float,
        output_folder: str
) -> bool:
    """
    Export highlight video clips around movement peaks

    Args:
        model_path: Path to classification model
        video_path: Original video path
        original_frames: List of video frames
        peaks_info: DataFrame with peak movement info
        fps: Video frames per second
        output_folder: Root output folder

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create highlights folder
        highlights_folder = os.path.join(output_folder, "movement_highlights")
        os.makedirs(highlights_folder, exist_ok=True)

        if peaks_info.empty:
            logger.warning("No significant movements detected")
            return False

        # Get landmark indices
        right_elbow_idx = mp.solutions.pose.PoseLandmark.RIGHT_ELBOW.value
        right_shoulder_idx = mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value
        right_wrist_idx = mp.solutions.pose.PoseLandmark.RIGHT_WRIST.value

        # Load full landmark data
        landmark_csv_path = os.path.join(output_folder, 'landmark_data.csv')
        if not os.path.exists(landmark_csv_path):
            logger.error("Landmark data file not found")
            return False

        full_landmark_data = pd.read_csv(landmark_csv_path)

        # Find max wrist x position
        max_wrist_x = full_landmark_data[f'landmark_{right_wrist_idx}_x'].max()
        max_wrist_x_row = full_landmark_data[
            full_landmark_data[f'landmark_{right_wrist_idx}_x'] == max_wrist_x
            ].iloc[0]
        max_wrist_x_time = max_wrist_x_row['time']

        # Filter peaks based on conditions
        valid_peaks = []
        fallback_peaks = []
        exclusion_reasons = {}

        for idx, peak in peaks_info.iterrows():
            frame_idx = int(peak['frame'])
            frame_data = full_landmark_data[full_landmark_data['frame'] == frame_idx].iloc[0]

            elbow_x = frame_data[f'landmark_{right_elbow_idx}_x']
            shoulder_x = frame_data[f'landmark_{right_shoulder_idx}_x']
            time = peak['time']

            condition1 = elbow_x < shoulder_x  # Elbow left of shoulder
            condition2 = time > max_wrist_x_time  # After max wrist x time

            if condition1 and not condition2:
                fallback_peaks.append((idx, peak['total_movement'], time))
            elif condition1:
                exclusion_reasons[idx] = (
                    f"Peak at {time:.2f}s excluded: Right elbow ({elbow_x:.2f}) "
                    f"is left of right shoulder ({shoulder_x:.2f})"
                )
            elif condition2:
                exclusion_reasons[idx] = (
                    f"Peak at {time:.2f}s excluded: Occurs after max wrist x time "
                    f"({max_wrist_x_time:.2f}s)"
                )
            else:
                valid_peaks.append((idx, peak['total_movement']))

        # Select best peak
        if valid_peaks:
            valid_peaks.sort(key=lambda x: x[1], reverse=True)
            peak_idx, _ = valid_peaks[0]
            highest_peak = peaks_info.loc[peak_idx]
            logger.info("Selected highest valid movement peak")
        elif fallback_peaks:
            fallback_peaks.sort(key=lambda x: x[2], reverse=True)
            fallback_idx, _, _ = fallback_peaks[0]
            highest_peak = peaks_info.loc[fallback_idx]
            logger.info("Selected fallback peak (no valid peaks found)")
        else:
            logger.warning("No valid or fallback peaks available")
            if exclusion_reasons:
                logger.info("Peak exclusion reasons:\n" + "\n".join(exclusion_reasons.values()))
            return False

        # Prepare clip
        frames_window = int(0.9 * fps)
        peak_frame = int(highest_peak['frame'])
        start_frame = max(0, peak_frame - frames_window)
        end_frame = min(len(original_frames) - 1, peak_frame + frames_window)

        if end_frame - start_frame <= fps / 2:  # Need at least 0.5s of footage
            logger.warning("Not enough frames for meaningful highlight clip")
            return False

        # Create clip
        height, width = original_frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        clip_filename = f"highest_peak_at_{highest_peak['time']:.2f}sec.mp4"
        clip_path = os.path.join(highlights_folder, clip_filename)
        clip_writer = cv2.VideoWriter(clip_path, fourcc, fps, (width, height))

        if not clip_writer.isOpened():
            logger.error(f"Could not create video writer for {clip_path}")
            return False

        # Capture frames at specific times (0.95s, 1.05s, 1.15s from start)
        capture_times = [0.95, 1.05, 1.15]
        frames_to_capture = {
            t: start_frame + int(t * fps)
            for t in capture_times
            if start_frame + int(t * fps) <= end_frame
        }

        # Write frames to clip
        for frame_idx in range(start_frame, end_frame + 1):
            if frame_idx >= len(original_frames):
                break

            frame = original_frames[frame_idx].copy()

            # Add annotations
            relative_time = (frame_idx - peak_frame) / fps
            time_text = f"T{relative_time:+.2f}s from peak"
            cv2.putText(frame, time_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(frame, f"Peak at {highest_peak['time']:.2f}s", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 255), 2)

            if frame_idx == peak_frame:
                frame = cv2.rectangle(frame, (0, 0), (width - 1, height - 1), (0, 0, 255), 5)

            clip_writer.write(frame)

            # Save specific frames as images
            for t, target_frame in frames_to_capture.items():
                if frame_idx == target_frame:
                    img_filename = f"frame_at_{t:.2f}sec.jpg"
                    img_path = os.path.join(highlights_folder, img_filename)
                    cv2.imwrite(img_path, frame)

        clip_writer.release()
        logger.info(f"Created highlight clip: {clip_path}")

        # Save exclusion reasons if any
        if exclusion_reasons:
            exclusion_path = os.path.join(highlights_folder, "peak_exclusion_reasons.txt")
            with open(exclusion_path, 'w') as f:
                f.write("Peaks excluded from selection and reasons:\n")
                f.write("\n".join(exclusion_reasons.values()) + "\n\n")
                f.write(f"Selected peak at {highest_peak['time']:.2f}s\n")

        # Run shot classification
        try:
            import shot_classifier
            shot_classifier.classify_highlight_frames(highlights_folder, model_path, output_folder)
        except Exception as e:
            logger.error(f"Shot classification failed: {e}")
            return False

        return True

    except Exception as e:
        logger.error(f"Error generating highlight clip: {e}")
        return False