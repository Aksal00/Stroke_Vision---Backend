#highlight_generator.py

import os
import cv2
from tkinter import messagebox
import shot_classifier
import mediapipe as mp
import pandas as pd


def export_highlight_clips(video_path, original_frames, peaks_info, fps, output_folder):
    """
    Export short video clip (1.8 seconds) centered around the highest peak movement point.
    The clip contains 0.9 seconds before and 0.9 seconds after the peak.
    Also export frames at specific times (1.05, 1.2, 1.35 seconds) from the clip.

    Args:
        video_path (str): Path to the original video file.
        original_frames (list): List of original video frames.
        peaks_info (DataFrame): DataFrame with information about movement peaks.
        fps (float): Frames per second of the video.
        output_folder (str): Directory where highlight clips will be saved.
    """
    # Create a highlights subfolder
    highlights_folder = os.path.join(output_folder, "movement_highlights")
    os.makedirs(highlights_folder, exist_ok=True)

    # Find the peak with highest combined displacement
    if peaks_info.empty:
        messagebox.showinfo("Info", "No significant movements detected for highlight clips.")
        return

    # Get right elbow and shoulder landmarks
    right_elbow_idx = mp.solutions.pose.PoseLandmark.RIGHT_ELBOW.value
    right_shoulder_idx = mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value
    right_wrist_idx = mp.solutions.pose.PoseLandmark.RIGHT_WRIST.value

    # We need the full landmark data to check positions
    # Load the full landmark data from CSV
    landmark_csv_path = os.path.join(output_folder, 'landmark_data.csv')
    if not os.path.exists(landmark_csv_path):
        messagebox.showerror("Error", "Landmark data file not found. Cannot analyze positions.")
        return

    full_landmark_data = pd.read_csv(landmark_csv_path)

    # Track right wrist x coordinate throughout video to find max x time
    max_wrist_x = -1
    max_wrist_x_frame = 0
    max_wrist_x_time = 0
    for _, row in full_landmark_data.iterrows():
        wrist_x = row[f'landmark_{right_wrist_idx}_x']
        if wrist_x > max_wrist_x:
            max_wrist_x = wrist_x
            max_wrist_x_time = row['time']
            max_wrist_x_frame = row['frame']

    # Filter peaks based on conditions
    valid_peaks = []
    fallback_peaks = []  # Peaks where elbow is left of shoulder but before max wrist x time
    exclusion_reasons = {}

    for idx, peak in peaks_info.iterrows():
        frame_idx = int(peak['frame'])
        time = peak['time']

        # Get the full landmark data for this frame
        frame_data = full_landmark_data[full_landmark_data['frame'] == frame_idx].iloc[0]

        # Condition 1: Right elbow x is more left than right shoulder x
        elbow_x = frame_data[f'landmark_{right_elbow_idx}_x']
        shoulder_x = frame_data[f'landmark_{right_shoulder_idx}_x']
        condition1 = elbow_x < shoulder_x

        # Condition 2: Peak happens after max wrist x time
        condition2 = time > max_wrist_x_time

        if condition1 and not condition2:
            fallback_peaks.append((idx, peak['total_movement'], time))
        elif condition1:
            exclusion_reasons[
                idx] = f"Peak at {time:.2f}s excluded: Right elbow ({elbow_x:.2f}) is left of right shoulder ({shoulder_x:.2f})"
        elif condition2:
            exclusion_reasons[
                idx] = f"Peak at {time:.2f}s excluded: Occurs after max wrist x time ({max_wrist_x_time:.2f}s)"
        else:
            valid_peaks.append((idx, peak['total_movement']))

    # Sort valid peaks by movement (descending)
    valid_peaks.sort(key=lambda x: x[1], reverse=True)

    # If no valid peaks, try fallback logic
    if not valid_peaks:
        if fallback_peaks:
            # Sort fallback peaks by time (descending to get peaks closest to max wrist x time)
            fallback_peaks.sort(key=lambda x: x[2], reverse=True)

            # Get the peak closest to max wrist x time (but before it)
            fallback_idx, fallback_movement, fallback_time = fallback_peaks[0]

            # Calculate middle x position between max wrist x and this peak's wrist x
            fallback_peak_data = \
            full_landmark_data[full_landmark_data['frame'] == int(peaks_info.loc[fallback_idx]['frame'])].iloc[0]
            fallback_wrist_x = fallback_peak_data[f'landmark_{right_wrist_idx}_x']
            middle_x = (max_wrist_x + fallback_wrist_x) / 2

            # Find the nearest frame where wrist x is >= middle_x and before max wrist x time
            candidate_frames = full_landmark_data[
                (full_landmark_data[f'landmark_{right_wrist_idx}_x'] >= middle_x) &
                (full_landmark_data['time'] <= max_wrist_x_time)
                ]

            if not candidate_frames.empty:
                # Find the frame closest to the max wrist x time
                candidate_frames = candidate_frames.sort_values('time', ascending=False)
                selected_frame_data = candidate_frames.iloc[0]

                # Create a pseudo peak entry
                highest_peak = {
                    'frame': selected_frame_data['frame'],
                    'time': selected_frame_data['time'],
                    'total_movement': fallback_movement
                }

                messagebox.showinfo("Info",
                                    f"No ideal peaks found. Using fallback selection:\n"
                                    f"Selected frame at {selected_frame_data['time']:.2f}s where wrist x ({selected_frame_data[f'landmark_{right_wrist_idx}_x']:.2f}) "
                                    f"is between peak wrist x ({fallback_wrist_x:.2f}) and max wrist x ({max_wrist_x:.2f})")
            else:
                messagebox.showinfo("Info", "No motion identified that meets even fallback criteria.")
                return
        else:
            reasons = "\n".join(exclusion_reasons.values())
            messagebox.showinfo("Info",
                                f"No valid peaks found after applying selection criteria:\n\n{reasons}")
            return
    else:
        # Select the highest valid peak
        highest_peak_idx, _ = valid_peaks[0]
        highest_peak = peaks_info.loc[highest_peak_idx]

    peak_frame_data = full_landmark_data[full_landmark_data['frame'] == int(highest_peak['frame'])].iloc[0]

    # Calculate frame window (0.9 seconds before and after)
    frames_window = int(0.9 * fps)

    # Set up video properties
    height, width = original_frames[0].shape[:2] if original_frames else (0, 0)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    peak_frame = int(highest_peak['frame'])
    peak_time = highest_peak['time']

    # Calculate start and end frames
    start_frame = max(0, peak_frame - frames_window)
    end_frame = min(len(original_frames) - 1, peak_frame + frames_window)

    # Only proceed if we have enough frames to make a meaningful clip
    if end_frame - start_frame > fps / 2:  # At least half a second of footage
        # Create output path for this clip
        clip_filename = f"highest_peak_at_{peak_time:.2f}sec.mp4"
        clip_path = os.path.join(highlights_folder, clip_filename)

        # Create VideoWriter
        clip_writer = cv2.VideoWriter(clip_path, fourcc, fps, (width, height))

        # Prepare to capture frames at specific times
        capture_times = [0.95, 1.05, 1.15]  # Times in seconds from start of clip
        frames_to_capture = {}

        # Calculate the absolute frame numbers for these times
        for t in capture_times:
            frame_num = start_frame + int(t * fps)
            if frame_num <= end_frame:
                frames_to_capture[t] = frame_num

        # Write frames
        for frame_idx in range(start_frame, end_frame + 1):
            if frame_idx < len(original_frames):
                frame = original_frames[frame_idx].copy()

                # Add information to the frame
                if frame_idx == peak_frame:
                    # Highlight the peak frame with a red border
                    frame = cv2.rectangle(frame, (0, 0), (width - 1, height - 1), (0, 0, 255), 5)

                # Add timing information
                relative_time = (frame_idx - peak_frame) / fps
                time_text = f"T{relative_time:+.2f}s from peak"
                cv2.putText(frame, time_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.putText(frame, f"Peak at {peak_time:.2f}s", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

                if valid_peaks:
                    cv2.putText(frame, "Highest Valid Movement Peak", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 255, 255), 2)
                else:
                    cv2.putText(frame, "Fallback Selected Frame", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255),
                                2)

                # Add selection criteria info
                cv2.putText(frame, f"Max wrist x time: {max_wrist_x_time:.2f}s", (20, 160),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                # Get current elbow and shoulder positions
                if frame_idx < len(full_landmark_data):
                    current_data = full_landmark_data.iloc[frame_idx]
                    elbow_x = current_data[f'landmark_{right_elbow_idx}_x']
                    shoulder_x = current_data[f'landmark_{right_shoulder_idx}_x']
                    cv2.putText(frame, f"Elbow x: {elbow_x:.2f}, Shoulder x: {shoulder_x:.2f}", (20, 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                clip_writer.write(frame)

                # Check if this is one of our target frames to export
                for t, target_frame in frames_to_capture.items():
                    if frame_idx == target_frame:
                        img_filename = f"frame_at_{t:.2f}sec.jpg"
                        img_path = os.path.join(highlights_folder, img_filename)
                        cv2.imwrite(img_path, frame)
                        print(f"Exported frame at {t:.2f} seconds to {img_path}")

        # Release resources
        clip_writer.release()
        print(f"Created highlight clip: {clip_path}")

        # Create a text file with exclusion reasons if any peaks were excluded
        if exclusion_reasons:
            exclusion_path = os.path.join(highlights_folder, "peak_exclusion_reasons.txt")
            with open(exclusion_path, 'w') as f:
                f.write("Peaks excluded from selection and reasons:\n")
                f.write("========================================\n\n")
                for idx, reason in exclusion_reasons.items():
                    f.write(f"{reason}\n")
                f.write(
                    f"\nSelected peak at {peak_time:.2f}s with movement value: {highest_peak['total_movement']:.4f}\n")

        # Show completion message
        messagebox.showinfo("Highlight Created",
                            f"Created highlight clip for {'highest valid movement peak' if valid_peaks else 'fallback selected frame'} at:\n{peak_time:.2f} seconds\n"
                            f"Saved to:\n{clip_path}\n\n"
                            f"Exported frames at 1.05s, 1.2s, and 1.35s from the clip.\n\n"
                            f"Max wrist x time: {max_wrist_x_time:.2f}s")

        # Run shot classification on the extracted frames
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Model", "image_classifier.h5")
        if os.path.exists(model_path):
            shot_classifier.classify_highlight_frames(highlights_folder, model_path, output_folder)
        else:
            print(f"Warning: Model file not found at {model_path}")
            messagebox.showwarning("Model Not Found",
                                   f"Shot classification model not found at {model_path}.\n"
                                   f"Shot classification will be skipped.")
    else:
        messagebox.showinfo("Info", "Not enough frames to create highlight clip.")