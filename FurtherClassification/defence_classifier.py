import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from matplotlib import pyplot as plt
import matplotlib

matplotlib.use('Agg')


class DefenceClassifier:
    def __init__(self):
        """Initialize MediaPipe Pose solution"""
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Landmark indices
        self.left_ankle = self.mp_pose.PoseLandmark.LEFT_ANKLE.value
        self.right_ankle = self.mp_pose.PoseLandmark.RIGHT_ANKLE.value

    def classify_defence(self, video_path, output_folder):
        """
        Classify a cricket defence shot into front foot or back foot defence

        Args:
            video_path (str): Path to input video
            output_folder (str): Directory to save results

        Returns:
            str: Classification result
        """
        # Setup output directory
        os.makedirs(output_folder, exist_ok=True)

        # Open video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return "Error: Could not open video file"

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        mid_frame = total_frames // 2

        # Data collection
        frames_data = []

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            time = frame_count / fps

            # Process frame
            results = self.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            frame_data = {
                'frame': frame_count,
                'time': time,
                'ankle_distance': 0
            }

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                # Calculate ankle distance
                left_ankle_pos = np.array([landmarks[self.left_ankle].x, landmarks[self.left_ankle].y])
                right_ankle_pos = np.array([landmarks[self.right_ankle].x, landmarks[self.right_ankle].y])
                frame_data['ankle_distance'] = np.linalg.norm(left_ankle_pos - right_ankle_pos)

            frames_data.append(frame_data)

        # Release resources
        cap.release()
        self.pose.close()

        # Convert to DataFrame
        df = pd.DataFrame(frames_data)

        # Classify footwork
        before_mid = df[df['frame'] <= mid_frame]['ankle_distance'].sum()
        after_mid = df[df['frame'] > mid_frame]['ankle_distance'].sum()
        diff = after_mid - before_mid

        if diff > 0.1:
            result = "Front Foot Defence"
        elif diff < -0.1:
            result = "Back Foot Defence"
        else:
            result = "Defence (Footwork unclear)"

        # Save results
        self._save_results(df, result, output_folder, before_mid, after_mid, diff, total_frames)
        return result

    def _save_results(self, df, result, output_folder, before_mid, after_mid, diff, total_frames):
        """Save analysis results and plots"""
        # Save data to CSV
        df.to_csv(os.path.join(output_folder, "defence_analysis.csv"), index=False)

        # Create and save plot
        plt.figure(figsize=(10, 6))
        plt.plot(df['time'], df['ankle_distance'])
        plt.axvline(x=df.iloc[len(df)//2]['time'], color='r', linestyle='--', label='Mid Point')
        plt.title('Ankle Distance Over Time')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Normalized Distance')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_folder, "ankle_distance_plot.png"))
        plt.close()

        # Create detailed report
        report = (
            f"Defence Classification Analysis:\n"
            f"===============================\n"
            f"Total frames analyzed: {total_frames}\n"
            f"Ankle distance sum (first half): {before_mid:.4f}\n"
            f"Ankle distance sum (second half): {after_mid:.4f}\n"
            f"Difference (after - before): {diff:.4f}\n"
            f"\nClassification Logic:\n"
            f"- Positive difference indicates feet moving apart (Front Foot)\n"
            f"- Negative difference indicates feet coming together (Back Foot)\n"
            f"- Threshold of ±0.1 used to determine significant movement\n"
            f"\nFinal Classification: {result}\n"
        )

        # Save report
        with open(os.path.join(output_folder, "defence_classification_report.txt"), 'w') as f:
            f.write(report)


def process_defence_classification(video_path, output_folder):
    """
    Main function to process defence classification

    Args:
        video_path (str): Path to input video
        output_folder (str): Output directory

    Returns:
        str: Classification result
    """
    try:
        classifier = DefenceClassifier()
        return classifier.classify_defence(video_path, output_folder)
    except Exception as e:
        return f"Error during defence classification: {str(e)}"