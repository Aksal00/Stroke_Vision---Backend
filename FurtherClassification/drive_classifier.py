import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from matplotlib import pyplot as plt
import matplotlib

matplotlib.use('Agg')


class DriveClassifier:
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
        self.left_elbow = self.mp_pose.PoseLandmark.LEFT_ELBOW.value
        self.right_elbow = self.mp_pose.PoseLandmark.RIGHT_ELBOW.value
        self.left_ankle = self.mp_pose.PoseLandmark.LEFT_ANKLE.value
        self.right_ankle = self.mp_pose.PoseLandmark.RIGHT_ANKLE.value

    def classify_drive(self, video_path, output_folder):
        """
        Classify a cricket drive shot into front foot or back foot drive

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
                'is_drive': False,
                'ankle_distance': 0
            }

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                # Check drive condition (elbow positions)
                if landmarks[self.right_elbow].x > landmarks[self.left_elbow].x:
                    frame_data['is_drive'] = True

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

        # Determine if it's a drive
        drive_frames = df[df['is_drive']].shape[0]
        if drive_frames < 5:  # Not enough drive frames
            return "Not a drive shot (insufficient drive characteristics)"

        # Classify footwork
        before_mid = df[df['frame'] <= mid_frame]['ankle_distance'].sum()
        after_mid = df[df['frame'] > mid_frame]['ankle_distance'].sum()
        diff = after_mid - before_mid

        if diff > 0.1:
            result = "Front Foot Drive"
        elif diff < -0.1:
            result = "Back Foot Drive"
        else:
            result = "Drive (Footwork unclear)"

        # Save results
        self._save_results(df, result, output_folder)
        return result

    def _save_results(self, df, result, output_folder):
        """Save analysis results and plots"""
        # Save data to CSV
        df.to_csv(os.path.join(output_folder, "drive_analysis.csv"), index=False)

        # Create and save plot
        plt.figure(figsize=(10, 6))
        plt.plot(df['time'], df['ankle_distance'])
        plt.title('Ankle Distance Over Time')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Normalized Distance')
        plt.grid(True)
        plt.savefig(os.path.join(output_folder, "ankle_distance_plot.png"))
        plt.close()

        # Save report
        report = f"Classification Result: {result}"
        with open(os.path.join(output_folder, "report.txt"), 'w') as f:
            f.write(report)


def process_drive_classification(video_path, output_folder):
    """
    Main function to process drive classification

    Args:
        video_path (str): Path to input video
        output_folder (str): Output directory

    Returns:
        str: Classification result
    """
    try:
        classifier = DriveClassifier()
        return classifier.classify_drive(video_path, output_folder)
    except Exception as e:
        return f"Error during drive classification: {str(e)}"