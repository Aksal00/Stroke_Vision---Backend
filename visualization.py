import os

import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # Set non-interactive backend
from scipy.signal import find_peaks
import mediapipe as mp
import logging
from typing import List, Dict, Any, Optional

# Configure logging
logger = logging.getLogger(__name__)


def visualize_landmark_data(
        landmark_data: List[Dict[str, Any]],
        output_folder: str
) -> Optional[pd.DataFrame]:
    """
    Analyze and visualize landmark movement data

    Args:
        landmark_data: List of landmark data dictionaries
        output_folder: Directory to save visualizations

    Returns:
        DataFrame with peak movement info or None if failed
    """
    try:
        # Convert to DataFrame
        df = pd.DataFrame(landmark_data)

        # Calculate movement metrics for key landmarks
        key_landmarks = {
            'left_wrist': mp.solutions.pose.PoseLandmark.LEFT_WRIST.value,
            'right_wrist': mp.solutions.pose.PoseLandmark.RIGHT_WRIST.value,
            'left_elbow': mp.solutions.pose.PoseLandmark.LEFT_ELBOW.value,
            'right_elbow': mp.solutions.pose.PoseLandmark.RIGHT_ELBOW.value,
            'left_shoulder': mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value,
            'right_shoulder': mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value,
        }

        # Calculate velocity for each landmark
        plt.figure(figsize=(15, 10))
        for name, idx in key_landmarks.items():
            x_col = f'landmark_{idx}_x'
            y_col = f'landmark_{idx}_y'

            # Calculate and smooth displacement
            df[f'{name}_displacement'] = np.sqrt(
                df[x_col].diff() ** 2 + df[y_col].diff() ** 2
            ).fillna(0)
            df[f'{name}_displacement_smoothed'] = (
                df[f'{name}_displacement'].rolling(window=5, min_periods=1).mean()
            )
            plt.plot(df['time'], df[f'{name}_displacement_smoothed'], label=name)

        # Save movement plot
        plt.title('Landmark Movement Velocity Over Time')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Displacement (normalized coordinates)')
        plt.legend()
        plt.grid(True)
        plot_path = os.path.join(output_folder, 'landmark_movement.png')
        plt.savefig(plot_path)
        plt.close()

        # Calculate total movement and find peaks
        df['total_movement'] = sum(df[f'{name}_displacement_smoothed'] for name in key_landmarks.keys())
        peaks, _ = find_peaks(df['total_movement'], height=0.05, distance=10)

        # Plot peaks
        plt.figure(figsize=(15, 10))
        plt.plot(df['time'], df['total_movement'], label='Total Movement')
        plt.plot(df['time'].iloc[peaks], df['total_movement'].iloc[peaks], 'ro', label='Significant Movements')
        plt.title('Total Body Movement with Significant Peaks Marked')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Combined Displacement')
        plt.legend()
        plt.grid(True)
        peak_plot_path = os.path.join(output_folder, 'movement_peaks.png')
        plt.savefig(peak_plot_path)
        plt.close()

        # Save data to files
        csv_path = os.path.join(output_folder, 'landmark_data.csv')
        df.to_csv(csv_path, index=False)

        # Get significant movements
        significant_movements = df.iloc[peaks][['frame', 'time', 'total_movement']]

        txt_path = os.path.join(output_folder, 'significant_movements.txt')
        with open(txt_path, 'w') as f:
            f.write("Significant movement periods (time in seconds):\n")
            f.write(significant_movements.to_string())

        logger.info(f"Found {len(significant_movements)} significant movements")
        return significant_movements

    except Exception as e:
        logger.error(f"Error visualizing landmark data: {e}")
        return None