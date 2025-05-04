#visualization.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import mediapipe as mp
from tkinter import messagebox


def visualize_landmark_data(landmark_data, output_folder):
    """
    Analyze and visualize landmark movement data, identifying significant movement periods.

    Args:
        landmark_data (list): List of dictionaries containing pose landmark data for each frame.
        output_folder (str): Directory where visualization outputs will be saved.

    Returns:
        DataFrame: Information about significant movement peaks, or None if no peaks were found.
    """
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

    # Calculate velocity (change in position per frame) for each landmark
    plt.figure(figsize=(15, 10))

    for name, idx in key_landmarks.items():
        x_col = f'landmark_{idx}_x'
        y_col = f'landmark_{idx}_y'

        # Calculate displacement between frames
        df[f'{name}_displacement'] = np.sqrt(
            df[x_col].diff() ** 2 + df[y_col].diff() ** 2
        ).fillna(0)

        # Smooth the displacement with rolling average
        df[f'{name}_displacement_smoothed'] = df[f'{name}_displacement'].rolling(window=5, min_periods=1).mean()

        # Plot the displacement over time
        plt.plot(df['time'], df[f'{name}_displacement_smoothed'], label=name)

    plt.title('Landmark Movement Velocity Over Time')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Displacement (normalized coordinates)')
    plt.legend()
    plt.grid(True)

    # Save the plot
    plot_path = os.path.join(output_folder, 'landmark_movement.png')
    plt.savefig(plot_path)
    plt.close()

    # Identify significant movement periods (peaks in velocity)
    plt.figure(figsize=(15, 10))

    # Calculate total movement (sum of all key landmarks)
    df['total_movement'] = sum(df[f'{name}_displacement_smoothed'] for name in key_landmarks.keys())

    # Find peaks in movement
    peaks, _ = find_peaks(df['total_movement'], height=0.05, distance=10)  # Adjust these parameters as needed

    plt.plot(df['time'], df['total_movement'], label='Total Movement')
    plt.plot(df['time'].iloc[peaks], df['total_movement'].iloc[peaks], 'ro', label='Significant Movements')
    plt.title('Total Body Movement with Significant Peaks Marked')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Combined Displacement')
    plt.legend()
    plt.grid(True)

    # Save the peak plot
    peak_plot_path = os.path.join(output_folder, 'movement_peaks.png')
    plt.savefig(peak_plot_path)
    plt.close()

    # Create a CSV with the movement data
    csv_path = os.path.join(output_folder, 'landmark_data.csv')
    df.to_csv(csv_path, index=False)

    # Get significant movement periods
    significant_movements = df.iloc[peaks][['frame', 'time', 'total_movement']]
    print("\nSignificant movement periods:")
    print(significant_movements)

    # Save significant movements to a text file
    txt_path = os.path.join(output_folder, 'significant_movements.txt')
    with open(txt_path, 'w') as f:
        f.write("Significant movement periods (time in seconds):\n")
        f.write(significant_movements.to_string())

    messagebox.showinfo("Analysis Complete",
                        f"Landmark analysis complete!\n"
                        f"Charts and data saved to:\n{output_folder}\n\n"
                        f"Significant movements detected at:\n{significant_movements['time'].values} seconds")

    return significant_movements