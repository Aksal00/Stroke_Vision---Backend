#main.py

import os
from tkinter import messagebox
from ui_utils import select_video_file, select_output_folder
from pose_analyzer import analyze_cricket_landmarks
def main():
    # Check for model file
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Model", "image_classifier.h5")
    model_available = os.path.exists(model_path)

    # Select input video file
    video_path = select_video_file()
    if not video_path:
        messagebox.showinfo("Info", "No video file selected. Exiting.")
        return

    # Inform user about shot classification capability
    if model_available:
        messagebox.showinfo("Shot Classification",
                            "Model for cricket shot classification detected.\n\n"
                            "After video analysis, extracted frames will be classified as one of:\n"
                            "- drive\n"
                            "- legglance-flick\n"
                            "- pullshot\n"
                            "- sweep")

    # Select output folder
    output_folder = select_output_folder()
    if not output_folder:
        messagebox.showinfo("Info", "No output folder selected. Exiting.")
        return

    # Create output filename
    input_filename = os.path.basename(video_path)
    output_filename = "analyzed_" + input_filename
    output_path = os.path.join(output_folder, output_filename)

    # Check if output file already exists
    if os.path.exists(output_path):
        if not messagebox.askyesno("Confirm", "Output file already exists. Overwrite?"):
            return

    # Analyze the video
    success = analyze_cricket_landmarks(video_path, output_path)

    if success:
        messagebox.showinfo("Success",
                            f"Analysis complete!\nOutput saved to:\n{output_path}\n\nMovement highlight clips saved to:\n{os.path.join(output_folder, 'movement_highlights')}")
    else:
        messagebox.showerror("Error", "Video analysis failed")


if __name__ == "__main__":
    main()