#ui_utils.py

from tkinter import Tk, filedialog


def select_video_file():
    """
    Opens a file dialog to select a cricket video file.

    Returns:
        str: Path to the selected video file, or an empty string if no file was selected.
    """
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select Cricket Video File",
        filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv"), ("All Files", "*.*")]
    )
    return file_path


def select_output_folder():
    """
    Opens a file dialog to select an output folder.

    Returns:
        str: Path to the selected output folder, or an empty string if no folder was selected.
    """
    root = Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory(title="Select Output Folder")
    return folder_path