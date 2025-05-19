# file_utils.py
import os
import shutil
import logging


def cleanup_feedback_images(feedback_images_dir: str) -> bool:
    """
    Cleans up all files in the feedback images directory.

    Args:
        feedback_images_dir: Path to the feedback images directory

    Returns:
        bool: True if cleanup was successful, False otherwise
    """
    try:
        if not os.path.exists(feedback_images_dir):
            logging.warning(f"Feedback images directory does not exist: {feedback_images_dir}")
            return True

        for filename in os.listdir(feedback_images_dir):
            file_path = os.path.join(feedback_images_dir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
                logging.info(f"Deleted: {file_path}")
            except Exception as e:
                logging.error(f"Failed to delete {file_path}. Reason: {e}")
                return False

        return True
    except Exception as e:
        logging.error(f"Error cleaning up feedback images: {e}")
        return False