import os
import subprocess
import tempfile
import logging
from typing import Tuple, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VideoTrimmer:
    @staticmethod
    def trim_video(input_path: str, output_path: str, start_ms: int, end_ms: int) -> Tuple[bool, Optional[str]]:
        """
        Trim video using FFmpeg.

        Args:
            input_path: Path to input video file
            output_path: Path to save trimmed video
            start_ms: Start time in milliseconds
            end_ms: End time in milliseconds

        Returns:
            Tuple of (success, error_message)
        """
        try:
            # Convert milliseconds to seconds for FFmpeg
            start_sec = start_ms / 1000
            end_sec = end_ms / 1000
            duration = end_sec - start_sec

            # FFmpeg command (using stream copy for fastest processing)
            cmd = [
                'ffmpeg',
                '-y',  # Overwrite output file if exists
                '-ss', str(start_sec),  # Start time
                '-i', input_path,  # Input file
                '-to', str(duration),  # Duration
                '-c', 'copy',  # Stream copy (no re-encoding)
                output_path
            ]

            logger.info(f"Executing FFmpeg command: {' '.join(cmd)}")

            # Run FFmpeg
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True
            )

            logger.info(f"FFmpeg output: {result.stdout.decode()}")
            logger.info(f"Video trimmed successfully: {output_path}")
            return True, None

        except subprocess.CalledProcessError as e:
            error_msg = f"FFmpeg error: {e.stderr.decode()}"
            logger.error(error_msg)
            return False, error_msg
        except Exception as e:
            error_msg = f"Trimming error: {str(e)}"
            logger.error(error_msg)
            return False, error_msg

    @staticmethod
    def validate_trim_times(start_ms: int, end_ms: int, max_duration_ms: int = 10000) -> Tuple[bool, str]:
        """
        Validate trim times.

        Args:
            start_ms: Start time in milliseconds
            end_ms: End time in milliseconds
            max_duration_ms: Maximum allowed duration

        Returns:
            Tuple of (is_valid, error_message)
        """
        if start_ms < 0:
            return False, "Start time cannot be negative"
        if end_ms <= start_ms:
            return False, "End time must be after start time"
        if (end_ms - start_ms) > max_duration_ms:
            return False, f"Duration exceeds maximum allowed {max_duration_ms}ms"
        return True, ""

    @staticmethod
    def create_temp_video_path(extension: str = '.mp4') -> str:
        """Create a temporary file path for trimmed video"""
        temp_dir = tempfile.gettempdir()
        return os.path.join(temp_dir, f"trimmed_{os.urandom(8).hex()}{extension}")