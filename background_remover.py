import io
import os
from rembg import remove
from PIL import Image
import cv2
import numpy as np
from typing import List
import logging

logger = logging.getLogger(__name__)
def remove_background_from_images(input_folder: str, output_folder: str) -> List[str]:
    """
    Remove background from all images in the input folder and save them to output folder

    Args:
        input_folder: Path to folder containing input images
        output_folder: Path to save images with removed background

    Returns:
        List of paths to processed images
    """
    processed_files = []
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            try:
                input_path = os.path.join(input_folder, filename)
                output_path = os.path.join(output_folder, f"no_bg_{filename}")

                # Open and process image
                with open(input_path, 'rb') as f:
                    input_image = f.read()

                # Remove background
                output_image = remove(input_image)

                # Save result
                with open(output_path, 'wb') as f:
                    f.write(output_image)

                processed_files.append(output_path)

            except Exception as e:
                print(f"❌ Error processing {filename}: {e}")

    return processed_files


def remove_background_single_image(input_path: str, output_path: str) -> bool:
    """
    Remove background from a single image

    Args:
        input_path: Path to input image
        output_path: Path to save processed image

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        with open(input_path, 'rb') as f:
            input_image = f.read()

        output_image = remove(input_image)

        with open(output_path, 'wb') as f:
            f.write(output_image)

        return True
    except Exception as e:
        print(f"❌ Error processing {input_path}: {e}")
        return False


def process_frame_with_grey_background(input_path: str, output_path: str) -> bool:
    """
    Process a single frame by removing background and adding grey background

    Args:
        input_path: Path to input image
        output_path: Path to save processed image (same as input for overwrite)

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Read the image
        with open(input_path, 'rb') as f:
            input_image = f.read()

        # Remove background
        output_image = remove(input_image)

        # Convert to PIL Image
        pil_image = Image.open(io.BytesIO(output_image))

        # Create grey background
        grey_bg = Image.new('RGB', pil_image.size, (128, 128, 128))

        # Composite the foreground (without background) onto grey background
        if pil_image.mode == 'RGBA':
            grey_bg.paste(pil_image, (0, 0), pil_image)
        else:
            grey_bg.paste(pil_image, (0, 0))

        # Save the result (overwriting the original)
        grey_bg.save(output_path)

        return True
    except Exception as e:
        logger.error(f"Error processing frame with grey background: {e}")
        return False