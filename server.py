from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import tempfile
import uuid
import logging
import traceback
import shutil
import json
import numpy as np
from werkzeug.utils import secure_filename
import cv2
from typing import Dict, Any

from dominant_hand import determine_dominant_hand
from file_utils import cleanup_feedback_images
from highlight_generator import export_highlight_frames
from video_analyzer import video_analyzer
from video_trimmer import VideoTrimmer
import shot_classifier

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
ANALYSIS_MODULES_LOADED = True

class NumpyJSONEncoder(json.JSONEncoder):
    """ Custom JSON encoder that handles numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyJSONEncoder, self).default(obj)

app = Flask(__name__)
app.json_encoder = NumpyJSONEncoder  # Use our custom encoder
CORS(app, resources={r"/*": {"origins": "*"}})

# Configure static file serving
FEEDBACK_IMAGES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "feedback-images")
REF_IMAGES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ref-images")
os.makedirs(FEEDBACK_IMAGES_DIR, exist_ok=True)
os.makedirs(REF_IMAGES_DIR, exist_ok=True)

@app.route('/feedback-images/<filename>')
def serve_feedback_image(filename):
    return send_from_directory(FEEDBACK_IMAGES_DIR, filename)

@app.route('/ref-images/<filename>')
def serve_ref_image(filename):
    return send_from_directory(REF_IMAGES_DIR, filename)

# Configuration
UPLOAD_FOLDER = os.path.join(tempfile.gettempdir(), 'cricket_analyzer')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Model", "StrokeVisionModel",
                          "pose_keypoint_classifier.h5")
ALLOWED_EXTENSIONS = {'.mp4', '.mov', '.avi'}

def ensure_directory_structure(output_folder: str) -> bool:
    """Ensure all required directories exist in the output folder structure"""
    try:
        # Create main directories
        os.makedirs(output_folder, exist_ok=True)

        # Create expected subdirectories
        required_dirs = [
            os.path.join(output_folder, "movement_highlights"),
            os.path.join(output_folder, "movement_highlights", "extracted_frames"),
            os.path.join(output_folder, "movement_highlights", "shot_classifications"),
            os.path.join(output_folder, "movement_highlights", "frames_with_landmarks")
        ]

        for dir_path in required_dirs:
            os.makedirs(dir_path, exist_ok=True)

        return True
    except Exception as e:
        logging.error(f"Error creating directory structure: {str(e)}")
        return False

def cleanup_folder(folder_path):
    """Recursively delete a folder and all its contents."""
    try:
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
            logging.info(f"Cleaned up temporary folder: {folder_path}")
    except Exception as e:
        logging.error(f"Error cleaning up folder {folder_path}: {str(e)}")
        logging.error(traceback.format_exc())

def convert_numpy_types(obj):
    """Recursively convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    return obj

@app.route('/analyze', methods=['POST'])
def analyze_video():
    if not ANALYSIS_MODULES_LOADED:
        return jsonify({'error': 'Analysis modules not available'}), 500

    # Clean up old feedback images
    cleanup_feedback_images(FEEDBACK_IMAGES_DIR)

    # Generate unique analysis ID and create output folder
    analysis_id = str(uuid.uuid4())
    output_folder = os.path.join(UPLOAD_FOLDER, analysis_id)

    # Ensure directory structure exists before processing
    if not ensure_directory_structure(output_folder):
        return jsonify({'error': 'Failed to create analysis directory structure'}), 500

    logging.info(f"Created analysis directory: {output_folder}")

    try:
        if 'video' not in request.files:
            logging.error("No video file in request")
            return jsonify({'error': 'No video file provided'}), 400

        video_file = request.files['video']
        if video_file.filename == '':
            logging.error("Empty filename received")
            return jsonify({'error': 'No selected file'}), 400

        filename = secure_filename(video_file.filename)
        file_ext = os.path.splitext(filename)[1].lower()

        if file_ext not in ALLOWED_EXTENSIONS:
            logging.error(f"Unsupported file extension: {file_ext}")
            return jsonify({'error': f'Unsupported file format: {file_ext}. Allowed: {ALLOWED_EXTENSIONS}'}), 400

        # Save original video with analysis ID in filename
        original_video_path = os.path.join(output_folder, f"original_{analysis_id}_{filename}")
        video_file.save(original_video_path)
        logging.info(f"Saved original video to {original_video_path}")

        if not os.path.exists(original_video_path):
            logging.error(f"File not saved: {original_video_path}")
            return jsonify({'error': 'Failed to save video file'}), 500

        # Get trim parameters
        try:
            trim_start = int(float(request.form.get('trimStart', '0')))
            trim_end = int(float(request.form.get('trimEnd', '0')))
        except (ValueError, TypeError) as e:
            logging.error(f"Trim conversion failed: {str(e)}")
            return jsonify({'error': 'Invalid trim parameters'}), 400

        # Determine which video to analyze
        video_to_analyze = original_video_path

        # If trim times are specified, trim the video
        if trim_end > 0:
            is_valid, error_msg = VideoTrimmer.validate_trim_times(trim_start, trim_end)
            if not is_valid:
                logging.error(f"Invalid trim times: {error_msg}")
                return jsonify({'error': error_msg}), 400

            trimmed_path = os.path.join(output_folder, f"trimmed_{analysis_id}_{filename}")
            success, error = VideoTrimmer.trim_video(
                original_video_path,
                trimmed_path,
                trim_start,
                trim_end
            )

            if not success:
                logging.error(f"Video trimming failed: {error}")
                return jsonify({'error': 'Video trimming failed'}), 500

            video_to_analyze = trimmed_path
            logging.info(f"Using trimmed video for analysis: {video_to_analyze}")

        # Determine dominant hand
        dominant_hand, processed_video_path = determine_dominant_hand(video_to_analyze)

        if dominant_hand == 'left':
            video_path_to_process = processed_video_path
            logger.info(f"Processing mirrored video for left-handed batsman: {processed_video_path}")
        else:
            video_path_to_process = video_to_analyze
            logger.info(f"Processing original video for right-handed batsman")
        # Create highlights folder and store the video
        highlights_folder = os.path.join(output_folder, "movement_highlights")
        os.makedirs(highlights_folder, exist_ok=True)
        video_in_highlights = os.path.join(highlights_folder, "extracted_frames", "video_to_process.mp4")
        shutil.copy2(video_path_to_process, video_in_highlights)
        # Generate highlight frames
        logger.info("Generating highlight frames")
        original_frames, fps = video_analyzer(video_path_to_process)

        export_success = export_highlight_frames(
            video_path_to_process,
            original_frames,
            fps,
            output_folder,
            dominant_hand
        )

        # Run shot classification on extracted frames
        highlights_folder = os.path.join(output_folder, "movement_highlights")
        result = shot_classifier.classify_highlight_frames(
            highlights_folder,
            MODEL_PATH,
            output_folder,
            dominant_hand,
            video_path_to_process
        )

        # Convert numpy types in the result before JSON serialization
        result = convert_numpy_types(result)

        # Format the response with analysis_id included
        response = {
            'analysis_id': analysis_id,
            'strokeName': result.get('final_stroke_name', 'Unknown'),
            'dominantHand': result.get('dominant_hand', 'Unknown'),
            'feedback': result.get('feedback_data', []),
            'processing_stats': result.get('processing_stats', {}) if app.debug else None
        }

        # Save the complete results for potential debugging
        if app.debug:
            analysis_json_path = os.path.join(output_folder, "analysis_results.json")
            try:
                with open(analysis_json_path, 'w') as f:
                    json.dump(response, f, indent=2, cls=NumpyJSONEncoder)
                logging.info(f"Saved complete analysis results to {analysis_json_path}")
            except Exception as e:
                logging.error(f"Failed to save analysis results: {str(e)}")

        if result.get('error'):
            response['error'] = result['error']
            return jsonify(response), 500

        return jsonify(response)

    except Exception as e:
        logging.error(f"Error processing video: {str(e)}")
        logging.error(traceback.format_exc())
        return jsonify({
            'analysis_id': analysis_id,
            'error': f'Internal server error: {str(e)}',
            'debug_info': {
                'output_folder': output_folder,
                'traceback': traceback.format_exc()
            } if app.debug else None
        }), 500

    finally:
        if not app.debug:
            cleanup_folder(output_folder)

@app.route('/', methods=['GET'])
def index():
    return jsonify({
        "status": "online",
        "server": "Cricket Stroke Analyzer API",
        "version": "1.0.0"
    })

@app.route('/status', methods=['GET'])
def check_status():
    try:
        return jsonify({
            "status": "online",
            "model_loaded": os.path.exists(MODEL_PATH),
            "analysis_modules": ANALYSIS_MODULES_LOADED,
            "upload_folder": UPLOAD_FOLDER,
            "allowed_extensions": list(ALLOWED_EXTENSIONS),
            "directories": {
                "feedback_images": FEEDBACK_IMAGES_DIR,
                "ref_images": REF_IMAGES_DIR
            }
        })
    except Exception as e:
        logging.error(f"Error in status check: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    logging.info("Starting Flask server on 0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)