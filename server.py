from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import tempfile
import uuid
import logging
import traceback
import shutil
from werkzeug.utils import secure_filename
import cv2
from typing import Dict, Any, Optional

# Import analysis modules
from pose_analyzer import analyze_cricket_landmarks
import shot_classifier

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure static file serving for feedback images
FEEDBACK_IMAGES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "feedback-images")
os.makedirs(FEEDBACK_IMAGES_DIR, exist_ok=True)

@app.route('/feedback-images/<filename>')
def serve_feedback_image(filename):
    """Serve feedback images from the feedback-images directory"""
    return send_from_directory(FEEDBACK_IMAGES_DIR, filename)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
UPLOAD_FOLDER = os.path.join(tempfile.gettempdir(), 'cricket_analyzer')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Model", "05", "image_classifier.h5")
MAX_VIDEO_SIZE = 100 * 1024 * 1024  # 100MB
ALLOWED_EXTENSIONS = {'.mp4', '.mov', '.avi'}

# Flag to track if initialization has been done
_is_initialized = False


def initialize_app():
    """Initialize resources when the first request comes in"""
    global _is_initialized
    if _is_initialized:
        return

    if not os.path.exists(MODEL_PATH):
        logger.warning(f"Model file not found at {MODEL_PATH}. Shot classification may not work properly.")

    _is_initialized = True


@app.before_request
def before_request():
    """Run before each request to ensure initialization"""
    initialize_app()


def allowed_file(filename: str) -> bool:
    """Check if the file has an allowed extension"""
    return os.path.splitext(filename)[1].lower() in ALLOWED_EXTENSIONS


def cleanup_folder(folder: str) -> None:
    """Clean up temporary files"""
    try:
        if os.path.exists(folder):
            shutil.rmtree(folder)
            logger.info(f"Cleaned up folder: {folder}")
    except Exception as e:
        logger.error(f"Error cleaning up folder {folder}: {e}")


def validate_video(video_path: str) -> bool:
    """Validate the video file is readable and has reasonable duration"""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        if fps <= 0 or frame_count <= 0:
            return False

        duration = frame_count / fps
        if duration > 120:  # 2 minutes max
            return False

        return True
    except Exception:
        return False


# In your existing server.py, modify the get_analysis_result function as follows:
def get_analysis_result(output_folder: str) -> Dict[str, Any]:
    """Extract analysis results from output folder"""
    highlights_folder = os.path.join(output_folder, "movement_highlights")
    shot_class_file = os.path.join(highlights_folder, "shot_classifications", "classification_summary.txt")

    logger.info(f"Looking for classification results in: {shot_class_file}")

    if not os.path.exists(shot_class_file):
        logger.error(f"Classification file not found at: {shot_class_file}")
        return {'strokeName': 'Unknown', 'error': 'Classification results not found'}

    with open(shot_class_file, 'r') as f:
        content = f.read()
        logger.info(f"Read classification file with {len(content)} characters")

    # Default result
    result = {
        'strokeName': 'Unknown',
        'feedback': []
    }

    # Extract stroke name (your existing code remains the same)
    sections = content.split('\n=== ')
    for section in sections:
        if "FINAL STROKE NAME" in section:
            logger.info(f"Found FINAL STROKE NAME section: {section}")
            lines = section.split('\n')
            if len(lines) > 1:
                final_stroke_name = lines[1].strip()
                logger.info(f"Extracted final stroke name: {final_stroke_name}")
                if final_stroke_name:
                    result['strokeName'] = final_stroke_name
                    break

    # If no FINAL STROKE NAME section, look for "FINAL RESULT"
    if result['strokeName'] == 'Unknown':
        for line in content.split('\n'):
            if "[FINAL RESULT] Final stroke name:" in line:
                logger.info(f"Found [FINAL RESULT] line: {line}")
                final_stroke_name = line.split("[FINAL RESULT] Final stroke name:")[1].strip()
                logger.info(f"Extracted final stroke name from FINAL RESULT: {final_stroke_name}")
                if final_stroke_name:
                    result['strokeName'] = final_stroke_name
                    break

    # Enhanced feedback extraction
    feedback_sections = []
    if "=== FEEDBACK ===" in content:
        feedback_content = content.split("=== FEEDBACK ===")[1]
        # Split by double newlines to separate feedback items
        feedback_items = [item.strip() for item in feedback_content.split('\n\n') if item.strip()]

        for item in feedback_items:
            lines = [line.strip() for line in item.split('\n') if line.strip()]
            if len(lines) >= 2:
                # Extract title by removing "Title:" prefix
                title_line = lines[0]
                if title_line.startswith('Title:'):
                    title = title_line.split('Title:')[1].strip()
                else:
                    title = title_line.replace(':', '').strip()

                description = '\n'.join(lines[1:]).strip()
                feedback_sections.append({
                    'title': title,  # Now contains just the title without "Title" prefix
                    'description': description
                })

    result['feedback'] = feedback_sections

    # Clean up stroke name by removing extra spaces
    if result['strokeName'] != 'Unknown':
        result['strokeName'] = ' '.join(result['strokeName'].split())

    logger.info(f"Returning analysis result: {result}")
    return result

@app.route('/', methods=['GET'])
def index():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'message': 'Cricket Shot Analyzer API is running',
        'model_available': os.path.exists(MODEL_PATH)
    })


@app.route('/analyze', methods=['POST'])
def analyze_video():
    """Endpoint to receive and analyze video uploads"""
    analysis_id = str(uuid.uuid4())
    output_folder = os.path.join(UPLOAD_FOLDER, analysis_id)
    os.makedirs(output_folder, exist_ok=True)

    try:
        # Check if video file was uploaded
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400

        video_file = request.files['video']
        if video_file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        # Validate file
        filename = secure_filename(video_file.filename)
        if not allowed_file(filename):
            return jsonify({'error': 'Unsupported file format'}), 400

        # Save uploaded file
        video_path = os.path.join(output_folder, filename)
        video_file.save(video_path)
        logger.info(f"Saved uploaded video to {video_path}")

        # Validate video
        if not validate_video(video_path):
            return jsonify({'error': 'Invalid video file'}), 400

        # Run analysis
        output_path = os.path.join(output_folder, f"analyzed_{filename}")
        success = analyze_cricket_landmarks(MODEL_PATH, video_path, output_path)

        if not success:
            return jsonify({'error': 'Video analysis failed'}), 500

        # Get and return results
        result = get_analysis_result(output_folder)
        logger.info(f"Returning stroke analysis result: {result}")
        return jsonify(result)

    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': 'Internal server error'}), 500

    finally:
        # Clean up in production (keep files for debugging)
        if not app.debug:
            cleanup_folder(output_folder)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)