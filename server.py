from flask import Flask, request, jsonify, send_from_directory, json
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

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Import analysis modules
try:
    from pose_analyzer import analyze_cricket_landmarks
    import shot_classifier

    ANALYSIS_MODULES_LOADED = True
except ImportError as e:
    logging.error(f"Error importing analysis modules: {str(e)}")
    ANALYSIS_MODULES_LOADED = False

app = Flask(__name__)
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
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Model", "05", "image_classifier.h5")
ALLOWED_EXTENSIONS = {'.mp4', '.mov', '.avi'}


@app.route('/', methods=['GET'])
def index():
    """Root endpoint to check server status"""
    return jsonify({
        "status": "online",
        "server": "Cricket Stroke Analyzer API",
        "version": "1.0.0"
    })


def cleanup_folder(folder_path):
    """
    Recursively delete a folder and all its contents.

    Args:
        folder_path: Path to the folder to be deleted
    """
    try:
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
            logging.info(f"Cleaned up temporary folder: {folder_path}")
    except Exception as e:
        logging.error(f"Error cleaning up folder {folder_path}: {str(e)}")
        logging.error(traceback.format_exc())


def get_analysis_result(output_folder: str) -> Dict[str, Any]:
    """Extract analysis results from output folder with improved feedback handling"""
    highlights_folder = os.path.join(output_folder, "movement_highlights")
    shot_class_folder = os.path.join(highlights_folder, "shot_classifications")
    shot_class_file = os.path.join(shot_class_folder, "classification_summary.txt")

    logging.info(f"Looking for results in: {output_folder}")

    # Initialize default result
    result = {
        'strokeName': 'Unknown',
        'dominantHand': 'Unknown',
        'feedback': []
    }

    # Check if classification file exists
    if not os.path.exists(shot_class_file):
        logging.error(f"Classification file not found: {shot_class_file}")
        return result

    try:
        with open(shot_class_file, 'r') as f:
            content = f.read()
            logging.info(f"Read {len(content)} bytes from classification file")

        # Extract stroke name
        stroke_name = 'Unknown'
        dominant_hand = 'Unknown'

        # Parse the file content
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if "=== FINAL STROKE NAME ===" in line:
                stroke_name = lines[i + 1].strip()
            elif "=== DOMINANT HAND ===" in line:
                dominant_hand = lines[i + 1].strip()

        result['strokeName'] = stroke_name
        result['dominantHand'] = dominant_hand

        # Improved feedback data extraction
        feedback_markers = [
            "Feedback data:",
            "=== FEEDBACK DATA ==="
        ]

        feedback_data = []
        for marker in feedback_markers:
            if marker in content:
                try:
                    json_start = content.find(marker) + len(marker)
                    json_str = content[json_start:].strip()

                    # Clean up the JSON string
                    json_str = json_str.split('\n===', 1)[0]  # Remove next section if present
                    json_str = json_str.strip()

                    if json_str:
                        feedback_data = json.loads(json_str)
                        logging.info(f"Successfully parsed feedback data with marker: {marker}")
                        break
                except json.JSONDecodeError as e:
                    logging.error(f"JSON decode error with marker {marker}: {str(e)}")
                    logging.error(f"Problematic JSON string: {json_str[:200]}...")
                except Exception as e:
                    logging.error(f"Error processing feedback with marker {marker}: {str(e)}")

        # Convert feedback data to expected format
        result['feedback'] = []
        for item in feedback_data:
            try:
                feedback_item = {
                    'title': item.get('title', ''),
                    'description': item.get('feedback_text', item.get('description', '')),
                    'is_ideal': item.get('is_ideal', False),
                    'image_url': item.get('image_url', ''),
                    'ref-images': item.get('ref-images', [])
                }
                result['feedback'].append(feedback_item)
            except Exception as e:
                logging.error(f"Error processing feedback item: {str(e)}")
                continue

        logging.info(f"Found {len(result['feedback'])} feedback items")

    except Exception as e:
        logging.error(f"Error processing analysis results: {str(e)}")
        logging.error(traceback.format_exc())

    return result


@app.route('/analyze', methods=['POST'])
def analyze_video():
    if not ANALYSIS_MODULES_LOADED:
        return jsonify({'error': 'Analysis modules not available'}), 500

    analysis_id = str(uuid.uuid4())
    output_folder = os.path.join(UPLOAD_FOLDER, analysis_id)
    os.makedirs(output_folder, exist_ok=True)

    logging.info(f"Created analysis directory: {output_folder}")

    try:
        logging.info(f"Received analyze request, content type: {request.content_type}")

        # Debug request information
        logging.info(f"Request form data keys: {list(request.form.keys()) if request.form else 'None'}")
        logging.info(f"Request files keys: {list(request.files.keys()) if request.files else 'None'}")

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

        video_path = os.path.join(output_folder, filename)
        video_file.save(video_path)
        logging.info(f"Saved video to {video_path}")

        # Check if file was actually saved and is readable
        if not os.path.exists(video_path):
            logging.error(f"File not saved: {video_path}")
            return jsonify({'error': 'Failed to save video file'}), 500

        file_size = os.path.getsize(video_path)
        if file_size == 0:
            logging.error(f"Saved file is empty: {video_path}")
            return jsonify({'error': 'Uploaded file is empty'}), 400

        logging.info(f"Video file saved successfully: {video_path}, size: {file_size} bytes")

        # Get dominant hand from form data if available
        dominant_hand = request.form.get('dominantHand', 'right')  # Default to right-handed
        logging.info(f"Using dominant hand: {dominant_hand}")

        # Run analysis
        output_path = os.path.join(output_folder, f"analyzed_{filename}")

        logging.info(f"Starting video analysis with model: {MODEL_PATH}")
        if not os.path.exists(MODEL_PATH):
            logging.error(f"Model file not found: {MODEL_PATH}")
            return jsonify({'error': 'Model file not found'}), 500

        # Log information before and after calling analyze_cricket_landmarks
        logging.info(f"About to call analyze_cricket_landmarks with parameters:")
        logging.info(f"  - MODEL_PATH: {MODEL_PATH}")
        logging.info(f"  - video_path: {video_path}")
        logging.info(f"  - output_path: {output_path}")
        logging.info(f"  - dominant_hand: {dominant_hand}")

        analysis_result = analyze_cricket_landmarks(
            MODEL_PATH,
            video_path,
            output_path,
        )

        logging.info(f"analyze_cricket_landmarks returned: {analysis_result}")

        if not analysis_result:
            logging.error("Video analysis failed")
            return jsonify({'error': 'Video analysis failed'}), 500

        # List contents of the output directory after analysis
        logging.info(f"After analysis, contents of output folder {output_folder}: {os.listdir(output_folder)}")

        # Check if the movement_highlights folder was created
        movement_highlights_path = os.path.join(output_folder, "movement_highlights")
        if os.path.exists(movement_highlights_path):
            logging.info(f"Contents of movement_highlights folder: {os.listdir(movement_highlights_path)}")

            # Check if shot_classifications folder exists
            shot_class_path = os.path.join(movement_highlights_path, "shot_classifications")
            if os.path.exists(shot_class_path):
                logging.info(f"Contents of shot_classifications folder: {os.listdir(shot_class_path)}")
        else:
            logging.error(f"movement_highlights folder not created at: {movement_highlights_path}")

        # Get the analysis results
        logging.info("Calling get_analysis_result to extract results")
        result = get_analysis_result(output_folder)

        logging.info(f"Analysis completed, final result to send to frontend: {result}")
        return jsonify(result)

    except Exception as e:
        logging.error(f"Error processing video: {str(e)}")
        logging.error(traceback.format_exc())
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

    finally:
        # In debug mode, keep the files for inspection
        if not app.debug:
            cleanup_folder(output_folder)
        else:
            logging.info(f"Debug mode enabled, keeping output folder: {output_folder}")


@app.route('/status', methods=['GET'])
def check_status():
    """
    Additional endpoint to check server status and configuration
    """
    try:
        return jsonify({
            "status": "online",
            "model_loaded": os.path.exists(MODEL_PATH),
            "analysis_modules": ANALYSIS_MODULES_LOADED,
            "upload_folder": UPLOAD_FOLDER,
            "allowed_extensions": list(ALLOWED_EXTENSIONS)
        })
    except Exception as e:
        logging.error(f"Error in status check: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == '__main__':
    logging.info("Starting Flask server on 0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)