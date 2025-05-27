import os
import traceback
import numpy as np
import cv2
import logging
from typing import List, Tuple, Dict, Any, Optional
import keras
from flask import json
import mediapipe as mp
from Feedback.ForwardDefence.ForwardDefence import process_forward_defence_feedback
from Feedback.BackwardDefence.BackwardDefence import process_backward_defence_feedback
from result_writer import write_classification_results

# Set up logging
logger = logging.getLogger(__name__)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

pose = mp_pose.Pose(
    static_image_mode=True,
    model_complexity=2,
    enable_segmentation=False,
    min_detection_confidence=0.5
)

# Model caching
_model_cache = None


def load_shot_classifier_model(model_path: str) -> Optional[keras.Model]:
    """Load and cache the shot classification model"""
    global _model_cache

    if _model_cache is not None:
        logger.info("Using cached model")
        return _model_cache

    try:
        logger.info(f"Loading model from {model_path}")
        _model_cache = keras.models.load_model(model_path)
        logger.info("Model loaded successfully")
        return _model_cache
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None


def extract_and_normalize_keypoints(img_path: str) -> Optional[np.ndarray]:
    """Extract and normalize MediaPipe keypoints from an image"""
    try:
        img = cv2.imread(img_path)
        if img is None:
            logger.error(f"Could not read image at {img_path}")
            return None

        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(img_rgb)

        if not results.pose_landmarks:
            logger.warning(f"No pose detected in {os.path.basename(img_path)}")
            return None

        # Extract all 33 landmarks (x,y,z)
        keypoints = []
        for landmark in results.pose_landmarks.landmark:
            keypoints.extend([landmark.x, landmark.y, landmark.z])

        keypoints = np.array(keypoints).reshape(-1, 3)  # Reshape to (33, 3)

        # Hip-centered normalization
        left_hip = keypoints[23]
        right_hip = keypoints[24]
        mid_hip = (left_hip + right_hip) / 2
        keypoints_normalized = keypoints - mid_hip

        # Scale by torso length (neck to mid-hip distance)
        neck = keypoints[11]  # Using shoulder as neck approximation
        torso_length = np.linalg.norm(neck - mid_hip)

        if torso_length > 0:
            keypoints_normalized /= torso_length
        else:
            logger.warning(f"Zero torso length detected in {os.path.basename(img_path)}")
            return None

        return keypoints_normalized.flatten()  # Return as flat array (99 values)

    except Exception as e:
        logger.error(f"Error extracting keypoints from {img_path}: {e}")
        return None


def save_frame_with_landmarks(img_path: str, output_path: str, confidence: float) -> bool:
    """Save frame with MediaPipe pose landmarks drawn on it"""
    try:
        img = cv2.imread(img_path)
        if img is None:
            logger.error(f"Could not read image at {img_path}")
            return False

        # Convert BGR to RGB for MediaPipe processing
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(img_rgb)

        if not results.pose_landmarks:
            logger.warning(f"No pose detected in {os.path.basename(img_path)}")
            return False

        # Draw landmarks on the original BGR image
        mp_drawing.draw_landmarks(
            img,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
        )

        # Add confidence text
        height, width = img.shape[:2]
        font_scale = max(0.5, min(width, height) / 1000)
        thickness = max(1, int(font_scale * 2))

        cv2.putText(
            img,
            f"Confidence: {confidence:.2%}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 255, 0),  # Green color
            thickness
        )

        # Save the image
        success = cv2.imwrite(output_path, img)
        if success:
            logger.info(f"Saved frame with landmarks to {output_path}")
        else:
            logger.error(f"Failed to save frame with landmarks to {output_path}")

        return success

    except Exception as e:
        logger.error(f"Error saving frame with landmarks: {e}")
        return False


def predict_top_shots_from_keypoints(
        model: keras.Model,
        img_path: str,
        class_names: List[str],
        top_n: int = 3
) -> List[Tuple[str, float]]:
    """Predict top shot classes from keypoints extracted from an image"""
    try:
        # Extract and normalize keypoints
        keypoints = extract_and_normalize_keypoints(img_path)
        if keypoints is None:
            logger.warning(f"Could not extract keypoints from {img_path}")
            return [("NoPoseDetected", 0.0)]

        # Verify keypoints shape
        if len(keypoints) != 99:  # 33 landmarks * 3 coordinates
            logger.error(f"Invalid keypoints shape for {img_path}")
            return [("InvalidKeypoints", 0.0)]

        # Reshape for prediction (batch of 1)
        keypoints = np.expand_dims(keypoints, axis=0)

        # Make prediction
        preds = model.predict(keypoints, verbose=0)[0]
        top_indices = np.argsort(preds)[-top_n:][::-1]
        return [(class_names[i], float(preds[i])) for i in top_indices]

    except Exception as e:
        logger.error(f"Error predicting image {img_path}: {e}")
        return [("Error", 0.0)]


def calculate_weighted_prediction(top_predictions: List[Tuple[str, float]]) -> str:
    """Calculate weighted prediction from multiple predictions"""
    class_scores = {}
    for pred_class, confidence in top_predictions:
        class_scores[pred_class] = class_scores.get(pred_class, 0.0) + confidence

    if not class_scores:
        logger.warning("No predictions available for weighted calculation")
        return "No predictions available"

    weighted_pred = max(class_scores.items(), key=lambda x: x[1])
    avg_confidence = weighted_pred[1] / len(top_predictions)
    return f"{weighted_pred[0]} ({avg_confidence:.2%} avg)"


def combine_frame_predictions(results: Dict[str, Any]) -> str:
    """Combine predictions from multiple frames"""
    if not results:
        logger.warning("No frames available for prediction")
        return "No frames available for prediction"

    vote_counts = {}
    confidence_sums = {}

    for frame_path, (top_predictions, _) in results.items():
        if top_predictions and top_predictions[0][0] != "Error":
            top_class = top_predictions[0][0]
            vote_counts[top_class] = vote_counts.get(top_class, 0) + 1
            confidence_sums[top_class] = confidence_sums.get(top_class, 0.0) + top_predictions[0][1]

    if not vote_counts:
        logger.warning("No valid predictions available")
        return "No valid predictions available"

    max_votes = max(vote_counts.values())
    winning_classes = [cls for cls, votes in vote_counts.items() if votes == max_votes]

    if len(winning_classes) > 1:
        max_confidence = max(confidence_sums[cls] for cls in winning_classes)
        confident_classes = [cls for cls in winning_classes if confidence_sums[cls] == max_confidence]

        if len(confident_classes) > 1:
            return "Stroke cannot be clearly identified (multiple possibilities)"
        return f"{confident_classes[0]}"

    avg_confidence = confidence_sums[winning_classes[0]] / vote_counts[winning_classes[0]]
    return f"Final Prediction: {winning_classes[0]}"


def get_top_confidence_frames(results: Dict[str, Any], top_n: int = 3) -> List[Tuple[str, float]]:
    """Get the top N frames with highest confidence scores"""
    frame_confidences = []

    for frame_path, (top_predictions, _) in results.items():
        if top_predictions and top_predictions[0][0] not in ["Error", "NoPoseDetected", "InvalidKeypoints"]:
            confidence = top_predictions[0][1]  # Get the top prediction's confidence
            frame_confidences.append((frame_path, confidence))

    # Sort by confidence in descending order
    frame_confidences.sort(key=lambda x: x[1], reverse=True)

    # Return top N frames
    return frame_confidences[:top_n]


def apply_further_classification(
        final_prediction: str,
        output_folder: str,
        video_path: str
) -> str:
    """Apply further classification based on the initial prediction"""
    logger.info(f"Starting further classification with prediction: {final_prediction}")

    # Initialize final_stroke with a default value
    final_stroke = final_prediction.split(':')[-1].strip().split('(')[0].strip().lower()

    # Extract base stroke name
    if "Final Prediction:" in final_prediction:
        parts = final_prediction.split("Final Prediction:")
        prediction_part = parts[1].strip()
        base_stroke = prediction_part.split("(")[0].strip().lower()
    else:
        base_stroke = final_prediction.split(':')[-1].strip().split('(')[0].strip().lower()

    logger.info(f"Base stroke extracted: {base_stroke}")

    # If base stroke is defence, run defence classifier
    if base_stroke == "defence":
        try:
            from FurtherClassification.defence_classifier import process_defence_classification
            defence_output_folder = os.path.join(output_folder, "defence_analysis")
            os.makedirs(defence_output_folder, exist_ok=True)

            logger.info(f"Running defence classifier on video: {video_path}")
            defence_result = process_defence_classification(video_path, defence_output_folder)

            # Update the final stroke name with more specific defence type
            if defence_result:
                final_stroke = defence_result.split("(")[0].strip().lower()
            logger.info(f"Defence classifier result: {defence_result}")
        except Exception as e:
            logger.error(f"Error running defence classifier: {str(e)}")
            # Fall back to generic defence if classifier fails
            final_stroke = "defence"

    elif base_stroke == "drive":
        try:
            from FurtherClassification.drive_classifier import process_drive_classification
            drive_output_folder = os.path.join(output_folder, "drive_analysis")
            os.makedirs(drive_output_folder, exist_ok=True)

            logger.info(f"Running drive classifier on video: {video_path}")
            drive_result = process_drive_classification(video_path, drive_output_folder)

            # Update the final stroke name with more specific drive type
            if drive_result:
                final_stroke = drive_result.split("(")[0].strip().lower()
            logger.info(f"Drive classifier result: {drive_result}")
        except Exception as e:
            logger.error(f"Error running drive classifier: {str(e)}")
            # Fall back to generic drive if classifier fails
            final_stroke = "drive"

    # Only capitalize if we have a non-empty string
    if final_stroke:
        final_stroke = final_stroke[0].upper() + final_stroke[1:].lower()
    else:
        final_stroke = "Unknown"  # Fallback value

    return final_stroke

def classify_highlight_frames(
        highlights_folder: str,
        model_path: str,
        output_folder: str,
        dominant_hand: str,
        video_path: str  # Add video_path parameter
) -> Dict[str, Any]:
    """Classify frames from highlight clips using MediaPipe keypoints"""
    logger.info(f"Starting shot classification process")

    # Initialize result with error field
    result = {
        'frame_predictions': {},
        'initial_prediction': "Analysis failed",
        'final_stroke_name': "Unknown",
        'dominant_hand': dominant_hand,
        'feedback_data': [],
        'report_path': None,
        'error': None,
        'processing_stats': {
            'total_frames': 0,
            'successful_predictions': 0,
            'failed_predictions': 0,
            'landmark_frames_saved': 0
        }
    }

    try:
        # Create necessary folders if they don't exist
        os.makedirs(highlights_folder, exist_ok=True)
        frame_folder = os.path.join(highlights_folder, "extracted_frames")
        os.makedirs(frame_folder, exist_ok=True)
        results_folder = os.path.join(highlights_folder, "shot_classifications")
        os.makedirs(results_folder, exist_ok=True)  # This is the missing folder
        landmarks_folder = os.path.join(highlights_folder, "frames_with_landmarks")
        os.makedirs(landmarks_folder, exist_ok=True)

        # Define class names
        class_names = ['cut', 'defence', 'drive', 'legglance', 'pullshot', 'sweep']
        logger.info(f"Using classes: {class_names}")

        # Load model
        model = load_shot_classifier_model(model_path)
        if model is None:
            error_msg = "Failed to load model, aborting classification"
            logger.error(error_msg)
            result['error'] = error_msg
            return result

        # Find frame images in extracted_frames folder
        frame_files = [f for f in os.listdir(frame_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not frame_files:
            error_msg = "No frame images found in highlights folder"
            logger.warning(error_msg)
            result['error'] = error_msg
            return result

        # Prepare output folders
        results_folder = os.path.join(highlights_folder, "shot_classifications")
        landmarks_folder = os.path.join(highlights_folder, "frames_with_landmarks")
        os.makedirs(results_folder, exist_ok=True)
        os.makedirs(landmarks_folder, exist_ok=True)

        # Classify each frame using keypoints
        results = {}
        successful_predictions = 0

        for frame_file in frame_files:
            frame_path = os.path.join(frame_folder, frame_file)
            top_predictions = predict_top_shots_from_keypoints(model, frame_path, class_names, top_n=3)

            if top_predictions and top_predictions[0][0] != "Error":
                successful_predictions += 1
                weighted_pred = calculate_weighted_prediction(top_predictions)
                results[frame_path] = (top_predictions, weighted_pred)

        if successful_predictions == 0:
            error_msg = "No frames could be successfully processed"
            logger.error(error_msg)
            result['error'] = error_msg
            return result

        # Update processing stats
        result['processing_stats'] = {
            'total_frames': len(frame_files),
            'successful_predictions': successful_predictions,
            'failed_predictions': len(frame_files) - successful_predictions,
            'landmark_frames_saved': 0
        }

        # Get top 3 frames with highest confidence and save them with landmarks
        top_frames = get_top_confidence_frames(results, top_n=3)
        logger.info(f"Saving top {len(top_frames)} frames with landmarks")

        saved_landmark_frames = []
        for i, (frame_path, confidence) in enumerate(top_frames, 1):
            frame_filename = os.path.basename(frame_path)
            name, ext = os.path.splitext(frame_filename)
            landmark_filename = f"top_{i}_{name}_landmarks{ext}"
            landmark_path = os.path.join(landmarks_folder, landmark_filename)

            if save_frame_with_landmarks(frame_path, landmark_path, confidence):
                saved_landmark_frames.append(landmark_path)
                logger.info(f"Saved landmark frame {i}: {landmark_filename} (confidence: {confidence:.2%})")
                result['processing_stats']['landmark_frames_saved'] += 1

        # Generate initial final prediction
        initial_final_prediction = combine_frame_predictions(results)
        result['initial_prediction'] = initial_final_prediction

        # Apply further classification to get the final stroke name
        final_stroke_name = apply_further_classification(
            initial_final_prediction,
            output_folder,
            video_path  # Pass the video_path
        )
        logger.info(f"Final stroke name: {final_stroke_name}")
        result['final_stroke_name'] = final_stroke_name

        # Process feedback if the stroke is Front Foot Defence or Back Foot Defence
        feedback_data = []
        if "front foot defence" in final_stroke_name.lower():
            logger.info("Processing feedback for Front Foot Defence")
            try:
                feedback_data = process_forward_defence_feedback(highlights_folder, output_folder)
            except Exception as e:
                logger.error(f"Failed to process feedback: {e}")

        if final_stroke_name.lower() == "back foot defence":
            logger.info("Processing feedback for Back Foot Defence")
            try:
                feedback_data = process_backward_defence_feedback(highlights_folder, output_folder)
            except Exception as e:
                logger.error(f"Failed to process feedback: {e}")

        result['feedback_data'] = feedback_data

        # Use the new result writer module
        result.update(write_classification_results(
            result,
            results,
            top_frames,
            initial_final_prediction,
            final_stroke_name,
            dominant_hand,
            feedback_data,
            os.path.join(highlights_folder, "shot_classifications")
        ))

    except Exception as e:
        error_msg = f"Classification failed: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        result['error'] = error_msg
        return result

    return result