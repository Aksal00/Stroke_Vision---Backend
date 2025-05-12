import os
import numpy as np
import cv2
import logging
from typing import List, Tuple, Dict, Any, Optional
import keras
from Feedback.ForwardDefence.ForwardDefence import processfeedback

# Set up logging
logger = logging.getLogger(__name__)

# Model caching
_model_cache = None


def load_shot_classifier_model(model_path: str) -> Optional[keras.Model]:
    """
    Load and cache the shot classification model
    Args:
        model_path: Path to the trained model file
    Returns:
        Loaded Keras model or None if failed
    """
    global _model_cache

    if _model_cache is not None:
        print(f"[INFO] Using cached model")
        return _model_cache

    try:
        print(f"[INFO] Loading model from {model_path}")
        logger.info(f"Loading model from {model_path}")
        _model_cache = keras.models.load_model(model_path)
        print(f"[SUCCESS] Model loaded successfully")
        return _model_cache
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        print(f"[ERROR] Failed to load model: {e}")
        return None


def predict_top_shots(
        model: keras.Model,
        img_path: str,
        class_names: List[str],
        img_size: int = 224,
        top_n: int = 3
) -> List[Tuple[str, float]]:
    """
    Predict top shot classes from an image
    Args:
        model: Loaded Keras model
        img_path: Path to input image
        class_names: List of class names
        img_size: Target image size
        top_n: Number of top predictions to return
    Returns:
        List of (class_name, confidence) tuples
    """
    try:
        # Load and preprocess image
        print(f"[INFO] Processing image: {os.path.basename(img_path)}")
        img = cv2.imread(img_path)
        if img is None:
            print(f"[ERROR] Could not read image at {img_path}")
            raise ValueError(f"Could not read image at {img_path}")

        img = cv2.resize(img, (img_size, img_size))
        img = img.astype('float32') / 255.0
        img = np.expand_dims(img, axis=0)

        # Make prediction
        print(f"[INFO] Running prediction on {os.path.basename(img_path)}")
        preds = model.predict(img, verbose=0)[0]
        top_indices = np.argsort(preds)[-top_n:][::-1]
        results = [(class_names[i], float(preds[i])) for i in top_indices]
        print(f"[RESULT] Top prediction: {results[0][0]} ({results[0][1]:.2%})")
        return results

    except Exception as e:
        logger.error(f"Error predicting image {img_path}: {e}")
        print(f"[ERROR] Failed to predict image {img_path}: {e}")
        return [("Error", 0.0)]


def calculate_weighted_prediction(top_predictions: List[Tuple[str, float]]) -> str:
    """
    Calculate weighted prediction from multiple predictions
    Args:
        top_predictions: List of (class, confidence) tuples
    Returns:
        Formatted prediction string
    """
    print(f"[INFO] Calculating weighted prediction from {len(top_predictions)} predictions")
    class_scores = {}
    for pred_class, confidence in top_predictions:
        class_scores[pred_class] = class_scores.get(pred_class, 0.0) + confidence

    if not class_scores:
        print(f"[WARNING] No predictions available for weighted calculation")
        return "No predictions available"

    weighted_pred = max(class_scores.items(), key=lambda x: x[1])
    avg_confidence = weighted_pred[1] / len(top_predictions)
    result = f"{weighted_pred[0]} ({avg_confidence:.2%} avg)"
    print(f"[RESULT] Weighted prediction: {result}")
    return result


def combine_frame_predictions(results: Dict[str, Any]) -> str:
    """
    Combine predictions from multiple frames
    Args:
        results: Dictionary of frame predictions
    Returns:
        Combined prediction string
    """
    print(f"[INFO] Combining predictions from {len(results)} frames")
    if not results:
        print(f"[WARNING] No frames available for prediction")
        return "No frames available for prediction"

    vote_counts = {}
    confidence_sums = {}

    for frame_path, (top_predictions, _) in results.items():
        if top_predictions and top_predictions[0][0] != "Error":
            top_class = top_predictions[0][0]
            vote_counts[top_class] = vote_counts.get(top_class, 0) + 1
            confidence_sums[top_class] = confidence_sums.get(top_class, 0.0) + top_predictions[0][1]

    if not vote_counts:
        print(f"[WARNING] No valid predictions available")
        return "No valid predictions available"

    print(f"[INFO] Vote counts: {vote_counts}")
    print(f"[INFO] Confidence sums: {confidence_sums}")

    max_votes = max(vote_counts.values())
    winning_classes = [cls for cls, votes in vote_counts.items() if votes == max_votes]
    print(f"[INFO] Classes with max votes ({max_votes}): {winning_classes}")

    if len(winning_classes) > 1:
        print(f"[INFO] Multiple classes tie for votes, breaking tie by confidence")
        max_confidence = max(confidence_sums[cls] for cls in winning_classes)
        confident_classes = [cls for cls in winning_classes if confidence_sums[cls] == max_confidence]

        if len(confident_classes) > 1:
            print(f"[INFO] Multiple classes tie for confidence as well")
            return "Stroke cannot be clearly identified (multiple possibilities)"
        print(f"[INFO] Tie broken by confidence, winner: {confident_classes[0]}")
        return f"Likely {confident_classes[0]} (tie broken by confidence)"

    avg_confidence = confidence_sums[winning_classes[0]] / vote_counts[winning_classes[0]]
    final_prediction = f"Final Prediction: {winning_classes[0]} (confidence: {avg_confidence:.2%})"
    print(f"[RESULT] {final_prediction}")
    return final_prediction


def find_highlight_clip(highlights_folder: str, output_folder: str) -> Optional[str]:
    """
    Find the highlight clip in the highlights folder or parent directories
    Args:
        highlights_folder: Path to the highlights folder
        output_folder: Root output folder
    Returns:
        Path to the highlight clip or None if not found
    """
    print(f"[INFO] Searching for highlight clip in {highlights_folder}")
    # First check in highlights folder
    highlight_clips = [f for f in os.listdir(highlights_folder)
                       if f.endswith('.mp4') and 'highest_peak' in f]

    # If not found, check parent folder
    if not highlight_clips:
        print(f"[INFO] No highlight clip in main folder, checking parent folder")
        parent_folder = os.path.dirname(highlights_folder)
        highlight_clips = [f for f in os.listdir(parent_folder)
                           if f.endswith('.mp4') and 'highest_peak' in f]
        if highlight_clips:
            result = os.path.join(parent_folder, highlight_clips[0])
            print(f"[INFO] Found highlight clip in parent folder: {os.path.basename(result)}")
            return result
        else:
            print(f"[INFO] Checking output folder as fallback")
            # Check output_folder as final fallback
            highlight_clips = [f for f in os.listdir(output_folder)
                               if f.endswith('.mp4') and 'highest_peak' in f]
            if highlight_clips:
                result = os.path.join(output_folder, highlight_clips[0])
                print(f"[INFO] Found highlight clip in output folder: {os.path.basename(result)}")
                return result
            print(f"[WARNING] No highlight clip found")

    if highlight_clips:
        result = os.path.join(highlights_folder, highlight_clips[0])
        print(f"[INFO] Found highlight clip: {os.path.basename(result)}")
        return result
    else:
        print(f"[WARNING] No highlight clip found after searching all locations")
        return None


def apply_further_classification(clip_path: str, final_prediction: str, output_folder: str) -> str:
    """
    Apply further classification based on the initial prediction
    Args:
        clip_path: Path to the highlight video clip
        final_prediction: The initial final prediction
        output_folder: Root output folder
    Returns:
        The further classification result (just the stroke name)
    """
    print(f"[INFO] Starting further classification process")
    print(f"[INFO] Initial prediction: {final_prediction}")

    if not clip_path:
        # Extract base stroke name with the same improved logic
        if "Final Prediction:" in final_prediction:
            parts = final_prediction.split("Final Prediction:")
            prediction_part = parts[1].strip()
            base_stroke = prediction_part.split("(")[0].strip()
        else:
            # Fallback case if format is different
            base_stroke = final_prediction.split(':')[-1].strip().split('(')[0].strip()
        # Capitalize first letter before returning
        base_stroke = base_stroke[0].upper() + base_stroke[1:].lower()
        print(f"[WARNING] No clip path available for further classification")
        print(f"[RESULT] Using base stroke without further classification: {base_stroke}")
        return base_stroke

    try:
        # Extract base stroke name
        # Fix the extraction logic to properly get the stroke name
        if "Final Prediction:" in final_prediction:
            parts = final_prediction.split("Final Prediction:")
            prediction_part = parts[1].strip()
            base_stroke = prediction_part.split("(")[0].strip().lower()
        else:
            # Fallback case if format is different
            base_stroke = final_prediction.split(':')[-1].strip().split('(')[0].strip().lower()
        print(f"[INFO] Base stroke extracted: {base_stroke}")

        # DRIVE CLASSIFICATION
        if "drive" in base_stroke:
            print(f"[INFO] Detected drive shot, starting drive classification")
            from FurtherClassification.drive_classifier import process_drive_classification
            result = process_drive_classification(clip_path, output_folder)
            print(f"[RESULT] Drive classification result: {result}")
            return result

        # LEG GLANCE/FLICK CLASSIFICATION
        elif "legglance-flick" in base_stroke:
            print(f"[INFO] Detected leg glance/flick shot, starting specialized classification")
            from FurtherClassification.legglance_classifier import process_leg_glance_classification
            result = process_leg_glance_classification(clip_path, output_folder)
            print(f"[RESULT] Leg glance classification result: {result}")
            return result

        # PULL SHOT CLASSIFICATION
        elif "pullshot" in base_stroke:
            print(f"[INFO] Detected pull shot, starting pull shot classification")
            from FurtherClassification.pullshot_classifier import process_pullshot_classification
            result = process_pullshot_classification(clip_path, output_folder)
            print(f"[RESULT] Pull shot classification result: {result}")
            return result

        # For other strokes, return the base stroke name with first letter capitalized
        base_stroke = base_stroke[0].upper() + base_stroke[1:].lower()
        print(f"[INFO] No specialized classifier for {base_stroke}, using base class")
        return base_stroke

    except Exception as e:
        logger.error(f"Error during further classification: {e}")
        print(f"[ERROR] Failed during further classification: {e}")
        # Capitalize first letter before returning
        base_stroke = base_stroke[0].upper() + base_stroke[1:].lower()
        print(f"[FALLBACK] Using base stroke name: {base_stroke}")
        return base_stroke


def classify_highlight_frames(
        highlights_folder: str,
        model_path: str,
        output_folder: str
) -> Dict[str, Any]:
    """
    Classify frames from highlight clips and generate results
    Args:
        highlights_folder: Path to folder containing highlight frames
        model_path: Path to classification model
        output_folder: Root output folder
    Returns:
        Dictionary containing classification results
    """
    print(f"\n[INFO] ===== STARTING SHOT CLASSIFICATION PROCESS =====")
    print(f"[INFO] Highlights folder: {highlights_folder}")
    print(f"[INFO] Model path: {model_path}")

    # Define class names
    class_names = ['drive', 'legglance-flick', 'pullshot', 'sweep', 'frontfootdefence']  # Added frontfootdefence
    logger.info(f"Classifying with classes: {class_names}")
    print(f"[INFO] Using classes: {class_names}")

    # Load model
    print(f"[INFO] Loading classification model")
    model = load_shot_classifier_model(model_path)
    if model is None:
        print(f"[ERROR] Failed to load model, aborting classification")
        return {}

    # Find frame images
    print(f"[INFO] Looking for frame images in {highlights_folder}")
    frame_files = [f for f in os.listdir(highlights_folder) if f.lower().endswith('.jpg')]
    if not frame_files:
        logger.warning("No frame images found in highlights folder")
        print(f"[WARNING] No frame images found in highlights folder")
        return {}
    print(f"[INFO] Found {len(frame_files)} frame images")

    # Prepare output folders
    results_folder = os.path.join(highlights_folder, "shot_classifications")
    os.makedirs(results_folder, exist_ok=True)
    print(f"[INFO] Results will be saved to {results_folder}")

    # Classify each frame
    print(f"[INFO] Starting frame-by-frame classification")
    results = {}
    for frame_file in frame_files:
        print(f"\n[INFO] Processing frame: {frame_file}")
        frame_path = os.path.join(highlights_folder, frame_file)
        top_predictions = predict_top_shots(model, frame_path, class_names, top_n=3)
        weighted_pred = calculate_weighted_prediction(top_predictions)
        results[frame_path] = (top_predictions, weighted_pred)

        # Save visualization (optional)
        viz_path = os.path.join(results_folder, f"classified_{frame_file}")
        try:
            img = cv2.imread(frame_path)
            if img is not None:
                cv2.imwrite(viz_path, img)
                print(f"[INFO] Saved visualization to {os.path.basename(viz_path)}")
        except Exception as e:
            logger.warning(f"Could not save visualization for {frame_file}: {e}")
            print(f"[WARNING] Could not save visualization for {frame_file}: {e}")

    # Generate initial final prediction
    print(f"\n[INFO] Generating initial prediction from all frames")
    initial_final_prediction = combine_frame_predictions(results)

    # Find highlight clip for further classification
    print(f"\n[INFO] Looking for highlight clip for further classification")
    clip_path = find_highlight_clip(highlights_folder, output_folder)

    # Apply further classification to get the final stroke name
    print(f"\n[INFO] Starting further classification to determine final stroke name")
    final_stroke_name = apply_further_classification(clip_path, initial_final_prediction, output_folder)
    print(f"\n[FINAL RESULT] Final stroke name: {final_stroke_name}")

    # Process feedback if the stroke is Front Foot Defence
    feedback_data = []
    if final_stroke_name.lower() == "front foot defence":
        print(f"[INFO] Processing feedback for Front Foot Defence")
        try:
            feedback_data = processfeedback(highlights_folder, output_folder)
            print(f"[INFO] Generated {len(feedback_data)} feedback items")
            print(f"Feedback data: {feedback_data}")
        except Exception as e:
            print(f"[ERROR] Failed to process feedback: {e}")

    # Save results to file
    print(f"[INFO] Saving comprehensive results to file")
    report_path = os.path.join(results_folder, "classification_summary.txt")
    with open(report_path, 'w') as f:
        f.write("=== FRAME-BY-FRAME PREDICTIONS ===\n")
        for path, (top_preds, weighted_pred) in results.items():
            f.write(f"\n{os.path.basename(path)}:\n")
            for i, (cls, conf) in enumerate(top_preds, 1):
                f.write(f"{i}. {cls} ({conf:.2%})\n")
            f.write(f"Weighted: {weighted_pred}\n")

        f.write("\n=== INITIAL FINAL PREDICTION ===\n")
        f.write(f"{initial_final_prediction}\n")

        f.write("\n=== FINAL STROKE NAME ===\n")
        f.write(f"{final_stroke_name}\n")

        if feedback_data:
            f.write("\n=== FEEDBACK ===\n")
            for item in feedback_data:
                f.write(f"\nTitle: {item['title']}\n")
                f.write(f"Image URL: {item['image_url']}\n")
                f.write(f"Feedback: {item['feedback_text']}\n")

    print(f"[INFO] Classification complete. Results saved to {report_path}")
    print(f"[INFO] ===== SHOT CLASSIFICATION PROCESS COMPLETE =====\n")

    logger.info(f"Classification complete. Results saved to {results_folder}")
    return {
        'frame_predictions': results,
        'initial_prediction': initial_final_prediction,
        'final_stroke_name': final_stroke_name,
        'feedback_data': feedback_data,
        'report_path': report_path,
        'highlight_clip_path': clip_path
    }