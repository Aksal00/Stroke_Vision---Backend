#shot_classifier.py

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from tkinter import messagebox

from FurtherClassification.drive_classifier import process_drive_classification


def load_shot_classifier_model(model_path):
    """
    Load the trained shot classification model.

    Args:
        model_path (str): Path to the h5 model file.

    Returns:
        model: Loaded Keras model, or None if loading failed.
    """
    try:
        model = keras.models.load_model(model_path)
        print(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        messagebox.showerror("Model Error", f"Failed to load classification model: {e}")
        return None


def predict_top_shots(model, img_path, class_names, img_size=224, top_n=3):
    """
    Make prediction on a single image and return top N predictions with confidence scores.

    Args:
        model: Trained Keras model
        img_path (str): Path to image file
        class_names (list): List of class names (in order)
        img_size (int): Target size for resizing (must match model input)
        top_n (int): Number of top predictions to return

    Returns:
        list: List of tuples (class_name, confidence_score) sorted by confidence
    """
    try:
        # Load and preprocess image
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Could not read image at {img_path}")

        # Preprocess identical to training pipeline
        img = cv2.resize(img, (img_size, img_size))
        img = img.astype('float32') / 255.0  # Normalize [0,1]
        img = np.expand_dims(img, axis=0)  # Add batch dimension

        # Make prediction
        preds = model.predict(img, verbose=0)[0]

        # Get top N predictions
        top_indices = np.argsort(preds)[-top_n:][::-1]
        top_predictions = [(class_names[i], preds[i]) for i in top_indices]

        return top_predictions

    except Exception as e:
        print(f"Error predicting image {img_path}: {e}")
        return [("Error", 0.0)]


def visualize_top_predictions(img_path, top_predictions, save_path=None):
    """
    Visualize the top predictions with the image.

    Args:
        img_path (str): Path to the input image
        top_predictions (list): List of (class, confidence) tuples
        save_path (str, optional): Path to save the visualization. If None, only displays.
    """
    # Load and convert image for display
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(10, 8))
    plt.imshow(img_rgb)

    # Create prediction text
    pred_text = "Top Predictions:\n"
    for i, (pred_class, confidence) in enumerate(top_predictions, 1):
        pred_text += f"{i}. {pred_class}: {confidence:.2%}\n"

    # Calculate weighted prediction
    weighted_pred = calculate_weighted_prediction(top_predictions)

    plt.title(f"{pred_text}\nWeighted Prediction: {weighted_pred}", fontsize=12)
    plt.axis('off')

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")

    plt.show()


def calculate_weighted_prediction(top_predictions):
    """
    Calculate weighted prediction based on confidence scores.

    Args:
        top_predictions (list): List of (class, confidence) tuples

    Returns:
        str: Weighted prediction result
    """
    # Create a dictionary to sum confidence scores for each class
    class_scores = {}

    for pred_class, confidence in top_predictions:
        if pred_class in class_scores:
            class_scores[pred_class] += confidence
        else:
            class_scores[pred_class] = confidence

    # Get the class with highest total confidence
    weighted_pred = max(class_scores.items(), key=lambda x: x[1])

    return f"{weighted_pred[0]} ({weighted_pred[1] / len(top_predictions):.2%} avg)"


def combine_frame_predictions(results):
    """
    Combine predictions from all three frames to make a final prediction.

    Args:
        results (dict): Dictionary with frame paths as keys and (top_predictions, weighted_pred) as values

    Returns:
        str: Final combined prediction or message if unclear
    """
    if not results:
        return "No frames available for prediction"

    # Count votes from all frames
    vote_counts = {}
    confidence_sums = {}

    for frame_path, (top_predictions, _) in results.items():
        # Get the top prediction for this frame
        if top_predictions and top_predictions[0][0] != "Error":
            top_class = top_predictions[0][0]
            vote_counts[top_class] = vote_counts.get(top_class, 0) + 1
            confidence_sums[top_class] = confidence_sums.get(top_class, 0) + top_predictions[0][1]

    if not vote_counts:
        return "No valid predictions available"

    # Find the class with most votes
    max_votes = max(vote_counts.values())
    winning_classes = [cls for cls, votes in vote_counts.items() if votes == max_votes]

    if len(winning_classes) > 1:
        # If tie, check confidence sums
        max_confidence = max(confidence_sums[cls] for cls in winning_classes)
        confident_classes = [cls for cls in winning_classes if confidence_sums[cls] == max_confidence]

        if len(confident_classes) > 1:
            return "Stroke cannot be clearly identified (multiple possibilities)"
        return f"Likely {confident_classes[0]} (tie broken by confidence)"

    # Calculate average confidence for the winning class
    avg_confidence = confidence_sums[winning_classes[0]] / vote_counts[winning_classes[0]]
    return f"Final Prediction: {winning_classes[0]} (confidence: {avg_confidence:.2%})"


def classify_highlight_frames(highlights_folder, model_path, output_folder):
    """
    Classify the three exported frames from the highlight clips and apply further classification if needed.
    Now supports further classification for drive, leg glance/flick, and pull shot.
    """
    # Define class names in the same order as training
    class_names = ['drive', 'legglance-flick', 'pullshot', 'sweep']
    print(f"Class names: {class_names}")

    # Load the model
    model = load_shot_classifier_model(model_path)
    if model is None:
        return {}

    # Find all jpg files in the highlights folder
    frame_files = [f for f in os.listdir(highlights_folder) if f.endswith('.jpg')]

    if not frame_files:
        messagebox.showinfo("Info", "No frame images found in the highlights folder.")
        return {}

    # Output folder for visualizations
    results_folder = os.path.join(highlights_folder, "shot_classifications")
    os.makedirs(results_folder, exist_ok=True)

    # Classify each frame
    results = {}
    for frame_file in frame_files:
        frame_path = os.path.join(highlights_folder, frame_file)
        top_predictions = predict_top_shots(model, frame_path, class_names, top_n=3)

        # Calculate weighted prediction
        weighted_pred = calculate_weighted_prediction(top_predictions)
        results[frame_path] = (top_predictions, weighted_pred)

        # Save visualization
        viz_path = os.path.join(results_folder, f"classified_{frame_file}")
        visualize_top_predictions(frame_path, top_predictions, viz_path)

    # Create a summary report
    report_path = os.path.join(results_folder, "classification_summary.txt")
    final_prediction = combine_frame_predictions(results)

    # Find the highlight clip for further classification
    highlight_clips = [f for f in os.listdir(highlights_folder)
                       if f.endswith('.mp4') and 'highest_peak' in f]

    # If not found, check parent folder
    if not highlight_clips:
        parent_folder = os.path.dirname(highlights_folder)
        highlight_clips = [f for f in os.listdir(parent_folder)
                           if f.endswith('.mp4') and 'highest_peak' in f]
        if highlight_clips:
            clip_path = os.path.join(parent_folder, highlight_clips[0])
        else:
            # Check output_folder as final fallback
            highlight_clips = [f for f in os.listdir(output_folder)
                               if f.endswith('.mp4') and 'highest_peak' in f]
            if highlight_clips:
                clip_path = os.path.join(output_folder, highlight_clips[0])
            else:
                clip_path = None
                final_prediction += "\n(Could not find highlight video clip for further classification)"
    else:
        clip_path = os.path.join(highlights_folder, highlight_clips[0])

    # Apply further classification if clip was found
    if clip_path:
        try:
            # DRIVE CLASSIFICATION
            if "drive" in final_prediction.lower():
                drive_result = process_drive_classification(clip_path, output_folder)
                final_prediction += f"\nFurther Classification: {drive_result}"

            # LEG GLANCE/FLICK CLASSIFICATION
            elif "legglance-flick" in final_prediction.lower():
                from FurtherClassification.legglance_classifier import process_leg_glance_classification
                glance_result = process_leg_glance_classification(clip_path, output_folder)
                final_prediction += f"\nFurther Classification: {glance_result}"

            # PULL SHOT CLASSIFICATION
            elif "pullshot" in final_prediction.lower():
                from FurtherClassification.pullshot_classifier import process_pullshot_classification
                pullshot_result = process_pullshot_classification(clip_path, output_folder)
                final_prediction += f"\nFurther Classification: {pullshot_result}"

            else:
                print("No further classification needed for this shot type")

        except Exception as e:
            final_prediction += f"\n(Error during further classification: {e})"

    # Update the results message to include the final prediction
    results_msg = "\n\n".join([
        f"{os.path.basename(path)}:\n" +
        "\n".join([f"{i}. {cls} ({conf:.2%})" for i, (cls, conf) in enumerate(top_preds, 1)]) +
        f"\nWeighted: {weighted_pred}"
        for path, (top_preds, weighted_pred) in results.items()
    ])

    results_msg += f"\n\n=== FINAL PREDICTION ===\n{final_prediction}"

    # Update the report file to include the final prediction
    with open(report_path, 'w') as f:
        f.write("=== FRAME-BY-FRAME PREDICTIONS ===\n")
        for path, (top_preds, weighted_pred) in results.items():
            f.write(f"\n{os.path.basename(path)}:\n")
            for i, (cls, conf) in enumerate(top_preds, 1):
                f.write(f"{i}. {cls} ({conf:.2%})\n")
            f.write(f"Weighted: {weighted_pred}\n")

        f.write("\n=== FINAL COMBINED PREDICTION ===\n")
        f.write(f"{final_prediction}\n")

    messagebox.showinfo(
        "Shot Classification Results",
        f"Classification complete!\n\n{results_msg}\n\nDetailed results saved to:\n{results_folder}"
    )

    return results