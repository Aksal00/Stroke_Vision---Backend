# result_writer.py
import os
import json
from typing import Dict, Any, List, Tuple

def write_classification_results(
        results: Dict[str, Any],
        frame_predictions: Dict[str, Any],
        top_frames: List[Tuple[str, float]],
        initial_final_prediction: str,
        final_stroke_name: str,
        dominant_hand: str,
        feedback_data: List[Dict[str, Any]],
        output_folder: str
) -> Dict[str, Any]:
    """Write classification results to a report file and return the result dictionary"""
    result = {
        'frame_predictions': frame_predictions,
        'initial_prediction': initial_final_prediction,
        'final_stroke_name': final_stroke_name,
        'dominant_hand': dominant_hand,
        'feedback_data': feedback_data,
        'report_path': None,
        'error': None
    }

    try:
        report_path = os.path.join(output_folder, "classification_summary.txt")
        with open(report_path, 'w') as f:
            f.write("=== FRAME-BY-FRAME PREDICTIONS (USING MEDIAPIPE KEYPOINTS) ===\n")
            for path, (top_preds, weighted_pred) in frame_predictions.items():
                f.write(f"\n{os.path.basename(path)}:\n")
                for i, (cls, conf) in enumerate(top_preds, 1):
                    f.write(f"{i}. {cls} ({conf:.2%})\n")
                f.write(f"Weighted: {weighted_pred}\n")

            f.write("\n=== TOP 3 FRAMES WITH LANDMARKS ===\n")
            for i, (frame_path, confidence) in enumerate(top_frames, 1):
                f.write(f"{i}. {os.path.basename(frame_path)} (confidence: {confidence:.2%})\n")

            f.write("\n=== INITIAL FINAL PREDICTION ===\n")
            f.write(f"{initial_final_prediction}\n")

            f.write("\n=== FINAL STROKE NAME ===\n")
            f.write(f"{final_stroke_name}\n")

            f.write("\n=== DOMINANT HAND ===\n")
            f.write(f"{dominant_hand}\n")

            if feedback_data:
                f.write("\n=== FEEDBACK DATA ===\n")
                f.write("Feedback data: " + json.dumps(feedback_data, indent=2) + "\n")

        result['report_path'] = report_path
        return result

    except Exception as e:
        error_msg = f"Failed to write report file: {e}"
        result['error'] = error_msg
        return result