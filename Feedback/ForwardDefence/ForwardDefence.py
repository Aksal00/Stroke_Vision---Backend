import os
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
import mediapipe as mp
from Feedback.ForwardDefence.Positions.BackfootAnalysis import process_backfoot_position

app = FastAPI()

# Configuration
FEEDBACK_IMAGES_DIR = "feedback-images"
os.makedirs(FEEDBACK_IMAGES_DIR, exist_ok=True)

# Mount static files directory to serve feedback images
app.mount("/feedback-images", StaticFiles(directory=FEEDBACK_IMAGES_DIR), name="feedback-images")


def processfeedback(highlights_folder: str, shot_type: str = None) -> List[Dict[str, Any]]:
    """
    Process feedback for cricket shots using MediaPipe landmarks
    Args:
        highlights_folder: Path to folder containing highlight frames
        shot_type: Type of cricket shot (optional, defaults to None)
    Returns:
        List of feedback items with web-accessible image URLs
    """
    feedback_data = []

    # Get all frame images from highlights folder
    frame_files = [f for f in os.listdir(highlights_folder)
                   if f.lower().endswith('.jpg') and 'frame_at_' in f]

    if not frame_files:
        return feedback_data

    # Initialize MediaPipe Pose
    with mp.solutions.pose.Pose(
            static_image_mode=True,
            model_complexity=1,
            min_detection_confidence=0.5
    ) as pose:
        # Process backfoot position feedback
        backfoot_feedback = process_backfoot_position(
            highlights_folder,
            frame_files,
            pose,
            FEEDBACK_IMAGES_DIR
        )

        if backfoot_feedback:
            # Convert image filename to web URL
            backfoot_feedback["image_url"] = f"/feedback-images/{backfoot_feedback.pop('image_filename')}"
            feedback_data.append(backfoot_feedback)

        # Here you can add more body position analyses as needed
        # For example:
        # head_position_feedback = process_head_position(...)
        # if head_position_feedback:
        #     feedback_data.append(head_position_feedback)

    return feedback_data


# API Endpoint
@app.post("/analyze-shot")
async def analyze_shot(highlights_folder: str, shot_type: str = None):
    try:
        # Pass both parameters to processfeedback
        feedback = processfeedback(highlights_folder, shot_type)
        return {"feedback": feedback}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))