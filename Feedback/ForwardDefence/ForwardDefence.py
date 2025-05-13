import os
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
import mediapipe as mp
from Feedback.ForwardDefence.Positions.BackfootToe import process_backfoot_position
from Feedback.ForwardDefence.Positions.BackfootKnee import process_backfoot_knee_position
from Feedback.ForwardDefence.Positions.FrontfootKnee import process_frontfoot_knee_position
from Feedback.ForwardDefence.Positions.Hand import process_hand_positions

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
        # Process backfoot toe position feedback
        backfoot_toe_feedback = process_backfoot_position(
            highlights_folder,
            frame_files,
            pose,
            FEEDBACK_IMAGES_DIR
        )

        if backfoot_toe_feedback:
            # Convert image filename to web URL
            backfoot_toe_feedback["image_url"] = f"/feedback-images/{backfoot_toe_feedback.pop('image_filename')}"
            feedback_data.append(backfoot_toe_feedback)

        # Process backfoot knee position feedback
        backfoot_knee_feedback = process_backfoot_knee_position(
            highlights_folder,
            frame_files,
            pose,
            FEEDBACK_IMAGES_DIR
        )

        if backfoot_knee_feedback:
            # Convert image filename to web URL
            backfoot_knee_feedback["image_url"] = f"/feedback-images/{backfoot_knee_feedback.pop('image_filename')}"
            feedback_data.append(backfoot_knee_feedback)

        # Process frontfoot knee position feedback
        frontfoot_knee_feedback = process_frontfoot_knee_position(
            highlights_folder,
            frame_files,
            pose,
            FEEDBACK_IMAGES_DIR
        )

        if frontfoot_knee_feedback:
            # Convert image filename to web URL
            frontfoot_knee_feedback["image_url"] = f"/feedback-images/{frontfoot_knee_feedback.pop('image_filename')}"
            feedback_data.append(frontfoot_knee_feedback)

        # Process hand position feedback
        hand_position_feedback = process_hand_positions(
            highlights_folder,
            frame_files,
            pose,
            FEEDBACK_IMAGES_DIR
        )

        if hand_position_feedback:
            # Convert image filename to web URL
            hand_position_feedback["image_url"] = f"/feedback-images/{hand_position_feedback.pop('image_filename')}"
            feedback_data.append(hand_position_feedback)

    # Sort feedback items by feedback_no to maintain consistent order
    feedback_data.sort(key=lambda x: x['feedback_no'])

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