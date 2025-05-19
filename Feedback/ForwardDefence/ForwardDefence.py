import os
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
import mediapipe as mp
from Feedback.ForwardDefence.Positions.BackfootHeel import process_backfoot_heel_position
from Feedback.ForwardDefence.Positions.BackfootKnee import process_backfoot_knee_position
from Feedback.ForwardDefence.Positions.FrontfootKnee import process_frontfoot_knee_position
from Feedback.ForwardDefence.Positions.Hand import process_hand_positions
from Feedback.ForwardDefence.Positions.Head import process_head_position
from Feedback.ForwardDefence.Positions.BottomArm import process_bottom_arm_bend
from Feedback.Backlift import process_backlift_angle
from Feedback.ForwardDefence.Positions.TopArm import process_top_arm_position

app = FastAPI()

# Configuration
FEEDBACK_IMAGES_DIR = "feedback-images"
REF_IMAGES_DIR = "ref-images"
os.makedirs(FEEDBACK_IMAGES_DIR, exist_ok=True)
os.makedirs(REF_IMAGES_DIR, exist_ok=True)

# Mount static files directories
app.mount("/feedback-images", StaticFiles(directory=FEEDBACK_IMAGES_DIR), name="feedback-images")
app.mount("/ref-images", StaticFiles(directory=REF_IMAGES_DIR), name="ref-images")


def processfeedback(highlights_folder: str, shot_type: str = None) -> List[Dict[str, Any]]:
    """Process feedback for cricket shots using MediaPipe landmarks"""
    feedback_data = []

    frame_files = [f for f in os.listdir(highlights_folder)
                   if f.lower().endswith('.jpg') and 'frame_at_' in f]

    if not frame_files:
        return feedback_data

    is_left_handed = any('left' in f.lower() for f in frame_files)

    with mp.solutions.pose.Pose(
            static_image_mode=True,
            model_complexity=1,
            min_detection_confidence=0.5
    ) as pose:
        # Process all feedback types
        feedback_processors = [
            (process_backfoot_heel_position, "Backfoot Heel"),
            (process_backfoot_knee_position, "Backfoot Knee"),
            (process_frontfoot_knee_position, "Frontfoot Knee"),
            (process_hand_positions, "Hand Position"),
            (process_head_position, "Head Position"),
            (process_bottom_arm_bend, "Bottom Arm Bend"),
            (process_backlift_angle, "Backlift Angle"),
            (process_top_arm_position, "Top Arm Bend")
        ]

        for processor, name in feedback_processors:
            feedback = processor(
                highlights_folder,
                frame_files,
                pose,
                FEEDBACK_IMAGES_DIR,
                is_left_handed
            )

            if feedback:
                # Convert image filename to URL
                if feedback.get("image_filename"):
                    feedback["image_url"] = f"/feedback-images/{feedback.pop('image_filename')}"

                # Convert ref-images filenames to URLs
                if feedback.get("ref-images"):
                    feedback["ref-images"] = [f"/ref-images/{img}" for img in feedback["ref-images"]]
                elif feedback.get("ref-images"):  # Handle both formats
                    feedback["ref-images"] = [f"/ref-images/{img}" for img in feedback.pop("ref-images")]

                feedback_data.append(feedback)

    # Sort feedback items by feedback_no
    feedback_data.sort(key=lambda x: x['feedback_no'])
    return feedback_data


@app.post("/analyze-shot")
async def analyze_shot(highlights_folder: str, shot_type: str = None):
    try:
        feedback = processfeedback(highlights_folder, shot_type)
        return {"feedback": feedback}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))