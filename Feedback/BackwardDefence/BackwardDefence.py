import os
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
import mediapipe as mp
from Feedback.BackwardDefence.Positions.BackfootPosition import process_backfoot_position
from Feedback.BackwardDefence.Positions.FrontfootPosition import process_frontfoot_position
from Feedback.BackwardDefence.Positions.Hand import process_hand_positions
from Feedback.BackwardDefence.Positions.Head import process_head_position
from Feedback.BackwardDefence.Positions.BottomArm import process_bottom_arm_bend
from Feedback.Backlift import process_backlift_angle
from Feedback.BackwardDefence.Positions.TopArm import process_top_arm_position
from Feedback.BackwardDefence.Positions.BackfootKnee import process_backfoot_knee_position
from Feedback.BackwardDefence.Positions.FrontfootKnee import process_frontfoot_knee_position

app = FastAPI()

# Configuration
FEEDBACK_IMAGES_DIR = "feedback-images"
REF_IMAGES_DIR = "ref-images"
os.makedirs(FEEDBACK_IMAGES_DIR, exist_ok=True)
os.makedirs(REF_IMAGES_DIR, exist_ok=True)

# Mount static files directories
app.mount("/feedback-images", StaticFiles(directory=FEEDBACK_IMAGES_DIR), name="feedback-images")
app.mount("/ref-images", StaticFiles(directory=REF_IMAGES_DIR), name="ref-images")


def process_backward_defence_feedback(highlights_folder: str) -> List[Dict[str, Any]]:
    """Process feedback for backward defence cricket shots using MediaPipe landmarks"""
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
        # Process all feedback types specific to backward defence
        feedback_processors = [
            (process_backfoot_position, "Backfoot Position"),
            (process_frontfoot_position, "Frontfoot Position"),
            (process_backfoot_knee_position, "Backfoot Knee Position"),  # Added
            (process_frontfoot_knee_position, "Frontfoot Knee Position"),  # Added
            (process_hand_positions, "Hand Position"),
            (process_head_position, "Head Position"),
            (process_bottom_arm_bend, "Bottom Arm Bend"),
            (process_backlift_angle, "Backlift Angle"),
            (process_top_arm_position, "Top Arm Position")
        ]

        for processor, name in feedback_processors:
            try:
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

                    feedback_data.append(feedback)
                    print(f"[INFO] Successfully processed {name}")
                else:
                    print(f"[WARNING] No feedback generated for {name}")

            except Exception as e:
                print(f"[ERROR] Failed to process {name}: {e}")
                continue

    # Sort feedback items by feedback_no
    feedback_data.sort(key=lambda x: x.get('feedback_no', 999))

    return feedback_data


@app.post("/analyze-backward-defence")
async def analyze_backward_defence(highlights_folder: str):
    """Analyze backward defence cricket shot and provide comprehensive feedback"""
    try:
        if not os.path.exists(highlights_folder):
            raise HTTPException(status_code=404, detail=f"Highlights folder not found: {highlights_folder}")

        feedback = process_backward_defence_feedback(highlights_folder)

        if not feedback:
            raise HTTPException(status_code=400, detail="No valid frames found for analysis")

        return {
            "shot_type": "Backward Defence",
            "total_feedback_items": len(feedback),
            "feedback": feedback
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "Backward Defence Analysis API is running"}


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Cricket Shot Analysis API",
        "shot_types": ["Backward Defence"],
        "endpoints": {
            "analyze": "/analyze-backward-defence",
            "health": "/health",
            "feedback_images": "/feedback-images/",
            "reference_images": "/ref-images/"
        }
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)