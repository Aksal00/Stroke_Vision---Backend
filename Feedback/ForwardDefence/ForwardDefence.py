import os
from typing import List, Dict, Any, Optional
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


def detect_handedness_from_frames(frame_files: List[str]) -> bool:
    """Determine if batsman is left-handed based on frame filenames"""
    left_keywords = ['left', '_l_', 'left_']
    right_keywords = ['right', '_r_', 'right_']

    left_count = sum(1 for f in frame_files if any(kw in f.lower() for kw in left_keywords))
    right_count = sum(1 for f in frame_files if any(kw in f.lower() for kw in right_keywords))

    # If we have more left markers than right, assume left-handed
    return left_count > right_count


def get_frame_files(highlights_folder: str) -> Optional[List[str]]:
    """Get frame files from the highlights folder with support for multiple naming patterns"""
    # Look in extracted_frames subfolder first
    frames_dir = os.path.join(highlights_folder, "extracted_frames")
    if not os.path.exists(frames_dir):
        frames_dir = highlights_folder  # fallback to main folder

    # Support multiple naming patterns
    patterns = [
        'frame_at_',  # Original pattern
        'highlight_',  # New pattern
        'frame_',  # Alternative pattern
        'shot_'  # Another possible pattern
    ]

    frame_files = []
    for f in os.listdir(frames_dir):
        lower_f = f.lower()
        if (lower_f.endswith(('.jpg', '.jpeg', '.png')) and
                any(p in lower_f for p in patterns)):
            frame_files.append(f)

    print(f"[DEBUG] Found {len(frame_files)} frame files in {frames_dir}")
    return frame_files if frame_files else None


def process_forward_defence_feedback(highlights_folder: str, shot_type: str = None) -> List[Dict[str, Any]]:
    """Process feedback for cricket shots using MediaPipe landmarks"""
    feedback_data = []

    # Get the correct frames directory path
    frames_dir = os.path.join(highlights_folder, "extracted_frames")
    if not os.path.exists(frames_dir):
        frames_dir = highlights_folder  # fallback to main folder

    frame_files = get_frame_files(highlights_folder)
    if not frame_files:
        return feedback_data

    # Create full paths to the frame files
    frame_paths = [os.path.join(frames_dir, f) for f in frame_files]

    # Verify files exist before processing
    missing_files = [f for f in frame_paths if not os.path.exists(f)]
    if missing_files:
        print(f"[ERROR] Missing frame files: {missing_files}")
        return feedback_data

    is_left_handed = detect_handedness_from_frames(frame_files)

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
            try:
                feedback = processor(
                    frames_dir,  # Pass the directory containing the frames
                    frame_files,  # Pass just the filenames
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
            except Exception as e:
                print(f"[ERROR] Failed to process {name}: {str(e)}")
                continue

    # Sort feedback items by feedback_no
    feedback_data.sort(key=lambda x: x['feedback_no'])
    return feedback_data


@app.post("/analyze-shot")
async def analyze_shot(highlights_folder: str, shot_type: str = None):
    try:
        feedback = process_forward_defence_feedback(highlights_folder, shot_type)
        return {"feedback": feedback}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))