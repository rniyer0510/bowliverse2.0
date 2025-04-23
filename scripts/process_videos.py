import os
import sys
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.keypoints_utils2 import extract_keypoints
from core.pitch_calibrator import extract_pitch_reference

logging.basicConfig(level=logging.INFO)

def process_videos(video_dir, action_type="fast"):
    """Process videos to extract 3D keypoints and pitch references."""
    if not os.path.exists(video_dir):
        logging.error(f"Video directory {video_dir} does not exist")
        sys.exit(1)
    output_dir = video_dir
    for filename in os.listdir(video_dir):
        if filename.startswith(f"{action_type}_") and filename.endswith(".mp4"):
            video_id = filename[len(action_type)+1:-4]
            video_path = os.path.join(output_dir, filename)
            keypoints_json = os.path.join(output_dir, f"bowling_analysis_{video_id}.json")
            pitch_json = os.path.join(output_dir, f"pitch_reference_{video_id}.json")
            logging.info(f"Processing {video_id} with utils.keypoints_utils2")
            # Generate 3D keypoints with pitch correction
            keypoints = extract_keypoints(
                video_path,
                keypoints_json,
                pitch_json if os.path.exists(pitch_json) else None,
                config={"min_detection_confidence": 0.6, "min_tracking_confidence": 0.6}
            )
            # Validate 3D keypoints
            if keypoints and any("z" in lm for frame in keypoints for lm in frame.get("keypoints", {}).values()):
                logging.info("Confirmed 3D keypoints with z-coordinates")
            else:
                logging.warning("No z-coordinates detected in keypoints")
            # Generate pitch reference if not exists
            if not os.path.exists(pitch_json):
                extract_pitch_reference(video_path, keypoints_json, pitch_json, action_type)
                # Reprocess keypoints with new pitch correction
                if os.path.exists(pitch_json):
                    keypoints = extract_keypoints(
                        video_path,
                        keypoints_json,
                        pitch_json,
                        config={"min_detection_confidence": 0.6, "min_tracking_confidence": 0.6}
                    )
                    if keypoints and any("z" in lm for frame in keypoints for lm in frame.get("keypoints", {}).values()):
                        logging.info("Confirmed pitch-corrected 3D keypoints with z-coordinates")
                    else:
                        logging.warning("No z-coordinates in pitch-corrected keypoints")
    logging.info("Video processing complete")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python process_videos.py <video_dir> [action_type]")
        sys.exit(1)
    process_videos(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "fast")
