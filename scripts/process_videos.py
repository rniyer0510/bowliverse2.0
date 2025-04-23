# scripts/process_videos.py
import os
import sys
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.keypoints_utils import extract_keypoints
from core.pitch_calibrator import extract_pitch_reference

logging.basicConfig(level=logging.INFO)

def process_videos(video_dir, action_type="fast"):
    """Process videos to extract keypoints and pitch references."""
    if not os.path.exists(video_dir):
        logging.error(f"Video directory {video_dir} does not exist")
        sys.exit(1)
    output_dir = video_dir
    for filename in os.listdir(video_dir):
        if filename.startswith(f"{action_type}_") and filename.endswith(".mp4"):
            video_id = filename[len(action_type)+1:-4]
            video_path = os.path.join(video_dir, filename)
            keypoints_json = os.path.join(output_dir, f"bowling_analysis_{video_id}.json")
            pitch_json = os.path.join(output_dir, f"pitch_reference_{video_id}.json")
            logging.info(f"Processing {video_id}")
            # Generate keypoints first
            extract_keypoints(video_path, keypoints_json, None)  # No pitch correction yet
            # Then pitch reference
            if not os.path.exists(pitch_json):
                extract_pitch_reference(video_path, keypoints_json, pitch_json, action_type)
            # Reprocess keypoints with pitch correction
            if os.path.exists(pitch_json):
                extract_keypoints(video_path, keypoints_json, pitch_json)
            else:
                logging.warning(f"No pitch file for {video_id}, skipping correction")
    logging.info("Video processing complete")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python process_videos.py <video_dir> [action_type]")
        sys.exit(1)
    process_videos(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "fast")
