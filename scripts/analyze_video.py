import os
import sys
import json
import pickle
import logging
import cv2
import numpy as np
from core.keypoints import smooth_keypoints
from core.biomechanics import analyze_biomechanics
from core.frame_selection import select_key_frames
from models.frame_detector import FrameDetector
from models.angle_adjuster import AngleAdjuster
from models.biomechanics_refiner import BiomechanicsRefiner

logging.basicConfig(level=logging.INFO)

def analyze_video(video_path, videos_dir, output_dir, hmm_path, action_type="fast"):
    """
    Analyze a bowling video and produce biomechanical assessment.
    Args:
        video_path: Path to the video file.
        videos_dir: Directory containing pitch reference and keypoints JSONs.
        output_dir: Directory to save the assessment JSON.
        hmm_path: Path to the trained HMM model.
        action_type: 'fast' or 'spin'.
    """
    # Load config
    config_path = os.path.join(os.path.dirname(__file__), "..", "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)

    # Extract video ID
    video_id = os.path.splitext(os.path.basename(video_path))[0].replace(f"{action_type}_", "")
    keypoints_path = os.path.join(videos_dir, f"bowling_analysis_{video_id}.json")
    pitch_ref_path = os.path.join(videos_dir, f"pitch_reference_{video_id}.json")

    # Load keypoints
    if not os.path.exists(keypoints_path):
        logging.error(f"Keypoints file not found: {keypoints_path}")
        sys.exit(1)
    with open(keypoints_path, 'r') as f:
        keypoints = json.load(f)

    # Smooth keypoints
    keypoints = smooth_keypoints(keypoints, window_size=config.get("smoothing_window", 3))

    # Load pitch reference
    pitch_ref = {"pitch_angle": 0, "crease_y": 0.8, "crease_direction": [1, 0]}
    if os.path.exists(pitch_ref_path):
        with open(pitch_ref_path, 'r') as f:
            pitch_ref = json.load(f)

    # Load models
    frame_detector_path = os.path.join(output_dir, f"frame_detector_{action_type}.pkl")
    angle_adjuster_path = os.path.join(output_dir, f"angle_adjuster_elbow_{action_type}.pkl")
    biomechanics_refiner_path = os.path.join(output_dir, f"biomechanics_refiner_{action_type}.pkl")

    try:
        with open(frame_detector_path, 'rb') as f:
            frame_detector = pickle.load(f)
        with open(angle_adjuster_path, 'rb') as f:
            angle_adjuster = pickle.load(f)
        with open(biomechanics_refiner_path, 'rb') as f:
            biomechanics_refiner = pickle.load(f)
        with open(hmm_path, 'rb') as f:
            hmm = pickle.load(f)
    except Exception as e:
        logging.error(f"Failed to load models: {e}")
        sys.exit(1)

    # Select key frames
    key_frames = select_key_frames(keypoints, frame_detector, action_type, config, pitch_ref)

    # Analyze biomechanics
    results = analyze_biomechanics(keypoints, key_frames, config, pitch_ref)

    # Refine action type
    action_type_pred = biomechanics_refiner.predict(keypoints, config, pitch_ref)
    results["alignment"]["action_type_pred"] = action_type_pred

    # Adjust angles
    for frame_type in ["bfc_frame", "ffc_frame", "uah_frame", "release_frame"]:
        frame_idx = key_frames.get(frame_type, 0)
        if frame_idx < len(keypoints):
            kp = keypoints[frame_idx]["keypoints"]
            features = [
                kp.get(f"landmark_{config.get('landmarks', {}).get('elbow_angle', {}).get('shoulder', 11)}", {"x": 0})["x"],
                kp.get(f"landmark_{config.get('landmarks', {}).get('elbow_angle', {}).get('shoulder', 11)}", {"y": 0})["y"],
                kp.get(f"landmark_{config.get('landmarks', {}).get('elbow_angle', {}).get('elbow', 13)}", {"x": 0})["x"],
                kp.get(f"landmark_{config.get('landmarks', {}).get('elbow_angle', {}).get('elbow', 13)}", {"y": 0})["y"],
                kp.get(f"landmark_{config.get('landmarks', {}).get('elbow_angle', {}).get('wrist', 14)}", {"x": 0})["x"],
                kp.get(f"landmark_{config.get('landmarks', {}).get('elbow_angle', {}).get('wrist', 14)}", {"y": 0})["y"]
            ]
            adjusted_angle = angle_adjuster.predict([features])[0]
            results["metrics"][f"{frame_type}_adjusted_angle"] = adjusted_angle

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"assessment_{video_id}.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    logging.info(f"Saved assessment to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 6:
        print("Usage: python analyze_video.py <video_path> <videos_dir> <output_dir> <hmm_path> <action_type>")
        sys.exit(1)
    analyze_video(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
