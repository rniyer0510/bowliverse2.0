import os
import numpy as np
import logging
import json

logging.basicConfig(level=logging.INFO)

def prepare_frame_data(keypoints_dir, assessments, action_type, config=None, pitch_refs=None, keypoints=None):
    """
    Prepare data for FrameDetector training.
    Args:
        keypoints_dir: Directory with keypoint JSONs (optional if keypoints provided).
        assessments: Dict of video_id to assessment data.
        action_type: 'fast' or 'spin'.
        config: Configuration parameters.
        pitch_refs: Dict of video_id to pitch reference.
        keypoints: Optional list of keypoints for single video.
    Returns:
        Tuple of (X, y) for training.
    """
    config = config or {}
    pitch_refs = pitch_refs or {}
    X = []
    y = []
    
    if keypoints:
        # Process single video
        kp_flat = []
        for kp in keypoints:
            frame_kp = kp.get("keypoints", {})
            features = []
            for i in range(33):
                lm = f"landmark_{i}"
                features.extend([
                    frame_kp.get(lm, {"x": 0})["x"],
                    frame_kp.get(lm, {"y": 0})["y"],
                    frame_kp.get(lm, {"visibility": 0})["visibility"]
                ])
            kp_flat.append(features)
        X.extend(kp_flat)
        # Placeholder labels (requires actual labels)
        y.extend([0] * len(kp_flat))
        logging.warning("Using placeholder labels for single video")
    else:
        # Process assessments
        for video_id, labels in assessments.items():
            keypoints_path = os.path.join(keypoints_dir, f"bowling_analysis_{video_id}.json")
            try:
                with open(keypoints_path, 'r') as f:
                    keypoints = json.load(f)
            except Exception as e:
                logging.error(f"Failed to load keypoints for {video_id}: {e}")
                continue
            
            kp_flat = []
            frame_labels = []
            for i, kp in enumerate(keypoints):
                frame_kp = kp.get("keypoints", {})
                features = []
                for j in range(33):
                    lm = f"landmark_{j}"
                    features.extend([
                        frame_kp.get(lm, {"x": 0})["x"],
                        frame_kp.get(lm, {"y": 0})["y"],
                        frame_kp.get(lm, {"visibility": 0})["visibility"]
                    ])
                kp_flat.append(features)
                # Assign labels based on assessments
                label = 0
                if i == labels.get("bfc_frame"):
                    label = 1
                elif i == labels.get("ffc_frame"):
                    label = 2
                elif i == labels.get("uah_frame"):
                    label = 3
                elif i == labels.get("release_frame"):
                    label = 4
                frame_labels.append(label)
            
            X.extend(kp_flat)
            y.extend(frame_labels)
    
    if X and y:
        return np.array(X), np.array(y)
    logging.warning("No valid data for FrameDetector training")
    return np.array([]), np.array([])
