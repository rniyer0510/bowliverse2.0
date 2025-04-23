import json
import numpy as np
import logging
from core.frame_selection import compute_elbow_angle

logging.basicConfig(level=logging.INFO)

def prepare_frame_data(keypoints_dir, assessments, action_type, config=None, pitch_refs=None):
    """
    Prepare frame classification data for FrameDetector training.
    Args:
        keypoints_dir: Directory with keypoint JSONs.
        assessments: Dict of video assessments.
        action_type: 'fast' or 'spin'.
        config: Configuration parameters.
        pitch_refs: Dict of video_id to pitch reference.
    Returns:
        Tuple of (X, y) for training.
    """
    config = config or {}
    pitch_refs = pitch_refs or {}
    
    X = []
    y = []
    
    for video_id, labels in assessments.items():
        keypoints_path = os.path.join(keypoints_dir, f"bowling_analysis_{video_id}.json")
        if not os.path.exists(keypoints_path):
            logging.warning(f"Missing keypoints for {video_id}")
            continue
        
        try:
            with open(keypoints_path, 'r') as f:
                keypoints = json.load(f)
        except Exception as e:
            logging.error(f"Failed to load keypoints for {video_id}: {e}")
            continue
        
        pitch_ref = pitch_refs.get(video_id, {"pitch_angle": 0})
        
        for frame_type, frame_idx in labels.items():
            if frame_idx >= len(keypoints) or frame_idx < 0 or not keypoints[frame_idx]:
                continue
            
            kp = keypoints[frame_idx].get("keypoints", {})
            if not kp:
                continue
            
            features = [
                kp.get("landmark_11", {"x": 0, "y": 0})["x"],
                kp.get("landmark_11", {"x": 0, "y": 0})["y"],
                kp.get("landmark_13", {"x": 0, "y": 0})["x"],
                kp.get("landmark_13", {"x": 0, "y": 0})["y"],
                kp.get("landmark_14", {"x": 0, "y": 0})["x"],
                kp.get("landmark_14", {"x": 0, "y": 0})["y"],
                kp.get("landmark_27", {"x": 0, "y": 0})["x"],
                kp.get("landmark_27", {"x": 0, "y": 0})["y"],
                kp.get("landmark_31", {"x": 0, "y": 0})["x"],
                kp.get("landmark_31", {"x": 0, "y": 0})["y"]
            ]
            
            elbow_angle = compute_elbow_angle({"keypoints": kp}, config)
            features.append(elbow_angle)
            
            label_map = {
                "bfc_frame": 1,
                "ffc_frame": 2,
                "uah_frame": 3,
                "release_frame": 4
            }
            label = label_map.get(frame_type, 0)
            
            if label > 0:
                X.append(features)
                y.append(label)
    
    return np.array(X), np.array(y)
