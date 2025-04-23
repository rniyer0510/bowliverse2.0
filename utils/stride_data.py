import numpy as np
import logging
import json
import os

logging.basicConfig(level=logging.INFO)

def prepare_stride_data(keypoints_dir, assessments, action_type, config=None, pitch_refs=None):
    """
    Prepare stride data for StridePredictor training.
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
        
        bfc_idx = labels.get("bfc_frame", 0)
        ffc_idx = labels.get("ffc_frame", 0)
        if bfc_idx >= len(keypoints) or ffc_idx >= len(keypoints) or not keypoints[bfc_idx] or not keypoints[ffc_idx]:
            continue
        
        bfc_kp = keypoints[bfc_idx].get("keypoints", {})
        ffc_kp = keypoints[ffc_idx].get("keypoints", {})
        
        features = []
        for i in range(33):
            lm = f"landmark_{i}"
            features.extend([
                bfc_kp.get(lm, {"x": 0, "y": 0})["x"],
                bfc_kp.get(lm, {"x": 0, "y": 0})["y"],
                ffc_kp.get(lm, {"x": 0, "y": 0})["x"],
                ffc_kp.get(lm, {"x": 0, "y": 0})["y"]
            ])
        
        # Estimate scale factor (simplified)
        scale_factor = abs(ffc_kp.get("landmark_27", {"y": 0})["y"] - bfc_kp.get("landmark_31", {"y": 0})["y"])
        features.append(scale_factor)
        
        stride_length = labels.get("stride_length", 0.0)
        if stride_length > 0:
            X.append(features)
            y.append(stride_length)
    
    return np.array(X), np.array(y)
