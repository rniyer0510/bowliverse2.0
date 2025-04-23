import os
import json
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)

def prepare_stride_data(keypoints_dir, assessments, action_type, pitch_angles=None):
    """
    Prepare stride data for training.
    Args:
        keypoints_dir: Directory with keypoint JSONs.
        assessments: Video assessments.
        action_type: Bowling action type.
        pitch_angles: Video_id to pitch angle map.
    Returns:
        X_stride, y_stride: Feature and stride arrays.
    """
    if pitch_angles is None:
        pitch_angles = {}
    X_stride = []
    y_stride = []
    for video_id in assessments:
        keypoints_path = os.path.join(keypoints_dir, f"bowling_analysis_{video_id}.json")
        if not os.path.exists(keypoints_path):
            continue
        with open(keypoints_path, 'r') as f:
            keypoints = json.load(f)
        if not keypoints or not isinstance(keypoints, list):
            logging.warning(f"Invalid keypoints for {video_id}")
            continue
        bfc_idx = assessments[video_id].get("bfc_frame", -1)
        ffc_idx = assessments[video_id].get("ffc_frame", -1)
        if bfc_idx != -1 and ffc_idx != -1 and bfc_idx < len(keypoints) and ffc_idx < len(keypoints):
            bfc_kp = keypoints[bfc_idx].get("keypoints", {})
            ffc_kp = keypoints[ffc_idx].get("keypoints", {})
            features = ([bfc_kp.get(f"landmark_{i}", {"x": 0, "y": 0})[k] for i in range(33) for k in ["x", "y"]] +
                       [ffc_kp.get(f"landmark_{i}", {"x": 0, "y": 0})[k] for i in range(33) for k in ["x", "y"]])
            X_stride.append(features[:132])
            y_stride.append(assessments[video_id].get("stride_length", 0))
    return np.array(X_stride), np.array(y_stride)
