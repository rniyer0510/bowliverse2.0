import json
import numpy as np
import logging
from utils.angle_utils import compute_elbow_angle, compute_wrist_fallback_angle

logging.basicConfig(level=logging.INFO)

def extract_features(keypoints_data, action_type, pitch_angle=0, config=None):
    """
    Extract features for HMM training or prediction.
    Args:
        keypoints_data: List of keypoint frames or JSON file path.
        action_type: 'fast' or 'spin'.
        pitch_angle: Pitch angle for adjustment.
        config: Configuration parameters.
    Returns:
        Tuple of (features, feature_labels, elbow_angles, wrist_fallback_frames).
    """
    config = config or {}
    visibility_threshold = config.get("visibility_threshold", 0.6)
    
    if isinstance(keypoints_data, str):
        try:
            with open(keypoints_data, 'r') as f:
                keypoints = json.load(f)
        except Exception as e:
            logging.error(f"Failed to load keypoints file {keypoints_data}: {e}")
            return None, None, None, None
    else:
        keypoints = keypoints_data

    features = []
    feature_labels = []
    elbow_angles = []
    wrist_fallback_frames = []

    for frame_idx, frame in enumerate(keypoints):
        if not frame.get("keypoints"):
            logging.warning(f"Frame {frame_idx}: No keypoints")
            continue

        kp = frame["keypoints"]
        feature = []
        shoulder = kp.get(f"landmark_{config.get('landmarks', {}).get('elbow_angle', {}).get('shoulder', 11)}", {"x": 0, "y": 0, "visibility": 0})
        elbow = kp.get(f"landmark_{config.get('landmarks', {}).get('elbow_angle', {}).get('elbow', 13)}", {"x": 0, "y": 0, "visibility": 0})
        wrist = kp.get(f"landmark_{config.get('landmarks', {}).get('elbow_angle', {}).get('wrist', 14)}", {"x": 0, "y": 0, "visibility": 0})

        if shoulder["visibility"] >= visibility_threshold and elbow["visibility"] >= visibility_threshold:
            angle = compute_elbow_angle({"keypoints": kp}, config)
            used_wrist = False
        elif shoulder["visibility"] >= visibility_threshold and wrist["visibility"] >= visibility_threshold:
            angle = compute_wrist_fallback_angle({"keypoints": kp}, config)
            used_wrist = True
        else:
            feature.extend([0, 0, 0, 0])
            elbow_angles.append(0.0)
            wrist_fallback_frames.append(frame_idx if used_wrist else None)
            features.append(feature)
            feature_labels.append(0)
            continue

        feature.extend([
            shoulder["x"], shoulder["y"],
            elbow["x"], elbow["y"]
        ])
        if used_wrist:
            feature.extend([wrist["x"], wrist["y"]])
        else:
            feature.extend([elbow["x"], elbow["y"]])

        features.append(feature)
        feature_labels.append(0)
        elbow_angles.append(angle)
        wrist_fallback_frames.append(frame_idx if used_wrist else None)

    return features, feature_labels, elbow_angles, wrist_fallback_frames
