import numpy as np
import logging

logging.basicConfig(level=logging.INFO)

def compute_elbow_angle(kps):
    """
    Compute elbow angle from keypoints.
    Args:
        kps: Keypoint dict (flat or nested under "keypoints").
    Returns:
        Angle in degrees.
    """
    if not isinstance(kps, dict):
        logging.warning("Invalid keypoint data")
        return 0.0
    
    # Handle flat or nested structure
    keypoints = kps.get("keypoints", kps)
    if not isinstance(keypoints, dict):
        logging.warning("No valid keypoints")
        return 0.0
    
    p1 = keypoints.get("landmark_11", {"x": 0, "y": 0})
    p2 = keypoints.get("landmark_13", {"x": 0, "y": 0})
    p3 = keypoints.get("landmark_14", {"x": 0, "y": 0})
    v1 = np.array([p1["x"] - p2["x"], p1["y"] - p2["y"]])
    v2 = np.array([p3["x"] - p2["x"], p3["y"] - p2["y"]])
    dot_product = np.dot(v1, v2)
    norms = np.linalg.norm(v1) * np.linalg.norm(v2)
    angle = np.degrees(np.arccos(dot_product / norms if norms > 0 else 1)) % 360
    return angle
