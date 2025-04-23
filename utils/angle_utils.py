import numpy as np
import logging

logging.basicConfig(level=logging.INFO)

def compute_elbow_angle(kps, config=None):
    """
    Compute elbow angle from keypoints (landmark_11–13–14).
    Args:
        kps: Keypoint dict (flat or nested under "keypoints").
        config: Configuration parameters (landmark indices, visibility threshold).
    Returns:
        Angle in degrees.
    """
    config = config or {}
    visibility_threshold = config.get("visibility_threshold", 0.6)
    landmarks = config.get("landmarks", {}).get("elbow_angle", {"shoulder": 11, "elbow": 13, "wrist": 14})
    
    keypoints = kps.get("keypoints", kps)
    if not isinstance(keypoints, dict):
        logging.debug("Invalid keypoint data")
        return 0.0
    
    p1 = keypoints.get(f"landmark_{landmarks['shoulder']}", {"x": 0, "y": 0, "visibility": 0})
    p2 = keypoints.get(f"landmark_{landmarks['elbow']}", {"x": 0, "y": 0, "visibility": 0})
    p3 = keypoints.get(f"landmark_{landmarks['wrist']}", {"x": 0, "y": 0, "visibility": 0})
    
    if (p1["visibility"] < visibility_threshold or 
        p2["visibility"] < visibility_threshold or 
        p3["visibility"] < visibility_threshold):
        logging.debug("Low visibility for elbow angle")
        return 0.0
    
    v1 = np.array([p1["x"] - p2["x"], p1["y"] - p2["y"]])
    v2 = np.array([p3["x"] - p2["x"], p3["y"] - p2["y"]])
    dot_product = np.dot(v1, v2)
    norms = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norms == 0:
        logging.debug("Zero-length vector in elbow angle")
        return 0.0
    
    angle = np.degrees(np.arccos(np.clip(dot_product / norms, -1.0, 1.0))) % 360
    return angle

def compute_wrist_fallback_angle(kps, config=None):
    """
    Compute elbow angle using wrist (landmark_11–14) when elbow visibility is low.
    Args:
        kps: Keypoint dict.
        config: Configuration parameters.
    Returns:
        Angle in degrees.
    """
    config = config or {}
    visibility_threshold = config.get("wrist_visibility_threshold", 0.6)
    landmarks = config.get("landmarks", {}).get("elbow_angle", {"shoulder": 11, "wrist": 14})
    
    keypoints = kps.get("keypoints", kps)
    if not isinstance(keypoints, dict):
        logging.debug("Invalid keypoint data")
        return 0.0
    
    p1 = keypoints.get(f"landmark_{landmarks['shoulder']}", {"x": 0, "y": 0, "visibility": 0})
    p3 = keypoints.get(f"landmark_{landmarks['wrist']}", {"x": 0, "y": 0, "visibility": 0})
    
    if p1["visibility"] < visibility_threshold or p3["visibility"] < visibility_threshold:
        logging.debug("Low visibility for wrist fallback angle")
        return 0.0
    
    # Approximate elbow position (midpoint or assume fixed offset)
    p2_approx = {
        "x": (p1["x"] + p3["x"]) / 2,
        "y": (p1["y"] + p3["y"]) / 2
    }
    
    v1 = np.array([p1["x"] - p2_approx["x"], p1["y"] - p2_approx["y"]])
    v2 = np.array([p3["x"] - p2_approx["x"], p3["y"] - p2_approx["y"]])
    dot_product = np.dot(v1, v2)
    norms = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norms == 0:
        logging.debug("Zero-length vector in wrist fallback angle")
        return 0.0
    
    angle = np.degrees(np.arccos(np.clip(dot_product / norms, -1.0, 1.0))) % 360
    if angle < config.get("elbow_angle_min", 90) or angle > config.get("elbow_angle_max", 180):
        logging.debug(f"Wrist fallback angle {angle:.2f} out of range")
        return 0.0
    
    return angle
