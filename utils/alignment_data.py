import numpy as np
import logging
from utils.angle_utils import compute_elbow_angle, compute_wrist_fallback_angle

logging.basicConfig(level=logging.INFO)

def compute_shoulder_angle(kp, config=None, pitch_ref=None):
    """
    Compute shoulder alignment angle (landmark_11 to landmark_12 relative to horizontal).
    Args:
        kp: Keypoint dict (flat or nested under "keypoints").
        config: Configuration parameters.
        pitch_ref: Pitch reference data.
    Returns:
        Angle in degrees.
    """
    config = config or {}
    visibility_threshold = config.get("visibility_threshold", 0.6)
    
    keypoints = kp.get("keypoints", kp)
    left_shoulder = keypoints.get("landmark_11", {"x": 0, "y": 0, "visibility": 0})
    right_shoulder = keypoints.get("landmark_12", {"x": 0, "y": 0, "visibility": 0})
    
    if left_shoulder["visibility"] < visibility_threshold or right_shoulder["visibility"] < visibility_threshold:
        logging.warning("Low visibility for shoulder angle calculation")
        return 0.0
    
    shoulder_vec = np.array([right_shoulder["x"] - left_shoulder["x"], right_shoulder["y"] - left_shoulder["y"]])
    horizontal = np.array([1, 0])
    
    dot = np.dot(shoulder_vec, horizontal)
    norm_vec = np.linalg.norm(shoulder_vec)
    if norm_vec == 0:
        logging.warning("Invalid shoulder vector for angle calculation")
        return 0.0
    
    cos_theta = dot / (norm_vec + 1e-6)
    angle = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
    return abs(angle)

def compute_hip_angle(kp, config=None, pitch_ref=None):
    """
    Compute hip alignment angle (landmark_23 to landmark_24 relative to horizontal).
    Args:
        kp: Keypoint dict (flat or nested under "keypoints").
        config: Configuration parameters.
        pitch_ref: Pitch reference data.
    Returns:
        Angle in degrees.
    """
    config = config or {}
    visibility_threshold = config.get("visibility_threshold", 0.6)
    
    keypoints = kp.get("keypoints", kp)
    left_hip = keypoints.get("landmark_23", {"x": 0, "y": 0, "visibility": 0})
    right_hip = keypoints.get("landmark_24", {"x": 0, "y": 0, "visibility": 0})
    
    if left_hip["visibility"] < visibility_threshold or right_hip["visibility"] < visibility_threshold:
        logging.warning("Low visibility for hip angle calculation")
        return 0.0
    
    hip_vec = np.array([right_hip["x"] - left_hip["x"], right_hip["y"] - left_hip["y"]])
    horizontal = np.array([1, 0])
    
    dot = np.dot(hip_vec, horizontal)
    norm_vec = np.linalg.norm(hip_vec)
    if norm_vec == 0:
        logging.warning("Invalid hip vector for angle calculation")
        return 0.0
    
    cos_theta = dot / (norm_vec + 1e-6)
    angle = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
    return abs(angle)

def prepare_style_data(keypoints, labels, action_type, config=None, pitch_ref=None):
    """
    Prepare alignment data (shoulder/hip angles) for training BiomechanicsRefiner.
    Args:
        keypoints: List of keypoint frames.
        labels: Dict with frame labels (bfc_frame, ffc_frame, uah_frame, release_frame).
        action_type: 'fast' or 'spin'.
        config: Configuration parameters.
        pitch_ref: Pitch reference data.
    Returns:
        X: Feature array (shoulder_angle, hip_angle, elbow_angle).
        y: Labels (action_type: front-on, side-on, mixed).
    """
    config = config or {}
    pitch_ref = pitch_ref or {}
    visibility_threshold = config.get("visibility_threshold", 0.6)
    X = []
    y = []
    
    for frame_idx, kp in enumerate(keypoints):
        try:
            keypoints = kp.get("keypoints", {})
            if not keypoints:
                X.append([0.0, 0.0, 0.0])
                y.append("Unknown")
                continue
            
            # Compute alignment angles
            shoulder_angle = compute_shoulder_angle(kp, config, pitch_ref)
            hip_angle = compute_hip_angle(kp, config, pitch_ref)
            
            # Compute elbow angle with wrist fallback
            shoulder_vis = keypoints.get("landmark_11", {}).get("visibility", 0)
            elbow_vis = keypoints.get("landmark_13", {}).get("visibility", 0)
            wrist_vis = keypoints.get("landmark_14", {}).get("visibility", 0)
            
            if shoulder_vis >= visibility_threshold and elbow_vis >= visibility_threshold:
                elbow_angle = compute_elbow_angle(kp, config)
            elif shoulder_vis >= visibility_threshold and wrist_vis >= visibility_threshold:
                elbow_angle = compute_wrist_fallback_angle(kp, config)
                logging.info(f"Frame {frame_idx}: Used wrist fallback for elbow angle")
            else:
                elbow_angle = 0.0
                logging.warning(f"Frame {frame_idx}: Insufficient visibility for elbow angle")
            
            # Feature vector
            features = [shoulder_angle, hip_angle, elbow_angle]
            X.append(features)
            
            # Assign label based on BFC frame
            if frame_idx == labels.get("bfc_frame", -1):
                action_type = "front-on" if shoulder_angle <= config.get("front_on_max", 30) and hip_angle <= config.get("front_on_max", 30) else \
                              "side-on" if shoulder_angle >= config.get("side_on_min", 60) and hip_angle >= config.get("side_on_min", 60) else "mixed"
                y.append(action_type)
            else:
                y.append("Unknown")
                
        except Exception as e:
            logging.warning(f"Frame {frame_idx}: Failed to compute alignment features: {e}")
            X.append([0.0, 0.0, 0.0])
            y.append("Unknown")
    
    X_array = np.array(X)
    y_array = np.array(y)
    
    if X_array.size == 0:
        logging.error("No valid alignment features extracted")
        return np.array([]), np.array([])
    
    return X_array, y_array
