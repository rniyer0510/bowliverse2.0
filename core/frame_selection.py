import numpy as np
import logging
from scipy.ndimage import uniform_filter1d
from utils.keypoints_utils import adjust_keypoints

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
    
    if p1["visibility"] < visibility_threshold or p2["visibility"] < visibility_threshold or p3["visibility"] < visibility_threshold:
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

def compute_foot_angle(kp, landmark_start, landmark_end, config=None, pitch_ref=None):
    """Compute foot angle relative to pitch crease."""
    config = config or {}
    pitch_ref = pitch_ref or {}
    start = kp.get(f"landmark_{landmark_start}", {"x": 0, "y": 0, "visibility": 0})
    end = kp.get(f"landmark_{landmark_end}", {"x": 0, "y": 0, "visibility": 0})
    if start["visibility"] < config.get("visibility_threshold", 0.6) or end["visibility"] < config.get("visibility_threshold", 0.6):
        logging.debug(f"Low visibility for foot angle (landmark_{landmark_start}-{landmark_end})")
        return float('inf')
    foot_vec = np.array([end["x"] - start["x"], end["y"] - start["y"]])
    crease_vec = pitch_ref.get("crease_direction", np.array([1, 0]))
    dot = np.dot(foot_vec, crease_vec)
    norm_vec = np.linalg.norm(foot_vec) * np.linalg.norm(crease_vec)
    if norm_vec == 0:
        return float('inf')
    cos_theta = dot / (norm_vec + 1e-6)
    angle = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
    return abs(angle)

def detect_bfc_frame(landmarks_per_frame, frame_detector, config=None, pitch_ref=None):
    """Detect BFC frame when back foot (landmark_31) lands on pitch crease."""
    config = config or {}
    pitch_ref = pitch_ref or {}
    pitch_angle = pitch_ref.get("pitch_angle", 0)
    crease_y = pitch_ref.get("crease_y", 0.8)
    y_positions = []
    valid_frames = []
    for i, frame in enumerate(landmarks_per_frame):
        kp = adjust_keypoints(frame.get("keypoints", {}), pitch_angle) if frame.get("keypoints") else {}
        if not kp:
            logging.debug(f"Frame {i}: No keypoints available")
            y_positions.append(float('inf'))
            continue
        ankle = kp.get("landmark_31", {"x": 0, "y": 0, "visibility": 0})
        if ankle["visibility"] >= config.get("visibility_threshold", 0.6):
            y_positions.append(ankle["y"])
            valid_frames.append(i)
        else:
            y_positions.append(float('inf'))
    if not valid_frames:
        logging.warning("No valid frames for BFC detection; using fallback frame")
        return config.get("bfc_fallback_frame", 20)
    y_positions = np.array(y_positions)
    velocity = np.gradient(y_positions, edge_order=1)
    for i in valid_frames[1:-1]:
        kp = adjust_keypoints(landmarks_per_frame[i].get("keypoints", {}), pitch_angle) if landmarks_per_frame[i].get("keypoints") else {}
        if not kp:
            continue
        ankle = kp.get("landmark_31", {"x": 0, "y": 0, "visibility": 0})
        if ankle["visibility"] < config.get("visibility_threshold", 0.6):
            continue
        foot_angle = compute_foot_angle(kp, 31, 30, config, pitch_ref)
        if (abs(ankle["y"] - crease_y) < config.get("crease_contact_threshold", 0.05) and
            foot_angle <= config.get("foot_angle_max", 20) and
            velocity[i] < config.get("landing_velocity_threshold", -0.01)):
            logging.info(f"BFC detected at frame {i}: landmark_31 y={ankle['y']:.2f}, angle={foot_angle:.2f}")
            return i
    logging.warning("No BFC detected; using fallback frame")
    return config.get("bfc_fallback_frame", 20)

def detect_ffc_frame(landmarks_per_frame, frame_detector, config=None, pitch_ref=None):
    """Detect FFC frame when front foot (landmark_27) lands on pitch crease."""
    config = config or {}
    pitch_ref = pitch_ref or {}
    pitch_angle = pitch_ref.get("pitch_angle", 0)
    crease_y = pitch_ref.get("crease_y", 0.8)
    y_positions = []
    valid_frames = []
    for i, frame in enumerate(landmarks_per_frame):
        kp = adjust_keypoints(frame.get("keypoints", {}), pitch_angle) if frame.get("keypoints") else {}
        if not kp:
            logging.debug(f"Frame {i}: No keypoints available")
            y_positions.append(float('inf'))
            continue
        ankle = kp.get("landmark_27", {"x": 0, "y": 0, "visibility": 0})
        if ankle["visibility"] >= config.get("visibility_threshold", 0.6):
            y_positions.append(ankle["y"])
            valid_frames.append(i)
        else:
            y_positions.append(float('inf'))
    if not valid_frames:
        logging.warning("No valid frames for FFC detection; using fallback frame")
        return config.get("ffc_fallback_frame", 50)
    y_positions = np.array(y_positions)
    velocity = np.gradient(y_positions, edge_order=1)
    for i in valid_frames[1:-1]:
        kp = adjust_keypoints(landmarks_per_frame[i].get("keypoints", {}), pitch_angle) if landmarks_per_frame[i].get("keypoints") else {}
        if not kp:
            continue
        ankle = kp.get("landmark_27", {"x": 0, "y": 0, "visibility": 0})
        if ankle["visibility"] < config.get("visibility_threshold", 0.6):
            continue
        foot_angle = compute_foot_angle(kp, 27, 28, config, pitch_ref)
        if (abs(ankle["y"] - crease_y) < config.get("crease_contact_threshold", 0.05) and
            foot_angle <= config.get("foot_angle_max", 20) and
            velocity[i] < config.get("landing_velocity_threshold", -0.01)):
            logging.info(f"FFC detected at frame {i}: landmark_27 y={ankle['y']:.2f}, angle={foot_angle:.2f}")
            return i
    logging.warning("No FFC detected; using fallback frame")
    return config.get("ffc_fallback_frame", 50)

def detect_release_frame(landmarks_per_frame, frame_detector, config=None, pitch_ref=None):
    """Detect Release frame using wrist (landmark_14) and elbow angle."""
    config = config or {}
    pitch_ref = pitch_ref or {}
    pitch_angle = pitch_ref.get("pitch_angle", 0)
    valid_frames = []
    for i, frame in enumerate(landmarks_per_frame):
        kp = adjust_keypoints(frame.get("keypoints", {}), pitch_angle) if frame.get("keypoints") else {}
        if not kp:
            logging.debug(f"Frame {i}: No keypoints available")
            continue
        wrist = kp.get("landmark_14", {"x": 0, "y": 0, "visibility": 0})
        if wrist["visibility"] >= config.get("visibility_threshold", 0.6):
            valid_frames.append(i)
    if not valid_frames:
        logging.warning("No valid frames for Release detection; using fallback frame")
        return config.get("release_frame_min", 224)
    for i in valid_frames:
        kp = adjust_keypoints(landmarks_per_frame[i].get("keypoints", {}), pitch_angle) if landmarks_per_frame[i].get("keypoints") else {}
        if not kp:
            continue
        wrist = kp.get("landmark_14", {"x": 0, "y": 0, "visibility": 0})
        elbow_angle = compute_elbow_angle({"keypoints": kp}, config)
        if (wrist["visibility"] >= config.get("visibility_threshold", 0.6) and
            config.get("release_angle_min", 90) <= elbow_angle <= config.get("release_angle_max", 120) and
            i >= config.get("release_frame_min", 200)):
            logging.info(f"Release detected at frame {i}: elbow_angle={elbow_angle:.2f}")
            return i
    logging.warning("No Release detected; using fallback frame")
    return config.get("release_frame_min", 224)

def detect_uah_frame_v2(landmarks_per_frame, release_frame_index, ffc_frame_index=None, arm_side='right', config=None, pitch_ref=None):
    """Detect UAH frame using shoulder–elbow angle to pitch horizontal."""
    config = config or {}
    pitch_ref = pitch_ref or {}
    pitch_angle = pitch_ref.get("pitch_angle", 0)
    transformed_frames = [
        {"keypoints": adjust_keypoints(frame.get("keypoints", {}), pitch_angle) if frame.get("keypoints") else {}}
        for frame in landmarks_per_frame
    ]
    shoulder_idx = 11 if arm_side == 'right' else 12
    elbow_idx = 13 if arm_side == 'right' else 14
    wrist_idx = 14
    start_idx = max(config.get("min_frame", 90), release_frame_index - config.get("uah_frame_window_max", 30))
    end_idx = max(config.get("min_frame", 90), release_frame_index - config.get("uah_frame_window_min", 15))
    angles, elbow_heights, frame_indices = [], [], []
    for i in range(end_idx, start_idx - 1, -1):
        try:
            lm = transformed_frames[i]["keypoints"]
            if not lm:
                logging.debug(f"Frame {i}: No keypoints available")
                continue
            shoulder = lm.get(f"landmark_{shoulder_idx}", {"x": 0, "y": 0, "visibility": 0})
            elbow = lm.get(f"landmark_{elbow_idx}", {"x": 0, "y": 0, "visibility": 0})
            wrist = lm.get(f"landmark_{wrist_idx}", {"x": 0, "y": 0, "visibility": 0})
            use_wrist = shoulder["visibility"] < config.get("uah_visibility_threshold", 0.3) or \
                        elbow["visibility"] < config.get("uah_visibility_threshold", 0.3)
            point = wrist if use_wrist else elbow
            if point["visibility"] < config.get("uah_visibility_threshold", 0.3):
                continue
            if not (point["x"] < shoulder["x"] and point["y"] < shoulder["y"]):
                continue
            upper_arm_vec = np.array([point["x"] - shoulder["x"], point["y"] - shoulder["y"]])
            horizontal = np.array([1, 0])
            dot = np.dot(upper_arm_vec, horizontal)
            norm_vec = np.linalg.norm(upper_arm_vec)
            if norm_vec == 0:
                continue
            cos_theta = dot / (norm_vec + 1e-6)
            angle = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
            angle = abs(angle)
            height_diff = point["y"] - shoulder["y"]
            angles.append(angle)
            elbow_heights.append(height_diff)
            frame_indices.append(i)
        except Exception as e:
            logging.debug(f"Error processing frame {i}: {e}")
            continue
    if not angles:
        logging.warning("No valid UAH frames")
        return release_frame_index + config.get("uah_default_frame_offset", -20)
    angles_smooth = uniform_filter1d(angles, size=config.get("uah_smoothing_window", 3))
    velocity = np.gradient(angles_smooth, edge_order=1)
    min_angle = config.get("uah_max_angle", 999)
    best_index = release_frame_index + config.get("uah_default_frame_offset", -20)
    for i in range(1, len(angles_smooth) - 1):
        if (angles_smooth[i] < config.get("uah_angle_threshold", 45) and
            elbow_heights[i] > config.get("uah_elbow_height", -0.05) and
            velocity[i + 1] - velocity[i] > config.get("uah_velocity_spike", 5)):
            if angles_smooth[i] < min_angle:
                min_angle = angles_smooth[i]
                best_index = frame_indices[i]
    logging.info(f"UAH detected at frame {best_index}: angle={min_angle:.2f}")
    return best_index

def select_key_frames(landmarks_per_frame, frame_detector, action_type="fast", config=None, pitch_ref=None):
    """Select key frames using detector and algorithms."""
    config = config or {}
    pitch_ref = pitch_ref or {}
    frames = {
        "bfc_frame": detect_bfc_frame(landmarks_per_frame, frame_detector, config, pitch_ref),
        "ffc_frame": detect_ffc_frame(landmarks_per_frame, frame_detector, config, pitch_ref),
        "release_frame": detect_release_frame(landmarks_per_frame, frame_detector, config, pitch_ref)
    }
    frames["uah_frame"] = detect_uah_frame_v2(landmarks_per_frame, frames["release_frame"], frames["ffc_frame"], config=config, pitch_ref=pitch_ref)
    return frames
