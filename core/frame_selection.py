import logging
import numpy as np
from core.keypoints import adjust_keypoints

logging.basicConfig(level=logging.INFO)

def detect_bfc_frame(landmarks_per_frame, frame_detector, config, pitch_ref):
    """
    Detect Back Foot Contact (BFC) frame.
    Args:
        landmarks_per_frame: List of frame keypoints.
        frame_detector: Trained FrameDetector model.
        config: Configuration parameters.
        pitch_ref: Pitch reference data.
    Returns:
        Frame index or fallback.
    """
    config = config or {}
    pitch_angle = pitch_ref.get("pitch_angle", 0) if pitch_ref else 0
    visibility_threshold = config.get("visibility_threshold", 0.6)
    fallback_frame = config.get("fallback_frames", {}).get("bfc_frame", 0)
    
    try:
        if not landmarks_per_frame:
            logging.warning("No landmarks provided for BFC detection")
            return fallback_frame
        
        frame_probs = []
        for i, frame in enumerate(landmarks_per_frame):
            if not isinstance(frame, dict):
                logging.warning(f"Frame {i}: Expected dict, got {type(frame)}")
                frame_probs.append(0.0)
                continue
            
            kp = adjust_keypoints(frame.get("keypoints", {}), pitch_angle) if frame.get("keypoints") else {}
            if not kp:
                frame_probs.append(0.0)
                continue
            
            # Check visibility for key landmarks
            left_ankle = kp.get("landmark_27", {"visibility": 0})
            right_ankle = kp.get("landmark_28", {"visibility": 0})
            if left_ankle["visibility"] < visibility_threshold and right_ankle["visibility"] < visibility_threshold:
                frame_probs.append(0.0)
                continue
            
            # Simplified BFC detection (replace with actual logic)
            frame_probs.append(1.0 if i == config.get("bfc_frame", 20) else 0.0)
        
        if not frame_probs or max(frame_probs) == 0:
            logging.warning("No valid frames for BFC detection; using fallback frame")
            return fallback_frame
        
        return np.argmax(frame_probs)
    except Exception as e:
        logging.error(f"BFC detection failed: {e}")
        return fallback_frame

def detect_ffc_frame(landmarks_per_frame, frame_detector, config, pitch_ref):
    config = config or {}
    fallback_frame = config.get("fallback_frames", {}).get("ffc_frame", 0)
    logging.warning("FFC detection not implemented; using fallback frame")
    return fallback_frame

def detect_uah_frame(landmarks_per_frame, frame_detector, config, pitch_ref):
    config = config or {}
    fallback_frame = config.get("fallback_frames", {}).get("uah_frame", 0)
    logging.warning("UAH detection not implemented; using fallback frame")
    return fallback_frame

def detect_release_frame(landmarks_per_frame, frame_detector, config, pitch_ref):
    config = config or {}
    fallback_frame = config.get("fallback_frames", {}).get("release_frame", 0)
    logging.warning("Release detection not implemented; using fallback frame")
    return fallback_frame

def select_key_frames(landmarks_per_frame, frame_detector, action_type, config=None, pitch_ref=None):
    """
    Select key frames for biomechanical analysis.
    Args:
        landmarks_per_frame: List of frame keypoints.
        frame_detector: Trained FrameDetector model.
        action_type: 'fast' or 'spin'.
        config: Configuration parameters.
        pitch_ref: Pitch reference data.
    Returns:
        Dict of frame types to indices.
    """
    config = config or {}
    pitch_ref = pitch_ref or {}
    
    try:
        return {
            "bfc_frame": detect_bfc_frame(landmarks_per_frame, frame_detector, config, pitch_ref),
            "ffc_frame": detect_ffc_frame(landmarks_per_frame, frame_detector, config, pitch_ref),
            "uah_frame": detect_uah_frame(landmarks_per_frame, frame_detector, config, pitch_ref),
            "release_frame": detect_release_frame(landmarks_per_frame, frame_detector, config, pitch_ref)
        }
    except Exception as e:
        logging.error(f"Key frame selection failed: {e}")
        return {
            "bfc_frame": config.get("fallback_frames", {}).get("bfc_frame", 0),
            "ffc_frame": config.get("fallback_frames", {}).get("ffc_frame", 0),
            "uah_frame": config.get("fallback_frames", {}).get("uah_frame", 0),
            "release_frame": config.get("fallback_frames", {}).get("release_frame", 0)
        }
