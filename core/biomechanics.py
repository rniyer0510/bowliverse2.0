import numpy as np
import logging
from core.frame_selection import compute_elbow_angle

logging.basicConfig(level=logging.INFO)

def analyze_biomechanics(keypoints, key_frames, config=None, pitch_ref=None):
    """
    Analyze biomechanics for bowling action.
    Args:
        keypoints: List of keypoint dictionaries.
        key_frames: Dict with frame indices (bfc_frame, ffc_frame, uah_frame, release_frame).
        config: Configuration parameters.
        pitch_ref: Pitch reference data.
    Returns:
        Dict with metrics and alignment analysis.
    """
    config = config or {}
    pitch_ref = pitch_ref or {"pitch_angle": 0}
    results = {"metrics": {}, "alignment": {}}
    
    # Initialize metrics
    for frame_type in ["bfc_frame", "ffc_frame", "uah_frame", "release_frame"]:
        results["metrics"][frame_type] = key_frames.get(frame_type, 0)
        results["metrics"][f"{frame_type}_elbow_angle"] = 0.0
        results["metrics"][f"{frame_type}_shoulder_angle"] = 0.0
    
    # Analyze each key frame
    for frame_type in ["bfc_frame", "ffc_frame", "uah_frame", "release_frame"]:
        frame_idx = key_frames.get(frame_type, 0)
        if frame_idx >= len(keypoints) or frame_idx < 0 or not keypoints[frame_idx]:
            logging.warning(f"Invalid frame index {frame_idx} for {frame_type}")
            continue
        
        kp = keypoints[frame_idx].get("keypoints", {})
        if not kp:
            logging.warning(f"No keypoints available for {frame_type} at frame {frame_idx}")
            continue
        
        # Elbow angle
        elbow_angle = compute_elbow_angle({"keypoints": kp}, config)
        results["metrics"][f"{frame_type}_elbow_angle"] = elbow_angle
        
        # Shoulder angle (relative to horizontal)
        shoulder_idx = config.get("landmarks", {}).get("elbow_angle", {}).get("shoulder", 11)
        elbow_idx = config.get("landmarks", {}).get("elbow_angle", {}).get("elbow", 13)
        shoulder = kp.get(f"landmark_{shoulder_idx}", {"x": 0, "y": 0, "visibility": 0})
        elbow = kp.get(f"landmark_{elbow_idx}", {"x": 0, "y": 0, "visibility": 0})
        shoulder_vis = shoulder.get("visibility", 0)
        elbow_vis = elbow.get("visibility", 0)
        
        if shoulder_vis >= config.get("visibility_threshold", 0.6) and elbow_vis >= config.get("visibility_threshold", 0.6):
            upper_arm_vec = np.array([elbow["x"] - shoulder["x"], elbow["y"] - shoulder["y"]])
            horizontal = np.array([1, 0])
            dot = np.dot(upper_arm_vec, horizontal)
            norm_vec = np.linalg.norm(upper_arm_vec)
            if norm_vec > 0:
                cos_theta = dot / (norm_vec + 1e-6)
                shoulder_angle = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
                results["metrics"][f"{frame_type}_shoulder_angle"] = abs(shoulder_angle)
            else:
                logging.warning(f"Zero-length vector for shoulder angle at {frame_type}")
        else:
            logging.debug(f"Low visibility for shoulder angle at {frame_type}")
    
    # Alignment analysis
    uah_idx = key_frames.get("uah_frame", 0)
    release_idx = key_frames.get("release_frame", 0)
    if uah_idx < len(keypoints) and release_idx < len(keypoints) and keypoints[uah_idx] and keypoints[release_idx]:
        uah_kp = keypoints[uah_idx].get("keypoints", {})
        release_kp = keypoints[release_idx].get("keypoints", {})
        shoulder = uah_kp.get(f"landmark_{config.get('landmarks', {}).get('elbow_angle', {}).get('shoulder', 11)}", {"x": 0, "y": 0})
        elbow = uah_kp.get(f"landmark_{config.get('landmarks', {}).get('elbow_angle', {}).get('elbow', 13)}", {"x": 0, "y": 0})
        wrist = release_kp.get(f"landmark_{config.get('landmarks', {}).get('elbow_angle', {}).get('wrist', 14)}", {"x": 0, "y": 0})
        
        if (shoulder.get("visibility", 0) >= config.get("visibility_threshold", 0.6) and
            elbow.get("visibility", 0) >= config.get("visibility_threshold", 0.6) and
            wrist.get("visibility", 0) >= config.get("visibility_threshold", 0.6)):
            arm_vec = np.array([elbow["x"] - shoulder["x"], elbow["y"] - shoulder["y"]])
            wrist_vec = np.array([wrist["x"] - elbow["x"], wrist["y"] - elbow["y"]])
            dot = np.dot(arm_vec, wrist_vec)
            norm_vec = np.linalg.norm(arm_vec) * np.linalg.norm(wrist_vec)
            if norm_vec > 0:
                cos_theta = dot / (norm_vec + 1e-6)
                alignment_angle = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
                results["alignment"]["arm_wrist_angle"] = abs(alignment_angle)
                results["alignment"]["is_aligned"] = alignment_angle < config.get("alignment_threshold", 30)
            else:
                logging.warning("Zero-length vector in alignment analysis")
        else:
            logging.debug("Low visibility for alignment analysis")
    
    logging.info(f"Biomechanics analysis completed for {len(key_frames)} key frames")
    return results
