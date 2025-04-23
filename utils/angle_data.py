import numpy as np
import logging
from core.frame_selection import select_key_frames
from utils.angle_utils import compute_elbow_angle

logging.basicConfig(level=logging.INFO)

def prepare_angle_data(keypoints, labels, action_type, config=None, pitch_ref=None):
    """
    Prepare angle data for training AngleAdjuster.
    Args:
        keypoints: List of keypoint frames.
        labels: Dict with frame labels (bfc_frame, ffc_frame, uah_frame, release_frame).
        action_type: 'fast' or 'spin'.
        config: Configuration parameters.
        pitch_ref: Pitch reference data (pitch_angle, crease_y, crease_direction).
    Returns:
        X_angle: Feature array.
        y_angle: Angle labels.
    """
    config = config or {}
    pitch_ref = pitch_ref or {"pitch_angle": 0}
    X_angle = []
    y_angle = []
    
    # Select key frames
    selected_frames = select_key_frames(keypoints, None, action_type, config, pitch_ref)
    
    for frame_type, frame_idx in selected_frames.items():
        if frame_idx >= len(keypoints) or frame_idx < 0:
            logging.warning(f"Invalid {frame_type} index: {frame_idx}")
            continue
        kp = keypoints[frame_idx]["keypoints"]
        angle = compute_elbow_angle({"keypoints": kp}, config)
        features = [
            kp.get(f"landmark_{config.get('landmarks', {}).get('elbow_angle', {}).get('shoulder', 11)}", {"x": 0, "y": 0})["x"],
            kp.get(f"landmark_{config.get('landmarks', {}).get('elbow_angle', {}).get('shoulder', 11)}", {"x": 0, "y": 0})["y"],
            kp.get(f"landmark_{config.get('landmarks', {}).get('elbow_angle', {}).get('elbow', 13)}", {"x": 0, "y": 0})["x"],
            kp.get(f"landmark_{config.get('landmarks', {}).get('elbow_angle', {}).get('elbow', 13)}", {"x": 0, "y": 0})["y"],
            kp.get(f"landmark_{config.get('landmarks', {}).get('elbow_angle', {}).get('wrist', 14)}", {"x": 0, "y": 0})["x"],
            kp.get(f"landmark_{config.get('landmarks', {}).get('elbow_angle', {}).get('wrist', 14)}", {"x": 0, "y": 0})["y"]
        ]
        X_angle.append(features)
        y_angle.append(angle)
        logging.info(f"Prepared {frame_type} (frame {frame_idx}): angle={angle:.2f}")
    
    return np.array(X_angle) if X_angle else np.array([]), np.array(y_angle) if y_angle else np.array([])
