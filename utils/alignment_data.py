import numpy as np
import logging
from core.feature_extraction import extract_features

logging.basicConfig(level=logging.INFO)

def prepare_alignment_data(keypoints_dir, assessments, action_type, config=None, pitch_refs=None, keypoints=None):
    """
    Prepare data for BiomechanicsRefiner training.
    Args:
        keypoints_dir: Directory with keypoint JSONs (optional).
        assessments: Dict of video_id to assessment data.
        action_type: 'fast' or 'spin'.
        config: Configuration parameters.
        pitch_refs: Dict of video_id to pitch reference.
        keypoints: Optional list of keypoints for single video.
    Returns:
        Tuple of (X, y) for training.
    """
    config = config or {}
    pitch_refs = pitch_refs or {}
    X = []
    y = []
    
    if keypoints:
        # Process single video
        features, _, _, _ = extract_features(keypoints, action_type, pitch_refs.get(None, {"pitch_angle": 0})["pitch_angle"], config)
        if features is not None:
            X.extend(features)
            y.extend([1 if action_type == 'fast' else 0] * len(features))
    else:
        for video_id, labels in assessments.items():
            keypoints_path = os.path.join(keypoints_dir, f"bowling_analysis_{video_id}.json")
            try:
                with open(keypoints_path, 'r') as f:
                    keypoints = json.load(f)
            except Exception as e:
                logging.error(f"Failed to load keypoints for {video_id}: {e}")
                continue
            
            pitch_ref = pitch_refs.get(video_id, {"pitch_angle": 0})
            features, _, _, _ = extract_features(keypoints, action_type, pitch_ref.get("pitch_angle", 0), config)
            if features is not None:
                X.extend(features)
                y.extend([1 if labels["action_type"] == 'fast' else 0] * len(features))
    
    if X and y:
        return np.array(X), np.array(y)
    logging.warning("No valid data for BiomechanicsRefiner training")
    return np.array([]), np.array([])
