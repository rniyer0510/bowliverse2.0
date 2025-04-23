import os
import numpy as np
import logging
from hmmlearn import hmm
from core.feature_extraction import extract_features

logging.basicConfig(level=logging.INFO)

def train_hmm(keypoints_dir, assessments, action_type, pitch_refs, config=None):
    """
    Train HMM to detect BFC, FFC, UAH, and Release frames.
    Args:
        keypoints_dir: Directory with keypoint JSONs.
        assessments: Dict of video assessments from bowliverse.db.
        action_type: 'fast' or 'spin'.
        pitch_refs: Dict of video_id to pitch reference (pitch_angle, crease_y, crease_direction).
        config: Configuration parameters (e.g., n_components, n_iter).
    Returns:
        Trained HMM model or None if training fails.
    """
    config = config or {}
    n_components = config.get("hmm_components", 4)
    n_iter = config.get("hmm_iterations", 100)
    random_state = config.get("hmm_random_state", 42)

    model = hmm.GaussianHMM(
        n_components=n_components,
        covariance_type="diag",
        n_iter=n_iter,
        random_state=random_state
    )
    X_hmm = []
    lengths = []

    for video_id, labels in assessments.items():
        keypoint_file = os.path.join(keypoints_dir, f"{config.get('keypoints_prefix', 'bowling_analysis')}_{video_id}.json")
        if not os.path.exists(keypoint_file):
            logging.warning(f"Missing keypoints for {video_id}")
            continue

        pitch_ref = pitch_refs.get(video_id, {"pitch_angle": 0})
        features, _, elbow_angles, _ = extract_features(keypoint_file, action_type, pitch_ref.get("pitch_angle", 0))
        if features is None or len(features) == 0:
            logging.warning(f"No valid features for {video_id}")
            continue

        X_hmm.extend(features)
        lengths.append(len(features))
        logging.info(f"Processed {video_id}: {len(features)} frames")

    if not X_hmm:
        logging.error("No features to train HMM")
        return None

    try:
        X_array = np.array(X_hmm)
        model.fit(X_array, lengths)
        logging.info(f"HMM trained with {n_components} components, {len(X_hmm)} total frames")
        return model
    except Exception as e:
        logging.error(f"HMM training failed: {e}")
        return None
