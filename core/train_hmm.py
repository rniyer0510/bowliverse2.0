import os
import numpy as np
import logging
from hmmlearn import hmm
from core.feature_extraction import extract_features

logging.basicConfig(level=logging.INFO)

def train_hmm(keypoints_dir, assessments, action_type, pitch_angles, config=None):
    """
    Train HMM to detect BFC, FFC, UAH, and Release frames.
    Args:
        keypoints_dir (str): Directory with keypoint JSONs.
        assessments (dict): Dict of video assessments from training_labels.json or bowliverse.db.
        action_type (str): 'fast' or 'spin'.
        pitch_angles (dict): Pitch angle data for each video.
        config (dict): Configuration parameters (e.g., n_components, n_iter).
    Returns:
        Trained HMM model or None if training fails.
    """
    config = config or {}
    n_components = config.get("hmm_components", 4)  # Default: 4 states (BFC, FFC, UAH, Release)
    n_iter = config.get("hmm_iterations", 100)
    random_state = config.get("hmm_random_state", 42)

    # Initialize HMM
    model = hmm.GaussianHMM(
        n_components=n_components,
        covariance_type="diag",
        n_iter=n_iter,
        random_state=random_state
    )
    X_hmm = []
    lengths = []

    # Collect features for all videos
    for video_id, labels in assessments.items():
        keypoint_file = os.path.join(keypoints_dir, f"{config.get('keypoints_prefix', 'bowling_analysis')}_{video_id}.json")
        if not os.path.exists(keypoint_file):
            logging.warning(f"Missing keypoints for {video_id}")
            continue

        # Extract features with pitch adjustment
        features, _, elbow_angles, _ = extract_features(keypoint_file, action_type, pitch_angles.get(video_id, 0))
        if features is None or len(features) == 0:
            logging.warning(f"No valid features for {video_id}")
            continue

        X_hmm.extend(features)
        lengths.append(len(features))
        logging.info(f"Processed {video_id}: {len(features)} frames")

    if not X_hmm:
        logging.error("No features to train HMM")
        return None

    # Train HMM
    try:
        X_array = np.array(X_hmm)
        model.fit(X_array, lengths)
        logging.info(f"HMM trained with {n_components} components, {len(X_hmm)} total frames")
        return model
    except Exception as e:
        logging.error(f"HMM training failed: {e}")
        return None
