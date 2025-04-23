import numpy as np
import logging
from hmmlearn import hmm
from core.feature_extraction import extract_features

logging.basicConfig(level=logging.INFO)

def train_hmm(X, n_components=4, n_iter=100):
    """
    Train an HMM model for frame sequence prediction.
    Args:
        X: Feature array (n_samples, n_features).
        n_components: Number of HMM states (e.g., 4 for BFC, FFC, UAH, Release).
        n_iter: Number of iterations for training.
    Returns:
        Trained HMM model.
    """
    try:
        model = hmm.GaussianHMM(
            n_components=n_components,
            covariance_type="diag",
            n_iter=n_iter,
            random_state=42
        )
        model.fit(X)
        logging.info(f"HMM trained with {n_components} components")
        return model
    except Exception as e:
        logging.error(f"Failed to train HMM: {e}")
        return None

def prepare_hmm_data(keypoints_dir, assessments, action_type, config=None, pitch_refs=None):
    """
    Prepare data for HMM training.
    Args:
        keypoints_dir: Directory with keypoint JSONs.
        assessments: Dict of video assessments.
        action_type: 'fast' or 'spin'.
        config: Configuration parameters.
        pitch_refs: Dict of video_id to pitch reference.
    Returns:
        Feature array for HMM training.
    """
    config = config or {}
    pitch_refs = pitch_refs or {}
    X = []
    
    for video_id, labels in assessments.items():
        keypoints_path = os.path.join(keypoints_dir, f"bowling_analysis_{video_id}.json")
        if not os.path.exists(keypoints_path):
            logging.warning(f"Missing keypoints for {video_id}")
            continue
        
        try:
            with open(keypoints_path, 'r') as f:
                keypoints = json.load(f)
        except Exception as e:
            logging.error(f"Failed to load keypoints for {video_id}: {e}")
            continue
        
        pitch_ref = pitch_refs.get(video_id, {"pitch_angle": 0})
        features, _, _, _ = extract_features(
            keypoints,
            action_type,
            pitch_ref.get("pitch_angle", 0),
            config
        )
        
        if features:
            X.append(features)
    
    if X:
        return np.vstack(X)
    logging.warning("No valid features for HMM training")
    return None

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python -m core.hmm_training <keypoints_dir> <action_type>")
        sys.exit(1)
    keypoints_dir = sys.argv[1]
    action_type = sys.argv[2]
    # Example usage
    assessments = {}  # Load from database or JSON
    config = {}
    pitch_refs = {}
    X = prepare_hmm_data(keypoints_dir, assessments, action_type, config, pitch_refs)
    if X is not None:
        model = train_hmm(X)
        if model:
            logging.info("HMM model trained successfully")
