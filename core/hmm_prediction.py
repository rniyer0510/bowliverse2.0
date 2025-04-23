
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)

def predict_hmm_sequence(model, X_frame):
    """
    Predict biomechanical frame sequence using a trained HMM.
    Args:
        model: Trained HMM model.
        X_frame: Feature matrix (n_frames x n_features).
    Returns:
        predictions: List of state labels (0-3 corresponding to BFC, FFC, UAH, Release)
    """
    try:
        predictions = model.predict(np.array(X_frame))
        logging.info(f"HMM predicted {len(predictions)} frame labels.")
        return predictions
    except Exception as e:
        logging.error(f"HMM prediction failed: {e}")
        return []
