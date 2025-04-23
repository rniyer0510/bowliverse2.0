import numpy as np
import logging
from sklearn.ensemble import RandomForestClassifier

logging.basicConfig(level=logging.INFO)

class FrameDetector:
    def __init__(self, action_type, config=None):
        self.action_type = action_type
        self.config = config or {}
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    def fit(self, X, y):
        """
        Train the FrameDetector model.
        Args:
            X: Feature array (n_samples, n_features).
            y: Label array (n_samples,).
        """
        try:
            self.model.fit(X, y)
            logging.info(f"FrameDetector trained for {self.action_type}")
        except Exception as e:
            logging.error(f"Failed to train FrameDetector: {e}")
    
    def predict_proba(self, keypoints):
        """
        Predict probabilities for key frames.
        Args:
            keypoints: List of keypoint dictionaries.
        Returns:
            List of probability arrays for each frame type.
        """
        try:
            from utils.frame_data import prepare_frame_data
            X = []
            for kp in keypoints:
                frame_kp = kp.get("keypoints", {})
                features = []
                for i in range(33):
                    lm = f"landmark_{i}"
                    features.extend([
                        frame_kp.get(lm, {"x": 0})["x"],
                        frame_kp.get(lm, {"y": 0})["y"],
                        frame_kp.get(lm, {"visibility": 0})["visibility"]
                    ])
                X.append(features)
            X = np.array(X)
            probs = []
            for i in range(4):  # BFC, FFC, UAH, Release
                probs.append(self.model.predict_proba(X)[:, i])
            return probs
        except Exception as e:
            logging.error(f"FrameDetector prediction failed: {e}")
            return None
