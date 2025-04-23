import numpy as np
import logging
from sklearn.linear_model import LinearRegression

logging.basicConfig(level=logging.INFO)

class AngleAdjuster:
    def __init__(self, action_type, config=None):
        """
        Initialize AngleAdjuster.
        Args:
            action_type: 'fast' or 'spin'.
            config: Configuration parameters.
        """
        self.action_type = action_type
        self.config = config or {}
        self.model = LinearRegression()
    
    def fit(self, X, y):
        """
        Train the AngleAdjuster model.
        Args:
            X: Feature array (n_samples, n_features).
            y: Target angles (n_samples,).
        """
        try:
            self.model.fit(X, y)
            logging.info(f"AngleAdjuster trained for {self.action_type}")
        except Exception as e:
            logging.error(f"Failed to train AngleAdjuster: {e}")
    
    def predict(self, keypoints, frame_idx):
        """
        Predict adjusted elbow angle for a specific frame.
        Args:
            keypoints: List of keypoint dictionaries.
            frame_idx: Index of the frame to analyze.
        Returns:
            Adjusted elbow angle.
        """
        try:
            if frame_idx >= len(keypoints) or frame_idx < 0 or not keypoints[frame_idx]:
                logging.warning(f"Invalid frame index {frame_idx}")
                return 0.0
            
            kp = keypoints[frame_idx].get("keypoints", {})
            features = []
            for i in range(33):
                lm = f"landmark_{i}"
                features.extend([
                    kp.get(lm, {"x": 0})["x"],
                    kp.get(lm, {"y": 0})["y"],
                    kp.get(lm, {"visibility": 0})["visibility"]
                ])
            features = np.array([features])
            return float(self.model.predict(features)[0])
        except Exception as e:
            logging.error(f"AngleAdjuster prediction failed: {e}")
            return 0.0
