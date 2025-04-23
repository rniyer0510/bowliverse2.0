from sklearn.ensemble import RandomForestClassifier
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)

class FrameDetector:
    def __init__(self, action_type, config=None):
        self.action_type = action_type
        self.config = config or {}
        self.model = RandomForestClassifier(
            n_estimators=self.config.get("rf_n_estimators", 100),
            random_state=self.config.get("random_state", 42)
        )
        self.trained = False
    
    def train(self, X, y):
        """Train the frame detector model."""
        try:
            self.model.fit(X, y)
            self.trained = True
            logging.info(f"FrameDetector trained for {self.action_type}")
        except Exception as e:
            logging.error(f"FrameDetector training failed: {e}")
    
    def predict(self, X):
        """Predict frame labels."""
        if not self.trained:
            logging.warning("FrameDetector not trained")
            return np.zeros(len(X), dtype=int)
        try:
            return self.model.predict(X)
        except Exception as e:
            logging.error(f"FrameDetector prediction failed: {e}")
            return np.zeros(len(X), dtype=int)
