import numpy as np
import logging
from sklearn.ensemble import RandomForestClassifier
from utils.alignment_data import prepare_alignment_data

logging.basicConfig(level=logging.INFO)

class BiomechanicsRefiner:
    def __init__(self, action_type, config=None):
        self.action_type = action_type
        self.config = config or {}
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    def fit(self, X, y):
        try:
            self.model.fit(X, y)
            logging.info(f"BiomechanicsRefiner trained for {self.action_type}")
        except Exception as e:
            logging.error(f"Failed to train BiomechanicsRefiner: {e}")
    
    def predict(self, keypoints, config=None, pitch_ref=None):
        """
        Predict action type from keypoints.
        Args:
            keypoints: List of keypoint dictionaries.
            config: Configuration parameters.
            pitch_ref: Pitch reference data.
        Returns:
            Predicted action type ('fast' or 'spin').
        """
        try:
            config = config or self.config
            pitch_ref = pitch_ref or {"pitch_angle": 0}
            X, _ = prepare_alignment_data(keypoints_dir=None, assessments=None, action_type=self.action_type, config=config, pitch_refs={None: pitch_ref}, keypoints=keypoints)
            if X.size == 0:
                logging.warning("No features for BiomechanicsRefiner prediction")
                return self.action_type
            return 'fast' if self.model.predict(X)[0] == 1 else 'spin'
        except Exception as e:
            logging.error(f"BiomechanicsRefiner prediction failed: {e}")
            return self.action_type
