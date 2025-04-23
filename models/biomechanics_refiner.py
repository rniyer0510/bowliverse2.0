import numpy as np
import logging
from utils.alignment_data import prepare_style_data
from core.frame_selection import compute_elbow_angle

logging.basicConfig(level=logging.INFO)

class BiomechanicsRefiner:
    def __init__(self, action_type, config=None):
        self.action_type = action_type
        self.config = config or {}
        self.model = None
        self.trained = False

    def fit(self, X, y):
        """Train a model (placeholder for actual implementation)."""
        try:
            from sklearn.linear_model import LogisticRegression
            self.model = LogisticRegression(random_state=self.config.get("random_state", 42))
            self.model.fit(X, y)
            self.trained = True
            logging.info(f"BiomechanicsRefiner trained for {self.action_type}")
        except Exception as e:
            logging.error(f"BiomechanicsRefiner training failed: {e}")

    def predict(self, keypoints, config=None, pitch_ref=None):
        """
        Predict action type and biomechanical adjustments.
        Args:
            keypoints: List of keypoint dictionaries.
            config: Configuration parameters.
            pitch_ref: Pitch reference data.
        Returns:
            Predicted action type.
        """
        config = config or self.config
        pitch_ref = pitch_ref or {"pitch_angle": 0}
        
        if not self.trained or not self.model:
            logging.warning("BiomechanicsRefiner not trained")
            return self.action_type

        # Select key frames
        from core.frame_selection import select_key_frames
        from models.frame_detector import FrameDetector
        frame_detector = FrameDetector(self.action_type, config)
        key_frames = select_key_frames(keypoints, frame_detector, self.action_type, config, pitch_ref)

        # Prepare features
        features = []
        for frame_type in ["bfc_frame", "ffc_frame", "uah_frame", "release_frame"]:
            frame_idx = key_frames.get(frame_type, 0)
            if frame_idx >= len(keypoints) or frame_idx < 0 or not keypoints[frame_idx]:
                logging.debug(f"Invalid frame index {frame_idx} for {frame_type}")
                features.extend([0.0] * 6)  # Placeholder for missing data
                continue
            
            kp = keypoints[frame_idx].get("keypoints", {})
            if not kp:
                logging.debug(f"No keypoints for {frame_type} at frame {frame_idx}")
                features.extend([0.0] * 6)
                continue
            
            shoulder = kp.get(f"landmark_{config.get('landmarks', {}).get('elbow_angle', {}).get('shoulder', 11)}", {"x": 0, "y": 0})
            elbow = kp.get(f"landmark_{config.get('landmarks', {}).get('elbow_angle', {}).get('elbow', 13)}", {"x": 0, "y": 0})
            wrist = kp.get(f"landmark_{config.get('landmarks', {}).get('elbow_angle', {}).get('wrist', 14)}", {"x": 0, "y": 0})
            features.extend([
                shoulder["x"], shoulder["y"],
                elbow["x"], elbow["y"],
                wrist["x"], wrist["y"]
            ])

        # Predict action type
        try:
            X, _ = prepare_style_data(keypoints, key_frames, self.action_type, config, pitch_ref)
            if X.size == 0:
                logging.warning("No valid style data for prediction")
                return self.action_type
            pred = self.model.predict(X)
            action_type_pred = "fast" if pred[0] == 1 else "spin"
            logging.info(f"Predicted action type: {action_type_pred}")
            return action_type_pred
        except Exception as e:
            logging.error(f"BiomechanicsRefiner prediction failed: {e}")
            return self.action_type
