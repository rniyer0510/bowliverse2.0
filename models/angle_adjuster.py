from sklearn.ensemble import RandomForestRegressor

class AngleAdjuster:
    def __init__(self, angle_type="elbow"):
        self.angle_type = angle_type
        self.model = RandomForestRegressor(n_estimators=100)
        self.trained = False

    def fit(self, X, y):
        self.model.fit(X, y)  # X: (N, 6) for elbow keypoints x,y
        self.trained = True

    def predict(self, keypoints, frame_idx):
        if not self.trained:
            return 0.0
        kp = keypoints[frame_idx]["keypoints"]
        features = [
            kp.get("landmark_11", {"x": 0, "y": 0})["x"],
            kp.get("landmark_11", {"x": 0, "y": 0})["y"],
            kp.get("landmark_13", {"x": 0, "y": 0})["x"],
            kp.get("landmark_13", {"x": 0, "y": 0})["y"],
            kp.get("landmark_14", {"x": 0, "y": 0})["x"],
            kp.get("landmark_14", {"x": 0, "y": 0})["y"]
        ]
        pred = self.model.predict([features])[0]
        return pred  # Predicted elbow angle
