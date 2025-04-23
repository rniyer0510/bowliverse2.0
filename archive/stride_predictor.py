from sklearn.ensemble import RandomForestRegressor

class StridePredictor:
    def __init__(self, action_type):
        self.action_type = action_type
        self.model = RandomForestRegressor(n_estimators=100)
        self.trained = False

    def fit(self, X, y):  # Added fit method
        self.model.fit(X, y)
        self.trained = True

    def predict(self, keypoints, bfc_frame, ffc_frame):
        if not self.trained:
            return 0.0
        bfc_kp = keypoints["bfc_frame"]["keypoints"]
        ffc_kp = keypoints["ffc_frame"]["keypoints"]
        features = [bfc_kp.get(f"landmark_{i}", {"x": 0, "y": 0})[k] for i in range(33) for k in ["x", "y"]] + \
                   [ffc_kp.get(f"landmark_{i}", {"x": 0, "y": 0})[k] for i in range(33) for k in ["x", "y"]]
        features.append(keypoints["scale_factor"])
        return self.model.predict([features[:133]])[0]
