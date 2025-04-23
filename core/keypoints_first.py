
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)

def smooth_keypoints(keypoints, window_size=3):
    """
    Smooth keypoint coordinates and visibility using a moving average.
    Returns a new list of smoothed keypoints.
    """
    if not keypoints or not isinstance(keypoints, list):
        logging.warning("Invalid keypoints for smoothing")
        return keypoints

    smoothed = []
    half_window = window_size // 2

    for i in range(len(keypoints)):
        frame_kp = keypoints[i].get("keypoints", {})
        if not frame_kp:
            smoothed.append({"keypoints": {}})
            continue

        smoothed_kp = {"keypoints": {}}
        start = max(0, i - half_window)
        end = min(len(keypoints), i + half_window + 1)

        for lm in range(33):
            lm_key = f"landmark_{lm}"
            x_vals, y_vals, vis_vals = [], [], []

            for j in range(start, end):
                neighbor_kp = keypoints[j].get("keypoints", {}).get(lm_key, {})
                x_vals.append(neighbor_kp.get("x", 0))
                y_vals.append(neighbor_kp.get("y", 0))
                vis_vals.append(neighbor_kp.get("visibility", 0))

            smoothed_kp["keypoints"][lm_key] = {
                "x": float(np.mean(x_vals)),
                "y": float(np.mean(y_vals)),
                "visibility": float(np.mean(vis_vals)),
            }

        smoothed.append(smoothed_kp)

    return smoothed
