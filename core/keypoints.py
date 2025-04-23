import numpy as np
from scipy.ndimage import uniform_filter1d
import logging

logging.basicConfig(level=logging.INFO)

def smooth_keypoints(keypoints):
    """
    Smooth keypoint coordinates using uniform filter.
    """
    smoothed = []
    for frame in keypoints:
        kp = frame.get("keypoints", {})
        smoothed_frame = {}
        for lm in kp:
            smoothed_frame[lm] = {
                "x": uniform_filter1d([kp[lm]["x"]], size=5)[0],
                "y": uniform_filter1d([kp[lm]["y"]], size=5)[0],
                "visibility": kp[lm]["visibility"]
            }
        smoothed.append({"keypoints": smoothed_frame})
    return smoothed

def adjust_keypoints(keypoints, pitch_angle):
    """
    Adjust keypoints for pitch angle rotation.
    Args:
        keypoints: Dict of landmarks.
        pitch_angle: Angle in degrees.
    Returns:
        Adjusted keypoints dict.
    """
    if not keypoints:
        logging.warning("No keypoints to adjust")
        return keypoints
    
    adjusted = {}
    try:
        angle_rad = np.radians(pitch_angle)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        for lm, data in keypoints.items():
            x, y = data["x"], data["y"]
            x_new = x * cos_a + y * sin_a
            y_new = -x * sin_a + y * cos_a
            adjusted[lm] = {
                "x": x_new,
                "y": y_new,
                "visibility": data.get("visibility", 0)
            }
        return adjusted
    except Exception as e:
        logging.error(f"Failed to adjust keypoints: {e}")
        return keypoints
