import numpy as np
import logging
from scipy.ndimage import uniform_filter1d

logging.basicConfig(level=logging.INFO)

def smooth_keypoints(keypoints, window_size=3):
    """
    Smooth keypoints across frames using a moving average filter.
    Args:
        keypoints: List of keypoint dictionaries, each containing 'keypoints' with landmark data.
        window_size: Size of the smoothing window.
    Returns:
        Smoothed keypoints list.
    """
    if not keypoints:
        logging.warning("No keypoints provided for smoothing")
        return keypoints

    smoothed_keypoints = []
    num_frames = len(keypoints)
    
    # Collect all landmarks
    landmark_keys = keypoints[0].get("keypoints", {}).keys() if keypoints else []
    for landmark in landmark_keys:
        x = np.array([kp.get("keypoints", {}).get(landmark, {"x": 0})["x"] for kp in keypoints])
        y = np.array([kp.get("keypoints", {}).get(landmark, {"y": 0})["y"] for kp in keypoints])
        visibility = np.array([kp.get("keypoints", {}).get(landmark, {"visibility": 0})["visibility"] for kp in keypoints])
        
        # Smooth coordinates and visibility
        x_smooth = uniform_filter1d(x, size=window_size)
        y_smooth = uniform_filter1d(y, size=window_size)
        vis_smooth = uniform_filter1d(visibility, size=window_size)
        
        # Update keypoints for each frame
        for i in range(num_frames):
            if i >= len(smoothed_keypoints):
                smoothed_keypoints.append({"keypoints": {}})
            smoothed_keypoints[i]["keypoints"][landmark] = {
                "x": float(x_smooth[i]),
                "y": float(y_smooth[i]),
                "visibility": float(vis_smooth[i])
            }
    
    logging.info(f"Smoothed {len(landmark_keys)} landmarks across {num_frames} frames")
    return smoothed_keypoints
