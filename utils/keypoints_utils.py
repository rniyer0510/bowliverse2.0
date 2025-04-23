import cv2
import mediapipe as mp
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)

def extract_keypoints(video_path, config=None):
    """
    Extract 3D keypoints from a video using MediaPipe Pose.
    Args:
        video_path: Path to video file.
        config: Configuration parameters (e.g., visibility threshold).
    Returns:
        List of keypoint dictionaries per frame with 3D coordinates.
    """
    # Check MediaPipe version
    import mediapipe
    if mediapipe.__version__ < '0.8.9':
        logging.error("MediaPipe version >= 0.8.9 required for 3D pose estimation")
        return []

    config = config or {}
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,  # Enable 3D pose estimation
        min_detection_confidence=0.6,  # Harmonized with keypoints_utils2.py
        min_tracking_confidence=0.6,   # Harmonized with keypoints_utils2.py
        smooth_landmarks=True
    )
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Cannot open video {video_path}")
        pose.close()
        return []
    
    keypoints = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        
        frame_keypoints = {"keypoints": {}}
        if results.pose_landmarks:
            for i, lm in enumerate(results.pose_landmarks.landmark):
                frame_keypoints["keypoints"][f"landmark_{i}"] = {
                    "x": float(lm.x),
                    "y": float(lm.y),
                    "z": float(lm.z),  # Include z-coordinate
                    "visibility": float(lm.visibility)
                }
            logging.info(f"Frame {len(keypoints)}: Extracted 3D keypoints for {len(frame_keypoints['keypoints'])} landmarks")
        else:
            logging.warning(f"Frame {len(keypoints)}: No landmarks detected")
            frame_keypoints["keypoints"] = {}
        
        keypoints.append(frame_keypoints)
    
    cap.release()
    pose.close()
    
    logging.info(f"Extracted {len(keypoints)} frames of 3D keypoints from {video_path}")
    return keypoints

def adjust_keypoints(keypoints, pitch_angle):
    """
    Adjust 3D keypoints for pitch angle.
    Args:
        keypoints: Dict of landmarks with x, y, z, visibility.
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
            x, y, z = data.get("x", 0), data.get("y", 0), data.get("z", 0)
            visibility = data.get("visibility", 0)
            # Rotate x, y; z remains unchanged
            x_new = x * cos_a + y * sin_a
            y_new = -x * sin_a + y * cos_a
            z_new = z
            adjusted[lm] = {
                "x": float(x_new),
                "y": float(y_new),
                "z": float(z_new),
                "visibility": visibility
            }
        return adjusted
    except Exception as e:
        logging.error(f"Failed to adjust keypoints: {e}")
        return keypoints
