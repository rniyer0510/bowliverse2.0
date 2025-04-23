import cv2
import mediapipe as mp
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)

def extract_keypoints(video_path, config=None):
    """
    Extract keypoints from a video using MediaPipe Pose.
    Args:
        video_path: Path to video file.
        config: Configuration parameters (e.g., visibility threshold).
    Returns:
        List of keypoint dictionaries per frame.
    """
    config = config or {}
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Cannot open video {video_path}")
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
                    "x": lm.x,
                    "y": lm.y,
                    "visibility": lm.visibility
                }
        keypoints.append(frame_keypoints)
    
    cap.release()
    pose.close()
    
    logging.info(f"Extracted {len(keypoints)} frames of keypoints from {video_path}")
    return keypoints

def adjust_keypoints(keypoints, pitch_angle):
    """
    Adjust keypoints for pitch angle.
    Args:
        keypoints: Dict of keypoints (landmark_i: {x, y, visibility}).
        pitch_angle: Angle in degrees.
    Returns:
        Adjusted keypoints dict.
    """
    if not keypoints:
        return {}
    
    adjusted = {}
    theta = np.radians(pitch_angle)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    
    for lm_key, lm_data in keypoints.items():
        x = lm_data.get("x", 0)
        y = lm_data.get("y", 0)
        visibility = lm_data.get("visibility", 0)
        
        # Rotate coordinates counterclockwise
        x_new = x * cos_theta + y * sin_theta
        y_new = -x * sin_theta + y * cos_theta
        
        adjusted[lm_key] = {
            "x": float(x_new),
            "y": float(y_new),
            "visibility": float(visibility)
        }
    
    return adjusted
