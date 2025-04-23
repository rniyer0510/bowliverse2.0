import mediapipe as mp
import cv2
import json
import numpy as np
import logging
import os
from packaging import version

mp_pose = mp.solutions.pose

logging.basicConfig(level=logging.INFO)

def extract_keypoints(video_path, output_json, pitch_json=None, config=None):
    """
    Extract 3D MediaPipe keypoints, apply pitch correction, save to JSON.
    Args:
        video_path: Path to video file.
        output_json: Path to save keypoint JSON.
        pitch_json: Path to pitch reference JSON.
        config: Configuration parameters.
    Returns:
        List of keypoint frames with 3D coordinates.
    """
    # Check MediaPipe version
    import mediapipe
    mp_version = mediapipe.__version__
    logging.info(f"Using MediaPipe version {mp_version}")
    if version.parse(mp_version) < version.parse('0.8.9'):
        logging.warning(f"MediaPipe version {mp_version} detected; >= 0.8.9 recommended for reliable 3D pose estimation")

    config = config or {}
    min_detection_confidence = config.get("min_detection_confidence", 0.6)
    min_tracking_confidence = config.get("min_tracking_confidence", 0.6)
    
    # Initialize MediaPipe Pose for 3D output
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
        smooth_landmarks=True
    )
    
    # Open video with OpenCV
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Cannot open video {video_path}")
        pose.close()
        return []
    
    keypoints_full = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        
        frame_keypoints = {"keypoints": {}}
        if results.pose_landmarks:
            for i, lm in enumerate(results.pose_landmarks.landmark):
                frame_keypoints["keypoints"][f"landmark_{i}"] = {
                    "x": float(lm.x),
                    "y": float(lm.y),
                    "z": float(lm.z),
                    "visibility": float(lm.visibility)
                }
            logging.info(f"Frame {len(keypoints_full)}: Extracted 3D keypoints for {len(frame_keypoints['keypoints'])} landmarks")
        else:
            logging.warning(f"Frame {len(keypoints_full)}: No landmarks detected")
            frame_keypoints["keypoints"] = {}
        
        keypoints_full.append(frame_keypoints)
    
    cap.release()
    pose.close()
    
    if pitch_json and os.path.exists(pitch_json):
        with open(pitch_json, 'r') as f:
            pitch_data = json.load(f)
            pitch_angle = pitch_data.get("pitch_angle", 0)
            if pitch_angle == 0:
                logging.warning(f"Zero pitch angle in {pitch_json}")
            else:
                logging.info(f"Applying pitch correction: {pitch_angle:.2f} degrees")
                corrected_frames = 0
                for kp in keypoints_full:
                    if "keypoints" in kp:
                        kp["keypoints"] = adjust_keypoints(kp["keypoints"], pitch_angle, config)
                        corrected_frames += 1
                logging.info(f"Applied pitch correction to {corrected_frames} frames")
    
    try:
        with open(output_json, 'w') as f:
            json.dump(keypoints_full, f, indent=4)
        logging.info(f"Saved 3D keypoints to {output_json}")
    except Exception as e:
        logging.error(f"Failed to save keypoints to {output_json}: {e}")
    
    return keypoints_full

def adjust_keypoints(keypoints, pitch_angle, config=None):
    """
    Rotate 3D keypoints by pitch_angle (degrees, converted to radians).
    Args:
        keypoints: Dict of keypoints with x, y, z, visibility.
        pitch_angle: Angle in degrees.
        config: Configuration parameters.
    Returns:
        Adjusted keypoints dict.
    """
    config = config or {}
    if not keypoints:
        return keypoints
    
    adjusted = {}
    theta = np.radians(pitch_angle)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    
    for lm_key, lm_data in keypoints.items():
        x, y, z = lm_data.get("x", 0), lm_data.get("y", 0), lm_data.get("z", 0)
        visibility = lm_data.get("visibility", 0)
        
        x_new = x * cos_theta + y * sin_theta
        y_new = -x * sin_theta + y * cos_theta
        z_new = z
        
        adjusted[lm_key] = {
            "x": float(x_new),
            "y": float(y_new),
            "z": float(z_new),
            "visibility": float(visibility)
        }
    
    return adjusted
