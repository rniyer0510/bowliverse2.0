import mediapipe as mp
import cv2
import json
import numpy as np
import logging
import os

mp_pose = mp.solutions.pose

logging.basicConfig(level=logging.INFO)

def extract_keypoints(video_path, output_json, pitch_json=None, config=None):
    """
    Extract MediaPipe keypoints, apply pitch correction, save to JSON.
    Args:
        video_path: Path to video file.
        output_json: Path to save keypoint JSON.
        pitch_json: Path to pitch reference JSON.
        config: Configuration parameters.
    Returns:
        List of keypoint frames.
    """
    config = config or {}
    min_detection_confidence = config.get("min_detection_confidence", 0.6)
    min_tracking_confidence = config.get("min_tracking_confidence", 0.6)
    
    pose = mp_pose.Pose(
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence
    )
    cap = cv2.VideoCapture(video_path)
    keypoints_full = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not results.pose_landmarks:
            logging.warning(f"Frame {len(keypoints_full)}: No landmarks detected")
            keypoints_full.append({})
            continue
        landmarks = {
            f"landmark_{i}": {
                "x": lm.x,
                "y": lm.y,
                "visibility": lm.visibility
            } for i, lm in enumerate(results.pose_landmarks.landmark)
        }
        keypoints_full.append({"keypoints": landmarks})
        for i in config.get("key_landmarks", [11, 13, 14, 27, 28, 29, 30, 31, 32]):
            vis = landmarks.get(f"landmark_{i}", {}).get("visibility", 0)
            logging.info(f"Frame {len(keypoints_full)-1}: landmark_{i}_vis={vis:.2f}")
    
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
                for kp in keypoints_full:
                    if "keypoints" in kp:
                        adjust_keypoints(kp["keypoints"], pitch_angle, config)
    
    with open(output_json, 'w') as f:
        json.dump(keypoints_full, f)
    
    return keypoints_full

def adjust_keypoints(keypoints, pitch_angle, config=None):
    """
    Rotate keypoints by pitch_angle (degrees, converted to radians).
    Args:
        keypoints: Dict of keypoints.
        pitch_angle: Angle in degrees.
        config: Configuration parameters.
    """
    config = config or {}
    for lm in keypoints.values():
        x, y = lm["x"], lm["y"]
        lm["x"] = x * np.cos(np.radians(pitch_angle)) + y * np.sin(np.radians(pitch_angle))
        lm["y"] = -x * np.sin(np.radians(pitch_angle)) + y * np.cos(np.radians(pitch_angle))
