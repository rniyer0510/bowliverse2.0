import cv2
import numpy as np
import json
import logging
import os
import math

logging.basicConfig(level=logging.INFO)

def estimate_pitch_angle(hip_y, foot_y, hip_z, foot_z):
    """
    Estimate pitch angle from hip to foot in the y-z plane, adjusted for vertical.
    Args:
        hip_y, foot_y: Vertical coordinates (image space, y increases downward).
        hip_z, foot_z: Depth coordinates (relative to hip center).
    Returns:
        Pitch angle in degrees, with 0° when vertical (hip above foot).
    """
    dy = hip_y - foot_y
    dz = hip_z - foot_z
    # Subtract 90° to set 0° when hip is above foot (dy > 0, dz ≈ 0)
    return math.degrees(math.atan2(dy, dz)) - 90

def extract_pitch_reference(video_path, keypoints_json, output_json, action_type):
    """
    Estimate pitch orientation angle using hip-to-foot vector in y-z plane.
    Writes pitch angle (in degrees) to output_json.
    """
    if not os.path.exists(keypoints_json):
        logging.error(f"No keypoints found for {video_path}")
        return

    with open(keypoints_json, 'r') as f:
        keypoints = json.load(f)

    if keypoints and any("z" in lm for frame in keypoints for lm in frame.get("keypoints", {}).values()):
        logging.info("Processing 3D keypoints for pitch estimation")
    else:
        logging.warning("No z-coordinates detected in keypoints")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Cannot open {video_path}")
        return

    pitch_angles = []
    frame_count = 0
    min_visibility = 0.4  # Lowered threshold for bowling videos
    max_frames = 200  # Extended to capture bowler approach

    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count < len(keypoints):
            kp = keypoints[frame_count].get("keypoints", {})
            # Primary landmarks: hips (23, 24), feet (27, 28)
            required_landmarks = ["landmark_23", "landmark_24", "landmark_27", "landmark_28"]
            use_fallback = False
            if not all(lm in kp and kp[lm]["visibility"] >= min_visibility for lm in required_landmarks):
                # Fallback to knees (25, 26) if feet are missing
                required_landmarks = ["landmark_23", "landmark_24", "landmark_25", "landmark_26"]
                use_fallback = True
                if not all(lm in kp and kp[lm]["visibility"] >= min_visibility for lm in required_landmarks):
                    logging.debug(f"Frame {frame_count}: Insufficient landmark visibility")
                    frame_count += 1
                    continue

            # Compute average hip position
            hip_avg_y = (kp["landmark_23"]["y"] + kp["landmark_24"]["y"]) / 2
            hip_avg_z = (kp["landmark_23"]["z"] + kp["landmark_24"]["z"]) / 2
            # Compute average lower body position (feet or knees)
            lower_idx = ["landmark_27", "landmark_28"] if not use_fallback else ["landmark_25", "landmark_26"]
            foot_avg_y = (kp[lower_idx[0]]["y"] + kp[lower_idx[1]]["y"]) / 2
            foot_avg_z = (kp[lower_idx[0]]["z"] + kp[lower_idx[1]]["z"]) / 2
            # Log visibility for debugging
            visibilities = {lm: kp[lm]["visibility"] for lm in required_landmarks}
            logging.debug(f"Frame {frame_count}: Using {'knees' if use_fallback else 'feet'}, visibilities: {visibilities}")
            # Estimate pitch angle
            pitch_angle = estimate_pitch_angle(hip_avg_y, foot_avg_y, hip_avg_z, foot_avg_z)
            pitch_angles.append(pitch_angle)
        
        frame_count += 1

    cap.release()

    if not pitch_angles:
        logging.warning(f"No valid hip-to-lower vectors found for {video_path} in first {frame_count} frames")
        return

    pitch_angle = float(np.mean(pitch_angles))
    logging.info(f"Collected {len(pitch_angles)} hip-to-lower vectors, estimated pitch angle: {pitch_angle:.2f} degrees")

    output_data = {
        "crease_front": {
            "left": {"x": 0.26944444444444443, "y": 0.4375},
            "right": {"x": 0.49722222222222223, "y": 0.4375}
        },
        "pitch_angle": pitch_angle,
        "scale_factor": 0.1
    }

    with open(output_json, 'w') as f_out:
        json.dump(output_data, f_out, indent=4)

    logging.info(f"Saved pitch reference to {output_json}")
