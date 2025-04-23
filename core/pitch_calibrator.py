import cv2
import numpy as np
import json
import logging
import os

logging.basicConfig(level=logging.INFO)

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

    # Validate 3D keypoints
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
    min_visibility = 0.6  # Minimum visibility threshold for landmarks

    while cap.isOpened() and frame_count < 100:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count < len(keypoints):
            kp = keypoints[frame_count].get("keypoints", {})
            # Check for required landmarks with sufficient visibility
            required_landmarks = ["landmark_23", "landmark_24", "landmark_27", "landmark_28"]
            if all(lm in kp and kp[lm]["visibility"] >= min_visibility for lm in required_landmarks):
                # Compute average hip position (landmarks 23 and 24)
                hip_avg_y = (kp["landmark_23"]["y"] + kp["landmark_24"]["y"]) / 2
                hip_avg_z = (kp["landmark_23"]["z"] + kp["landmark_24"]["z"]) / 2
                # Compute average foot position (landmarks 27 and 28)
                foot_avg_y = (kp["landmark_27"]["y"] + kp["landmark_28"]["y"]) / 2
                foot_avg_z = (kp["landmark_27"]["z"] + kp["landmark_28"]["z"]) / 2
                # Calculate pitch angle in y-z plane
                pitch_angle = np.degrees(np.arctan2(hip_avg_y - foot_avg_y, hip_avg_z - foot_avg_z)) % 360
                pitch_angles.append(pitch_angle)
        
        frame_count += 1

    cap.release()

    if not pitch_angles:
        logging.warning(f"No valid hip-to-foot vectors found for {video_path} in first {frame_count} frames")
        return

    # Average pitch angles to reduce noise
    pitch_angle = float(np.mean(pitch_angles))
    logging.info(f"Collected {len(pitch_angles)} hip-to-foot vectors, estimated pitch angle: {pitch_angle:.2f} degrees")

    # Preserve existing JSON structure
    output_data = {
        "crease_front": {
            "left": {"x": 0.26944444444444443, "y": 0.4375},  # From provided pitch_reference_cZ-JgPETblw.json
            "right": {"x": 0.49722222222222223, "y": 0.4375}
        },
        "pitch_angle": pitch_angle,
        "scale_factor": 0.1
    }

    with open(output_json, 'w') as f_out:
        json.dump(output_data, f_out, indent=4)

    logging.info(f"Saved pitch reference to {output_json}")
