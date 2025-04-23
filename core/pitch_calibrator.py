
import cv2
import numpy as np
import json
import logging
import os

logging.basicConfig(level=logging.INFO)

def extract_pitch_reference(video_path, keypoints_json, output_json, action_type):
    """
    Estimate pitch orientation angle using initial foot positions.
    Writes pitch angle (in degrees) to output_json.
    """
    if not os.path.exists(keypoints_json):
        logging.error(f"No keypoints found for {video_path}")
        return

    with open(keypoints_json, 'r') as f:
        keypoints = json.load(f)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Cannot open {video_path}")
        return

    vectors = []
    frame_count = 0

    while cap.isOpened() and frame_count < 100:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count < len(keypoints):
            kp = keypoints[frame_count].get("keypoints", {})
            if "landmark_27" in kp and "landmark_28" in kp:
                vec = np.array([
                    kp["landmark_28"]["x"] - kp["landmark_27"]["x"],
                    kp["landmark_28"]["y"] - kp["landmark_27"]["y"]
                ])
                vectors.append(vec)
        frame_count += 1

    cap.release()

    if not vectors:
        logging.warning("No valid foot vectors found.")
        return

    mean_vec = np.mean(vectors, axis=0)
    pitch_angle = np.degrees(np.arctan2(mean_vec[1], mean_vec[0])) % 360

    with open(output_json, 'w') as f_out:
        json.dump({"pitch_angle": pitch_angle}, f_out)

    logging.info(f"Pitch angle estimated: {pitch_angle:.2f} degrees")
