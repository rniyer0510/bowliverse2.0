import cv2
import numpy as np
import json
import logging
import os
import math

logging.basicConfig(level=logging.INFO)

def estimate_pitch_angle(hip_y, foot_y, hip_z, foot_z):
    """
    Estimate pitch angle from hip to foot in the Y-Z plane, adjusted for vertical.
    Args:
        hip_y, foot_y: Vertical coordinates (image space, y increases downward).
        hip_z, foot_z: Depth coordinates (relative to hip center).
    Returns:
        Pitch angle in degrees, with 0° when vertical (hip above foot).
    """
    dy = hip_y - foot_y
    dz = hip_z - foot_z
    angle = math.degrees(math.atan2(dy, dz)) - 90
    # Normalize to [-180, 180)
    angle = ((angle + 180) % 360) - 180
    return angle

def extract_pitch_reference(video_path, keypoints_json, output_json, action_type):
    """
    Estimate pitch orientation angle using hip-to-foot vector in Y-Z plane.
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
    start_frame = 50  # Skip initial setup frames
    max_frames = 400  # Extended to capture bowler approach
    min_visibility = 0.2  # Lowered for bowling videos
    debug_frames_dir = os.path.join(os.path.dirname(output_json), "debug_frames")
    os.makedirs(debug_frames_dir, exist_ok=True)
    debug_frame_count = 0
    max_debug_frames = 5
    debug_interval = max_frames // max_debug_frames

    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count < start_frame or frame_count >= len(keypoints):
            frame_count += 1
            continue

        kp = keypoints[frame_count].get("keypoints", {})
        # Try single-side first: left hip (23) and left foot (27)
        required_landmarks = ["landmark_23", "landmark_27"]
        single_side = True
        use_fallback = False
        if not all(lm in kp and kp[lm]["visibility"] >= min_visibility for lm in required_landmarks):
            # Fallback to left knee (25)
            required_landmarks = ["landmark_23", "landmark_25"]
            use_fallback = True
            if not all(lm in kp and kp[lm]["visibility"] >= min_visibility for lm in required_landmarks):
                # Try both sides: hips (23, 24), feet (27, 28)
                required_landmarks = ["landmark_23", "landmark_24", "landmark_27", "landmark_28"]
                single_side = False
                use_fallback = False
                if not all(lm in kp and kp[lm]["visibility"] >= min_visibility for lm in required_landmarks):
                    # Fallback to knees (25, 26)
                    required_landmarks = ["landmark_23", "landmark_24", "landmark_25", "landmark_26"]
                    use_fallback = True
                    if not all(lm in kp and kp[lm]["visibility"] >= min_visibility for lm in required_landmarks):
                        logging.debug(f"Frame {frame_count}: Insufficient landmark visibility")
                        frame_count += 1
                        continue

        # Compute positions
        if single_side:
            hip_avg_y = kp["landmark_23"]["y"]
            hip_avg_z = kp["landmark_23"]["z"]
            hip_avg_x = kp["landmark_23"]["x"]
            foot_avg_y = kp[required_landmarks[1]]["y"]  # Left foot (27) or knee (25)
            foot_avg_z = kp[required_landmarks[1]]["z"]
            foot_avg_x = kp[required_landmarks[1]]["x"]
        else:
            hip_avg_y = (kp["landmark_23"]["y"] + kp["landmark_24"]["y"]) / 2
            hip_avg_z = (kp["landmark_23"]["z"] + kp["landmark_24"]["z"]) / 2
            hip_avg_x = (kp["landmark_23"]["x"] + kp["landmark_24"]["x"]) / 2
            lower_idx = ["landmark_27", "landmark_28"] if not use_fallback else ["landmark_25", "landmark_26"]
            foot_avg_y = (kp[lower_idx[0]]["y"] + kp[lower_idx[1]]["y"]) / 2
            foot_avg_z = (kp[lower_idx[0]]["z"] + kp[lower_idx[1]]["z"]) / 2
            foot_avg_x = (kp[lower_idx[0]]["x"] + kp[lower_idx[1]]["x"]) / 2

        visibilities = {lm: kp[lm]["visibility"] for lm in required_landmarks}
        logging.debug(f"Frame {frame_count}: Using {'left-side' if single_side else 'knees' if use_fallback else 'feet'}, visibilities: {visibilities}")

        pitch_angle = estimate_pitch_angle(hip_avg_y, foot_avg_y, hip_avg_z, foot_avg_z)
        pitch_angles.append(pitch_angle)

        # Visual verification
        if debug_frame_count < max_debug_frames and frame_count % debug_interval == 0:
            h, w = frame.shape[:2]
            x1 = int(hip_avg_x * w)
            y1 = int(hip_avg_y * h)
            x2 = int(foot_avg_x * w)
            y2 = int(foot_avg_y * h)
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Pitch: {pitch_angle:.2f}°", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            debug_path = os.path.join(debug_frames_dir, f"debug_frame_{frame_count:03d}.jpg")
            cv2.imwrite(debug_path, frame)
            logging.info(f"Saved debug frame {debug_path}")
            debug_frame_count += 1

        frame_count += 1

    cap.release()

    if not pitch_angles:
        logging.warning(f"No valid hip-to-lower vectors found for {video_path} in {frame_count} frames")
        return

    pitch_angle = float(np.mean(pitch_angles))
    logging.info(f"Collected {len(pitch_angles)} hip-to-lower vectors, estimated pitch angle: {pitch_angle:.2f} degrees")
    logging.info(f"Pitch angle of {pitch_angle:.2f} means the camera is {'tilted up' if pitch_angle > 0 else 'tilted down' if pitch_angle < 0 else 'level'}")

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
