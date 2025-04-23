import cv2
import numpy as np
import json
import logging
import os
import math

logging.basicConfig(level=logging.DEBUG)

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
    angle = ((angle + 180) % 360) - 180  # Normalize to [-180, 180)
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
    start_frame = 50
    min_visibility = 0.05  # Very low threshold to maximize landmark usage
    debug_frames_dir = os.path.join(os.path.dirname(output_json), "debug_frames")
    os.makedirs(debug_frames_dir, exist_ok=True)
    debug_frame_count = 0
    max_debug_frames = 5
    debug_interval = 100  # Spread debug frames across video

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame_count >= len(keypoints):
            break

        if frame_count < start_frame:
            frame_count += 1
            continue

        kp = keypoints[frame_count].get("keypoints", {})
        hip_landmarks = ["landmark_23", "landmark_24"]
        lower_landmarks = ["landmark_27", "landmark_28", "landmark_25", "landmark_26"]
        hip_idx = None
        lower_idx = None

        # Find valid hip landmark
        for h in hip_landmarks:
            if h in kp and kp[h]["visibility"] >= min_visibility:
                hip_idx = h
                break
        
        # Find valid lower body landmark
        for l in lower_landmarks:
            if l in kp and kp[l]["visibility"] >= min_visibility:
                lower_idx = l
                break

        if hip_idx is None or lower_idx is None:
            visibilities = {lm: kp.get(lm, {}).get("visibility", 0) for lm in hip_landmarks + lower_landmarks}
            logging.debug(f"Frame {frame_count}: Insufficient landmark visibility, visibilities: {visibilities}")
            frame_count += 1
            continue

        # Compute positions
        hip_avg_y = kp[hip_idx]["y"]
        hip_avg_z = kp[hip_idx]["z"]
        hip_avg_x = kp[hip_idx]["x"]
        foot_avg_y = kp[lower_idx]["y"]
        foot_avg_z = kp[lower_idx]["z"]
        foot_avg_x = kp[lower_idx]["x"]

        visibilities = {hip_idx: kp[hip_idx]["visibility"], lower_idx: kp[lower_idx]["visibility"]}
        logging.debug(f"Frame {frame_count}: Using hip {hip_idx}, lower {lower_idx}, visibilities: {visibilities}")

        pitch_angle = estimate_pitch_angle(hip_avg_y, foot_avg_y, hip_avg_z, foot_avg_z)
        logging.debug(f"Frame {frame_count}: Estimated pitch angle {pitch_angle:.2f}")
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
        # Fallback: Try all frames with any hip and lower landmark, ignoring visibility
        fallback_angles = []
        for frame_idx, kp_data in enumerate(keypoints):
            kp = kp_data.get("keypoints", {})
            hip_idx = None
            lower_idx = None
            for h in hip_landmarks:
                if h in kp:
                    hip_idx = h
                    break
            for l in lower_landmarks:
                if l in kp:
                    lower_idx = l
                    break
            if hip_idx and lower_idx:
                hip策

System: Thank you for sharing your updated `core/pitch_calibrator.py`. I’ve reviewed the changes and analyzed why you’re still encountering the `WARNING:root:Zero pitch angle in videos/pitch_reference_cZ-JgPETblw.json` despite the modifications. Since this thread focuses on generating 3D keypoints (`x`, `y`, `z`, `visibility`) for all 33 landmarks using OpenCV and MediaPipe, and your logs confirm successful keypoint generation (`bowling_analysis_cZ-JgPETblw.json` has `z` coordinates), I’ll address the zero pitch angle issue and propose a refined version of `core/pitch_calibrator.py` to ensure a non-zero pitch angle. I’ll also incorporate your changes, provide feedback, and outline steps to validate the fix for `fast_cZ-JgPETblw.mp4` (and potentially `fast_61ukNhaghRo.mp4`).

---

### Analysis of Your Changes
Your updated `core/pitch_calibrator.py` reverts to an earlier version with specific modifications. Below, I compare it to the previous version I provided and evaluate the impact of your changes.

#### Key Changes You Made
1. **Logging Level**:
   - Changed from `logging.basicConfig(level=logging.DEBUG)` to `logging.basicConfig(level=logging.INFO)`.
   - **Impact**: This disables detailed `DEBUG` logging (e.g., `Frame {frame_count}: Insufficient landmark visibility`), which is critical for diagnosing why no valid hip-to-lower vectors are found. The zero pitch angle warning suggests that the code is failing to collect vectors due to missing or low-visibility landmarks.

2. **Frame Range**:
   - Reintroduced `max_frames = 300` (previously unlimited) and kept `start_frame = 50`.
   - **Impact**: Limiting analysis to frames 50–300 may exclude frames where the bowler’s landmarks (hips 23, 24; feet 27, 28; knees 25, 26) are most visible, especially if the bowler’s approach occurs later in the video.

3. **Visibility Threshold**:
   - Set `min_visibility = 0.2` (previously 0.1).
   - **Impact**: A stricter threshold may reject landmarks with low confidence, common in bowling videos due to occlusions or side angles, increasing the likelihood of no valid vectors.

4. **Landmark Selection**:
   - Reverted to ordered fallbacks:
     - Primary: Hips (23, 24), feet (27, 28).
     - Fallback 1: Hips (23, 24), knees (25, 26).
     - Fallback 2: Left hip (23), left foot (27).
     - Fallback 3: Left hip (23), left knee (25).
   - Removed the flexible approach (any hip 23 or 24 with any lower 27, 28, 25, or 26).
   - **Impact**: Less flexible than the previous version, as it doesn’t try right-side landmarks (e.g., 24, 28) independently, potentially missing valid vectors if the left side is occluded.

5. **Fallback Pitch Angle**:
   - Added `pitch_angle = 6.5` if no vectors are found:
     ```python
     if not pitch_angles:
         pitch_angle = 6.5
         logging.warning(f"No valid hip-to-lower vectors found for {video_path} in {frame_count} frames. Forcing fallback pitch angle: {pitch_angle}")
     ```
   - **Impact**: Prevents a zero pitch angle, avoiding the warning in `utils/keypoints_utils2.py`:
     ```python
     if pitch_angle == 0:
         logging.warning(f"Zero pitch angle in {pitch_json}")
     ```
     However, 6.5° is arbitrary and may not reflect the actual camera tilt or bowler posture. A data-driven fallback would be more accurate.

6. **Debug Logging**:
   - Added `logging.debug(f"Frame {frame_count}: Estimated pitch angle {pitch_angle:.2f}")`.
   - **Impact**: Useful for tracking per-frame angles, but requires `DEBUG` logging to be visible.

7. **Removed Tilt Logging**:
   - Removed `logging.info(f"Pitch angle of {pitch_angle:.2f} means the camera is {'tilted up' if pitch_angle > 0 else 'tilted down' if pitch_angle < 0 else 'level'}")`.
   - **Impact**: Loses interpretability of the pitch angle’s physical meaning (camera tilt direction).

#### Why Zero Pitch Angle Persists
The log `WARNING:root:Zero pitch angle in videos/pitch_reference_cZ-JgPETblw.json` indicates that `pitch_reference_cZ-JgPETblw.json` likely has `"pitch_angle": 0.0`, despite your fallback. This occurs because:
- The JSON is written with the computed or fallback `pitch_angle`, but `utils/keypoints_utils2.py` checks the JSON’s `pitch_angle` during keypoint adjustment and logs the warning if it’s 0.
- Your code sets `pitch_angle = 6.5` when `pitch_angles` is empty, but the warning suggests the JSON being read still has 0.0. This could happen if:
  - The `pitch_reference_cZ-JgPETblw.json` from a previous run (with `pitch_angle = 0.0`) wasn’t overwritten.
  - The pipeline reuses an existing `pitch_reference_cZ-JgPETblw.json` before generating a new one with `pitch_angle = 6.5`.
- The root cause of no vectors (`pitch_angles` empty) is likely:
  - **Low Visibility**: Landmarks (23, 24, 27, 28, 25, 26) have visibility < 0.2 in frames 50–300.
  - **Frame Range**: Frames 50–300 may not capture the bowler’s approach where landmarks are most visible.
  - **Strict Fallbacks**: The ordered fallback logic misses valid right-side landmarks if the left side is occluded.

---

### Recommendations and Proposed Fixes
Your changes improve robustness by adding a fallback pitch angle, but the zero pitch angle warning persists due to:
1. An outdated `pitch_reference_cZ-JgPETblw.json` being read.
2. No valid vectors being collected due to strict visibility (`min_visibility = 0.2`), limited frame range (50–300), and inflexible landmark selection.

I’ll propose a refined `core/pitch_calibrator.py` that:
- Builds on your code, retaining the 6.5° fallback as a last resort.
- Reintroduces flexible landmark selection (any hip 23 or 24 with any lower 27, 28, 25, or 26).
- Lowers `min_visibility` to 0.05 to accept more landmarks.
- Removes `max_frames` to analyze all frames.
- Re-enables `DEBUG` logging for visibility diagnostics.
- Adds a data-driven fallback if no vectors are found, using low-visibility landmarks.
- Restores tilt logging for interpretability.
- Ensures the JSON is always overwritten to avoid stale data.

This addresses the zero pitch angle warning, ensures a non-zero `pitch_angle` (either computed or data-driven), and preserves your debug frame logic.

---

### Updated Code
Only `core/pitch_calibrator.py` is modified. `utils/keypoints_utils2.py` and `scripts/process_videos.py` remain unchanged, as they correctly handle 3D keypoint generation and Y-Z rotation (as fixed previously).

##### `core/pitch_calibrator.py`
```bash
cat > core/pitch_calibrator.py << 'EOF'
import cv2
import numpy as np
import json
import logging
import os
import math

logging.basicConfig(level=logging.DEBUG)

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
    angle = ((angle + 180) % 360) - 180  # Normalize to [-180, 180)
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
    start_frame = 50
    min_visibility = 0.05  # Very low threshold to maximize landmark usage
    debug_frames_dir = os.path.join(os.path.dirname(output_json), "debug_frames")
    os.makedirs(debug_frames_dir, exist_ok=True)
    debug_frame_count = 0
    max_debug_frames = 5
    debug_interval = 100  # Spread debug frames across video

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame_count >= len(keypoints):
            break

        if frame_count < start_frame:
            frame_count += 1
            continue

        kp = keypoints[frame_count].get("keypoints", {})
        hip_landmarks = ["landmark_23", "landmark_24"]
        lower_landmarks = ["landmark_27", "landmark_28", "landmark_25", "landmark_26"]
        hip_idx = None
        lower_idx = None

        # Find valid hip landmark
        for h in hip_landmarks:
            if h in kp and kp[h]["visibility"] >= min_visibility:
                hip_idx = h
                break
        
        # Find valid lower body landmark
        for l in lower_landmarks:
            if l in kp and kp[l]["visibility"] >= min_visibility:
                lower_idx = l
                break

        if hip_idx is None or lower_idx is None:
            visibilities = {lm: kp.get(lm, {}).get("visibility", 0) for lm in hip_landmarks + lower_landmarks}
            logging.debug(f"Frame {frame_count}: Insufficient landmark visibility, visibilities: {visibilities}")
            frame_count += 1
            continue

        # Compute positions
        hip_avg_y = kp[hip_idx]["y"]
        hip_avg_z = kp[hip_idx]["z"]
        hip_avg_x = kp[hip_idx]["x"]
        foot_avg_y = kp[lower_idx]["y"]
        foot_avg_z = kp[lower_idx]["z"]
        foot_avg_x = kp[lower_idx]["x"]

        visibilities = {hip_idx: kp[hip_idx]["visibility"], lower_idx: kp[lower_idx]["visibility"]}
        logging.debug(f"Frame {frame_count}: Using hip {hip_idx}, lower {lower_idx}, visibilities: {visibilities}")

        pitch_angle = estimate_pitch_angle(hip_avg_y, foot_avg_y, hip_avg_z, foot_avg_z)
        logging.debug(f"Frame {frame_count}: Estimated pitch angle {pitch_angle:.2f}")
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
        # Fallback: Try all frames with any hip and lower landmark, ignoring visibility
        fallback_angles = []
        for frame_idx, kp_data in enumerate(keypoints):
            kp = kp_data.get("keypoints", {})
            hip_idx = None
            lower_idx = None
            for h in hip_landmarks:
                if h in kp:
                    hip_idx = h
                    break
            for l in lower_landmarks:
                if l in kp:
                    lower_idx = l
                    break
            if hip_idx and lower_idx:
                hip_avg_y = kp[hip_idx]["y"]
                hip_avg_z = kp[hip_idx]["z"]
                foot_avg_y = kp[lower_idx]["y"]
                foot_avg_z = kp[lower_idx]["z"]
                angle = estimate_pitch_angle(hip_avg_y, foot_avg_y, hip_avg_z, foot_avg_z)
                fallback_angles.append(angle)
        if fallback_angles:
            pitch_angle = float(np.mean(fallback_angles))
            logging.info(f"Used {len(fallback_angles)} low-visibility vectors for fallback, estimated pitch angle: {pitch_angle:.2f} degrees")
        else:
            pitch_angle = 6.5  # Your fallback
            logging.warning(f"No landmarks found even ignoring visibility. Forcing fallback pitch angle: {pitch_angle:.2f}")
    else:
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
