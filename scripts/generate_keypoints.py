import json
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)

def generate_keypoints(output_path, num_frames=240, num_landmarks=33):
    keypoints = []
    for frame in range(num_frames):
        frame_keypoints = {"keypoints": {}}
        for lm in range(num_landmarks):
            frame_keypoints["keypoints"][f"landmark_{lm}"] = {
                "x": float(np.random.uniform(0, 1)),
                "y": float(np.random.uniform(0, 1)),
                "visibility": float(np.random.uniform(0.8, 1.0))
            }
        keypoints.append(frame_keypoints)
    
    try:
        with open(output_path, 'w') as f:
            json.dump(keypoints, f, indent=4)
        logging.info(f"Generated keypoints saved to {output_path}")
    except Exception as e:
        logging.error(f"Failed to save keypoints: {e}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python -m scripts.generate_keypoints <output_path>")
        sys.exit(1)
    generate_keypoints(sys.argv[1])
