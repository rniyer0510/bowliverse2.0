import json
import numpy as np

def generate_keypoints_file(output_path, num_frames=240):
    """
    Generate a sample keypoints file with 240 frames for bowling analysis.
    Args:
        output_path: Path to save the JSON file.
        num_frames: Number of frames to generate.
    """
    keypoints_data = []
    for frame_idx in range(num_frames):
        t = frame_idx / (num_frames - 1)  # Normalized time [0, 1]
        # Simulate keypoint movement for a bowling action
        shoulder = {"x": 0.5 + 0.1 * t, "y": 0.3 - 0.05 * t, "visibility": 0.9}
        elbow = {"x": 0.6 + 0.15 * t, "y": 0.5 + 0.1 * np.sin(np.pi * t), "visibility": 0.9}
        wrist = {"x": 0.7 + 0.2 * t, "y": 0.7 + 0.1 * np.cos(np.pi * t), "visibility": 0.9}
        back_ankle = {"x": 0.3 + 0.05 * t, "y": 0.8 - 0.02 * t, "visibility": 0.9}
        front_ankle = {"x": 0.4 + 0.1 * t, "y": 0.8 - 0.03 * t, "visibility": 0.9}
        foot_end = {"x": 0.45 + 0.1 * t, "y": 0.8 - 0.03 * t, "visibility": 0.9}
        
        frame = {
            "keypoints": {
                "landmark_11": shoulder,
                "landmark_12": {"x": shoulder["x"] + 0.2, "y": shoulder["y"], "visibility": 0.9},
                "landmark_13": elbow,
                "landmark_14": wrist,
                "landmark_27": front_ankle,
                "landmark_30": foot_end,
                "landmark_31": back_ankle
            }
        }
        keypoints_data.append(frame)
    
    with open(output_path, 'w') as f:
        json.dump(keypoints_data, f, indent=2)
    print(f"Generated keypoints file with {num_frames} frames at {output_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python generate_keypoints.py <output_path>")
        sys.exit(1)
    generate_keypoints_file(sys.argv[1])
