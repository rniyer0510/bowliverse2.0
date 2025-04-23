import os
import sys
import json
import pickle
import logging
import numpy as np
from collections import Counter

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.data import load_assessments
from core.hmm_training import train_hmm
from models.frame_detector import FrameDetector
from models.angle_adjuster import AngleAdjuster
from models.biomechanics_refiner import BiomechanicsRefiner
from utils.frame_data import prepare_frame_data
from utils.angle_data import prepare_angle_data
from utils.alignment_data import prepare_style_data

logging.basicConfig(level=logging.INFO)

def train_models(videos_dir, output_dir, db_path, action_type="fast"):
    # Load config
    config_path = os.path.join(os.path.dirname(__file__), "..", "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
    
    # Load assessments
    assessments = load_assessments(db_path, action_type)
    if not assessments:
        logging.error("No assessments found")
        sys.exit(1)
    
    # Load pitch references
    pitch_refs = {}
    for video_id in assessments:
        pitch_json = os.path.join(videos_dir, f"pitch_reference_{video_id}.json")
        if os.path.exists(pitch_json):
            with open(pitch_json, "r") as f:
                pitch_refs[video_id] = json.load(f)
    
    # Prepare frame data
    X_frame, y_frame = [], []
    for video_id, labels in assessments.items():
        keypoints_path = os.path.join(videos_dir, f"bowling_analysis_{video_id}.json")
        if not os.path.exists(keypoints_path):
            logging.warning(f"Skipping {video_id}: missing keypoints")
            continue
        
        with open(keypoints_path, 'r') as f:
            keypoints = json.load(f)
        
        logging.info(f"Video {video_id}: Processed {len(keypoints)} frames")
        
        pitch_ref = pitch_refs.get(video_id, {"pitch_angle": 0, "crease_y": 0.8, "crease_direction": [1, 0]})
        X, y = prepare_frame_data(keypoints, labels, action_type, config, pitch_ref)
        if X.size == 0:
            logging.warning(f"Skipping {video_id}: no valid frame data")
            continue
        X_frame.append(X)
        y_frame.append(y)
    
    X_frame = np.vstack(X_frame) if X_frame else np.array([])
    y_frame = np.hstack(y_frame) if y_frame else np.array([])
    
    if X_frame.size == 0:
        logging.error("No frame data")
        sys.exit(1)
    
    logging.info(f"Frame labels: {Counter(y_frame)}")
    
    # Train frame detector
    frame_detector = FrameDetector(action_type, config)
    frame_detector.train(X_frame, y_frame)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save frame detector
    with open(os.path.join(output_dir, f"frame_detector_{action_type}.pkl"), 'wb') as f:
        pickle.dump(frame_detector, f)
    
    # Prepare angle data
    X_angle, y_angle = [], []
    for video_id, labels in assessments.items():
        keypoints_path = os.path.join(videos_dir, f"bowling_analysis_{video_id}.json")
        if not os.path.exists(keypoints_path):
            continue
        with open(keypoints_path, 'r') as f:
            keypoints = json.load(f)
        pitch_ref = pitch_refs.get(video_id, {"pitch_angle": 0, "crease_y": 0.8, "crease_direction": [1, 0]})
        X, y = prepare_angle_data(keypoints, labels, action_type, config, pitch_ref)
        if X.size == 0:
            continue
        X_angle.append(X)
        y_angle.append(y)
    
    X_angle = np.vstack(X_angle) if X_angle else np.array([])
    y_angle = np.hstack(y_angle) if y_angle else np.array([])
    
    # Train angle adjuster
    angle_adjuster = AngleAdjuster(config)
    if X_angle.size > 0:
        angle_adjuster.fit(X_angle, y_angle)
        with open(os.path.join(output_dir, f"angle_adjuster_elbow_{action_type}.pkl"), 'wb') as f:
            pickle.dump(angle_adjuster, f)
    
    # Prepare alignment data
    X_style, y_style = [], []
    for video_id, labels in assessments.items():
        keypoints_path = os.path.join(videos_dir, f"bowling_analysis_{video_id}.json")
        if not os.path.exists(keypoints_path):
            continue
        with open(keypoints_path, 'r') as f:
            keypoints = json.load(f)
        pitch_ref = pitch_refs.get(video_id, {"pitch_angle": 0, "crease_y": 0.8, "crease_direction": [1, 0]})
        X, y = prepare_style_data(keypoints, labels, action_type, config, pitch_ref)
        if X.size == 0:
            continue
        X_style.append(X)
        y_style.append(y)
    
    X_style = np.vstack(X_style) if X_style else np.array([])
    y_style = np.hstack(y_style) if y_style else np.array([])
    
    # Train biomechanics refiner
    biomechanics_refiner = BiomechanicsRefiner(action_type, config)
    if X_style.size > 0:
        biomechanics_refiner.fit(X_style, y_style)
        with open(os.path.join(output_dir, f"biomechanics_refiner_{action_type}.pkl"), 'wb') as f:
            pickle.dump(biomechanics_refiner, f)
    
    # Train HMM
    hmm = train_hmm(videos_dir, assessments, action_type, pitch_refs, config)
    if hmm is None:
        logging.error("HMM training failed")
        sys.exit(1)
    with open(os.path.join(output_dir, f"hmm_release_elbow_{action_type}.pkl"), 'wb') as f:
        pickle.dump(hmm, f)

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python train_models.py <videos_dir> <output_dir> <db_path> [action_type]")
        sys.exit(1)
    train_models(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else "fast")
