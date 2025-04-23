import os
import json
import pickle
import logging
from core.data import load_assessments
from core.hmm_training import train_hmm, prepare_hmm_data
from models.frame_detector import FrameDetector
from models.angle_adjuster import AngleAdjuster
from models.biomechanics_refiner import BiomechanicsRefiner
from utils.frame_data import prepare_frame_data
from utils.angle_data import prepare_angle_data
from utils.alignment_data import prepare_alignment_data

logging.basicConfig(level=logging.INFO)

def train_models(keypoints_dir, output_dir, action_type, config=None, pitch_refs=None):
    """
    Train FrameDetector, AngleAdjuster, BiomechanicsRefiner, and HMM models.
    Args:
        keypoints_dir: Directory with keypoint JSONs.
        output_dir: Directory to save trained models.
        action_type: 'fast' or 'spin'.
        config: Configuration parameters.
        pitch_refs: Dict of video_id to pitch reference.
    """
    config = config or {}
    pitch_refs = pitch_refs or {}
    
    # Load assessments
    assessments = load_assessments(action_type, db_path="bowliverse.db")
    if not assessments:
        logging.error("No assessments available for training")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Train FrameDetector
    frame_detector = FrameDetector(action_type, config)
    X_frame, y_frame = prepare_frame_data(keypoints_dir, assessments, action_type, config, pitch_refs)
    if X_frame.size > 0:
        frame_detector.fit(X_frame, y_frame)
        with open(os.path.join(output_dir, f"frame_detector_{action_type}.pkl"), 'wb') as f:
            pickle.dump(frame_detector, f)
        logging.info(f"FrameDetector saved for {action_type}")
    
    # Train AngleAdjuster
    angle_adjuster = AngleAdjuster(action_type, config)
    X_angle, y_angle = prepare_angle_data(keypoints_dir, assessments, action_type, config, pitch_refs)
    if X_angle.size > 0:
        angle_adjuster.fit(X_angle, y_angle)
        with open(os.path.join(output_dir, f"angle_adjuster_{action_type}.pkl"), 'wb') as f:
            pickle.dump(angle_adjuster, f)
        logging.info(f"AngleAdjuster saved for {action_type}")
    
    # Train BiomechanicsRefiner
    biomechanics_refiner = BiomechanicsRefiner(action_type, config)
    X_align, y_align = prepare_alignment_data(keypoints_dir, assessments, action_type, config, pitch_refs)
    if X_align.size > 0:
        biomechanics_refiner.fit(X_align, y_align)
        with open(os.path.join(output_dir, f"biomechanics_refiner_{action_type}.pkl"), 'wb') as f:
            pickle.dump(biomechanics_refiner, f)
        logging.info(f"BiomechanicsRefiner saved for {action_type}")
    
    # Train HMM
    X_hmm = prepare_hmm_data(keypoints_dir, assessments, action_type, config, pitch_refs)
    if X_hmm is not None:
        hmm_model = train_hmm(X_hmm)
        if hmm_model:
            with open(os.path.join(output_dir, f"hmm_release_elbow_{action_type}.pkl"), 'wb') as f:
                pickle.dump(hmm_model, f)
            logging.info(f"HMM model saved for {action_type}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 4:
        print("Usage: python -m scripts.train_models <keypoints_dir> <output_dir> <action_type>")
        sys.exit(1)
    train_models(sys.argv[1], sys.argv[2], sys.argv[3])
