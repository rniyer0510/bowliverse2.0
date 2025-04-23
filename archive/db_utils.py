
import sqlite3
import json
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def bootstrap_assessments(db_path):
    assessments = {}
    db_path = os.path.expanduser(db_path)
    annotation_path = os.path.expanduser("~/bowliverse_ai/training_labels.json")

    # Try loading from SQLite DB
    try:
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute("""
            SELECT video_id, type, bfc_frame, ffc_frame, upper_frame, release_frame,
                   elbow_extension, stride_length, style
            FROM Assessments
        """)
        for row in c.fetchall():
            assessments[row[0]] = {
                "video_id": row[0],
                "type": row[1],
                "bfc_frame": row[2],
                "ffc_frame": row[3],
                "uah_frame": row[4],  # renamed from upper_frame
                "release_frame": row[5],
                "elbow_extension": row[6],
                "stride_length": row[7],
                "style": row[8]
            }
        conn.close()
        logging.info(f"Loaded {len(assessments)} assessments from DB.")
        return assessments
    except Exception as e:
        logging.warning(f"DB load failed: {e}")

    # Fallback: Load from JSON
    try:
        with open(annotation_path, "r") as f:
            assessments = json.load(f)
        logging.info(f"Loaded {len(assessments)} assessments from JSON.")
    except Exception as e:
        logging.error(f"Failed to load JSON fallback: {e}")

    return assessments
