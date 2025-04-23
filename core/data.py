import json
import logging
import sqlite3
import os

logging.basicConfig(level=logging.INFO)

def load_assessments(action_type, db_path="bowliverse.db"):
    """
    Load assessments from SQLite database.
    Args:
        action_type: 'fast' or 'spin'.
        db_path: Path to SQLite database.
    Returns:
        Dict of video_id to assessment data.
    """
    assessments = {}
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT video_id, bfc_frame, ffc_frame, uah_frame, release_frame, action_type
            FROM assessments
            WHERE action_type = ?
        """, (action_type,))
        rows = cursor.fetchall()
        for row in rows:
            video_id, bfc, ffc, uah, release, act_type = row
            assessments[video_id] = {
                "bfc_frame": bfc,
                "ffc_frame": ffc,
                "uah_frame": uah,
                "release_frame": release,
                "action_type": act_type
            }
        conn.close()
        logging.info(f"Loaded {len(assessments)} assessments for {action_type}")
    except Exception as e:
        logging.error(f"Failed to load assessments: {e}")
    return assessments
