import sqlite3
import logging

logging.basicConfig(level=logging.INFO)

def load_assessments(db_path, action_type):
    """
    Load assessments from SQLite database.
    Args:
        db_path: Path to bowliverse.db.
        action_type: 'fast' or 'spin'.
    Returns:
        Dict of video_id to labels (bfc_frame, ffc_frame, uah_frame, release_frame).
    """
    assessments = {}
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        # Normalize action_type to lowercase to avoid case sensitivity
        query_action_type = action_type.lower()
        cursor.execute(
            "SELECT video_id, bfc_frame, ffc_frame, uah_frame, release_frame FROM assessments WHERE LOWER(action_type) = ?",
            (query_action_type,)
        )
        rows = cursor.fetchall()
        logging.info(f"Queried assessments for action_type='{query_action_type}', found {len(rows)} rows")
        if not rows:
            logging.warning(f"No rows returned for action_type='{query_action_type}'. Checking available action types...")
            cursor.execute("SELECT DISTINCT action_type FROM assessments")
            available_types = [row[0] for row in cursor.fetchall()]
            logging.info(f"Available action types in database: {available_types}")
        for row in rows:
            video_id, bfc_frame, ffc_frame, uah_frame, release_frame = row
            assessments[video_id] = {
                "bfc_frame": bfc_frame,
                "ffc_frame": ffc_frame,
                "uah_frame": uah_frame,
                "release_frame": release_frame
            }
            logging.debug(f"Loaded assessment for video_id='{video_id}': {assessments[video_id]}")
        conn.close()
        if not assessments:
            logging.warning(f"No assessments found for action_type='{query_action_type}'")
        else:
            logging.info(f"Loaded {len(assessments)} assessments for action_type='{query_action_type}'")
    except Exception as e:
        logging.error(f"Failed to load assessments: {e}")
    return assessments
