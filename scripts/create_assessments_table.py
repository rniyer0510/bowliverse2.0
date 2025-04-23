import sqlite3
import logging

logging.basicConfig(level=logging.INFO)

def create_assessments_table(db_path):
    """
    Create assessments table in bowliverse.db and insert sample data.
    Args:
        db_path: Path to bowliverse.db.
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Create assessments table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS assessments (
                video_id TEXT PRIMARY KEY,
                bfc_frame INTEGER,
                ffc_frame INTEGER,
                uah_frame INTEGER,
                release_frame INTEGER,
                action_type TEXT
            )
        """)

        # Insert sample data for action_type='fast'
        sample_data = [
            (
                "61ukNhaghRo",
                20,   # bfc_frame
                50,   # ffc_frame
                180,  # uah_frame
                224,  # release_frame
                "fast"
            )
        ]

        cursor.executemany("""
            INSERT OR REPLACE INTO assessments (video_id, bfc_frame, ffc_frame, uah_frame, release_frame, action_type)
            VALUES (?, ?, ?, ?, ?, ?)
        """, sample_data)

        conn.commit()
        logging.info(f"Created assessments table and inserted {len(sample_data)} records")
    except Exception as e:
        logging.error(f"Failed to create assessments table: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python create_assessments_table.py <db_path>")
        sys.exit(1)
    create_assessments_table(sys.argv[1])
