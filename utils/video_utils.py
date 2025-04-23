import cv2
import logging

logging.basicConfig(level=logging.INFO)

def is_video_good(video_path, config=None):
    """
    Check if video meets quality standards.
    Args:
        video_path: Path to video file.
        config: Configuration parameters (e.g., min_resolution).
    Returns:
        Boolean indicating if video is good.
    """
    config = config or {}
    min_resolution = config.get("min_resolution", 720)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Cannot open video {video_path}")
        return False
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    cap.release()
    
    if height < min_resolution:
        logging.warning(f"Video {video_path} resolution {height}p is below minimum {min_resolution}p")
        return False
    
    if fps < 24:
        logging.warning(f"Video {video_path} FPS {fps} is below minimum 24")
        return False
    
    return True
