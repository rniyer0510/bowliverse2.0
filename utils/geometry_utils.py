import numpy as np

def calculate_angle(p1, p2, p3):
    """
    Calculate angle between three points (p1–p2–p3).
    Args:
        p1, p2, p3: Points with 'x' and 'y' coordinates.
    Returns:
        Angle in degrees.
    """
    v1 = np.array([p1["x"] - p2["x"], p1["y"] - p2["y"]])
    v2 = np.array([p3["x"] - p2["x"], p3["y"] - p2["y"]])
    dot_product = np.dot(v1, v2)
    norms = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norms == 0:
        return 0.0
    angle = np.degrees(np.arccos(np.clip(dot_product / norms, -1.0, 1.0))) % 360
    return angle
