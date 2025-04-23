import numpy as np
import logging

logging.basicConfig(level=logging.INFO)

def calculate_angle(p1, p2, p3, pitch_angle=0):
    v1 = np.array([p1["x"] - p2["x"], p1["y"] - p2["y"]])
    v2 = np.array([p3["x"] - p2["x"], p3["y"] - p2["y"]])
    dot_product = np.dot(v1, v2)
    norms = np.linalg.norm(v1) * np.linalg.norm(v2)
    angle = np.degrees(np.arccos(dot_product / norms if norms > 0 else 1)) % 360
    adjusted_angle = angle - pitch_angle
    logging.info(f"Calculated angle: raw={angle:.2f}, pitch={pitch_angle:.2f}, adjusted={adjusted_angle:.2f}")
    return adjusted_angle
