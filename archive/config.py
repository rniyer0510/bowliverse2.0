
# Configuration settings
class CONFIG:
    @staticmethod
    def get_min_resolution():
        return 720

    @staticmethod
    def get_landmarks(section):
        if section == "pitch_reference":
            return {"left_heel": 27, "right_heel": 28}
        elif section == "elbow":
            return {"shoulder": 11, "elbow": 13}
        return {}

    @staticmethod
    def get_action(action_type):
        return {"key_angles": ["elbow"]} if action_type == "fast" else {"key_angles": ["elbow"]}
