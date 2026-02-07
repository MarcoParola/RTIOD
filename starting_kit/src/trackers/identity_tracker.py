from .base_tracker import BaseTracker

class IdentityTracker(BaseTracker):
    """
    Baseline tracker: assigns a new ID to every detection
    (useful sanity check)
    """

    def __init__(self):
        self.next_id = 1

    def reset(self):
        self.next_id = 1

    def update(self, detections):
        tracks = []
        for det in detections:
            tracks.append({
                "id": self.next_id,
                "bbox": det["bbox"]
            })
            self.next_id += 1
        return tracks