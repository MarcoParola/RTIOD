from .base_tracker import BaseTracker


class DeepSORTTracker(BaseTracker):
    """
    Appearance-aware tracker (embeddings can be plugged later)
    """

    def __init__(self):
        self.tracks = {}
        self.next_id = 1

    def reset(self):
        self.tracks = {}
        self.next_id = 1

    def update(self, detections):
        # Placeholder for appearance matching
        for det in detections:
            self.tracks[self.next_id] = det["bbox"]
            self.next_id += 1

        return [{"id": tid, "bbox": box} for tid, box in self.tracks.items()]
