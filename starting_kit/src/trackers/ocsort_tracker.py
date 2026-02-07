from .base_tracker import BaseTracker
import numpy as np


class OCSORTTracker(BaseTracker):
    """
    Simplified OC-SORT: motion consistency without Kalman
    """

    def __init__(self):
        self.tracks = {}
        self.velocities = {}
        self.next_id = 1

    def reset(self):
        self.tracks = {}
        self.velocities = {}
        self.next_id = 1

    def update(self, detections):
        for det in detections:
            best_id = None
            best_dist = float("inf")

            for tid, box in self.tracks.items():
                cx1 = box[0] + box[2] / 2
                cy1 = box[1] + box[3] / 2
                cx2 = det["bbox"][0] + det["bbox"][2] / 2
                cy2 = det["bbox"][1] + det["bbox"][3] / 2
                dist = np.hypot(cx1 - cx2, cy1 - cy2)

                if dist < best_dist:
                    best_dist = dist
                    best_id = tid

            if best_id is not None and best_dist < 50:
                self.tracks[best_id] = det["bbox"]
            else:
                self.tracks[self.next_id] = det["bbox"]
                self.next_id += 1

        return [{"id": tid, "bbox": box} for tid, box in self.tracks.items()]
