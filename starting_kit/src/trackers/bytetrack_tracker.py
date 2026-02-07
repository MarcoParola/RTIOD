from .base_tracker import BaseTracker
from .iou_tracker import iou
import numpy as np


class ByteTrackTracker(BaseTracker):
    def __init__(self, high_thresh=0.6, low_thresh=0.1):
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.tracks = {}
        self.next_id = 1

    def reset(self):
        self.tracks = {}
        self.next_id = 1

    def update(self, detections):
        high = [d for d in detections if d["score"] >= self.high_thresh]
        low = [d for d in detections if self.low_thresh <= d["score"] < self.high_thresh]

        for det in high:
            self.tracks[self.next_id] = det["bbox"]
            self.next_id += 1

        for det in low:
            for tid, box in self.tracks.items():
                if iou(box, det["bbox"]) > 0.5:
                    self.tracks[tid] = det["bbox"]
                    break

        return [{"id": tid, "bbox": box} for tid, box in self.tracks.items()]
