from .base_tracker import BaseTracker
from .iou_tracker import iou
import numpy as np
from scipy.optimize import linear_sum_assignment


class SORTTracker(BaseTracker):
    def __init__(self, iou_thresh=0.3, max_age=3):
        self.iou_thresh = iou_thresh
        self.max_age = max_age
        self.tracks = {}
        self.ages = {}
        self.next_id = 1

    def reset(self):
        self.tracks = {}
        self.ages = {}
        self.next_id = 1

    def update(self, detections):
        for tid in self.ages:
            self.ages[tid] += 1

        det_boxes = [d["bbox"] for d in detections]
        track_ids = list(self.tracks.keys())
        track_boxes = [self.tracks[i] for i in track_ids]

        if track_boxes and det_boxes:
            cost = np.zeros((len(track_boxes), len(det_boxes)))
            for i, tb in enumerate(track_boxes):
                for j, db in enumerate(det_boxes):
                    cost[i, j] = 1 - iou(tb, db)

            r, c = linear_sum_assignment(cost)
            for i, j in zip(r, c):
                if 1 - cost[i, j] >= self.iou_thresh:
                    tid = track_ids[i]
                    self.tracks[tid] = det_boxes[j]
                    self.ages[tid] = 0

        for det in det_boxes:
            self.tracks[self.next_id] = det
            self.ages[self.next_id] = 0
            self.next_id += 1

        self.tracks = {
            tid: box for tid, box in self.tracks.items()
            if self.ages[tid] <= self.max_age
        }

        return [{"id": tid, "bbox": box} for tid, box in self.tracks.items()]
