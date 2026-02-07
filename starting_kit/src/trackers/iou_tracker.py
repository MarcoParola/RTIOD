from .base_tracker import BaseTracker
import numpy as np
from scipy.optimize import linear_sum_assignment


def iou(b1, b2):
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2

    xa = max(x1, x2)
    ya = max(y1, y2)
    xb = min(x1 + w1, x2 + w2)
    yb = min(y1 + h1, y2 + h2)

    inter = max(0, xb - xa) * max(0, yb - ya)
    union = w1 * h1 + w2 * h2 - inter
    return inter / union if union > 0 else 0


class IOUTracker(BaseTracker):
    def __init__(self, iou_thresh=0.3):
        self.iou_thresh = iou_thresh
        self.tracks = {}
        self.next_id = 1

    def reset(self):
        self.tracks = {}
        self.next_id = 1

    def update(self, detections):
        det_boxes = [d["bbox"] for d in detections]
        track_ids = list(self.tracks.keys())
        track_boxes = [self.tracks[i] for i in track_ids]

        matches = set()

        if track_boxes and det_boxes:
            cost = np.zeros((len(track_boxes), len(det_boxes)))
            for i, tb in enumerate(track_boxes):
                for j, db in enumerate(det_boxes):
                    cost[i, j] = 1 - iou(tb, db)

            r, c = linear_sum_assignment(cost)
            for i, j in zip(r, c):
                if 1 - cost[i, j] >= self.iou_thresh:
                    self.tracks[track_ids[i]] = det_boxes[j]
                    matches.add(j)

        for j, det in enumerate(detections):
            if j not in matches:
                self.tracks[self.next_id] = det["bbox"]
                self.next_id += 1

        return [{"id": tid, "bbox": box} for tid, box in self.tracks.items()]
