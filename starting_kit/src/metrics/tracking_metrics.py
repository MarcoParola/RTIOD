import motmetrics as mm
import numpy as np
import pandas as pd
import os
from typing import Tuple


def compute_mot_metrics(gt_file: str, pred_file: str):
    """
    Compute MOT metrics between GT and prediction files
    (both in MOTChallenge format). Returns a tuple (summary_df, accumulator)
    so callers can aggregate across sequences if needed.
    """

    acc = mm.MOTAccumulator(auto_id=True)

    def load_motchallenge(path: str) -> pd.DataFrame:
        # Parse MOTChallenge-style text file into a DataFrame
        # Expected columns per line: frame,id,x,y,width,height,...
        if not path or not os.path.exists(path):
            return pd.DataFrame()

        rows = []
        with open(path, "r") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(',')
                if len(parts) < 6:
                    continue
                try:
                    frame = int(parts[0])
                    oid = int(parts[1])
                    x = float(parts[2])
                    y = float(parts[3])
                    w = float(parts[4])
                    h = float(parts[5])
                except ValueError:
                    continue

                rows.append({
                    "frame": frame,
                    "id": oid,
                    "X": x,
                    "Y": y,
                    "Width": w,
                    "Height": h,
                })

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df = df.set_index(["frame", "id"]).sort_index()
        return df

    gt = load_motchallenge(gt_file)
    pred = load_motchallenge(pred_file)

    if gt.empty:
        mh = mm.metrics.create()
        empty_summary = mh.compute(mm.MOTAccumulator(), metrics=[], name="tracker")
        return empty_summary, acc

    for frame in sorted(gt.index.get_level_values(0).unique()):
        gt_frame = gt.loc[frame]
        pred_frame = pred.loc[frame] if (not pred.empty and frame in pred.index) else []

        # gt_frame and pred_frame may be Series if only one object present
        if isinstance(gt_frame, pd.Series):
            gt_ids = [gt_frame.name]
            gt_boxes = gt_frame[["X", "Y", "Width", "Height"]].values.reshape(1, 4)
        else:
            gt_ids = gt_frame.index.tolist()
            gt_boxes = gt_frame[["X", "Y", "Width", "Height"]].values

        if isinstance(pred_frame, pd.Series):
            pred_ids = [pred_frame.name]
            pred_boxes = pred_frame[["X", "Y", "Width", "Height"]].values.reshape(1, 4)
        else:
            pred_ids = pred_frame.index.tolist() if len(pred_frame) else []
            pred_boxes = (
                pred_frame[["X", "Y", "Width", "Height"]].values
                if len(pred_frame) else np.empty((0, 4))
            )

        distances = mm.distances.iou_matrix(
            gt_boxes, pred_boxes, max_iou=0.5
        )

        acc.update(gt_ids, pred_ids, distances)

    mh = mm.metrics.create()
    summary = mh.compute(
        acc,
        metrics=[
            "mota", "motp",
            "idf1", "idp", "idr",
            "num_switches", "num_false_positives", "num_misses"
        ],
        name="tracker"
    )

    return summary, acc
