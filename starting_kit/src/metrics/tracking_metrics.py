import os
import numpy as np
import pandas as pd
import motmetrics as mm
from typing import Tuple
from collections import defaultdict
from scipy.optimize import linear_sum_assignment

from src.utils.utils import iou, load_motchallenge


def compute_hota_manual(
    gt: pd.DataFrame,
    pred: pd.DataFrame,
    alphas=np.linspace(0.05, 0.95, 19)
) -> float:

    hota_scores = []

    for alpha in alphas:
        matches = []
        gt_total = 0
        pr_total = 0

        for frame in sorted(gt.index.get_level_values(0).unique()):
            gt_f = gt.loc[frame]
            pr_f = pred.loc[frame] if frame in pred.index else []

            # GT
            if isinstance(gt_f, pd.Series):
                gt_ids = [gt_f.name]
                gt_boxes = gt_f[["X", "Y", "Width", "Height"]].values.reshape(1, 4)
            else:
                gt_ids = gt_f.index.tolist()
                gt_boxes = gt_f[["X", "Y", "Width", "Height"]].values

            # Predictions
            if isinstance(pr_f, pd.Series):
                pr_ids = [pr_f.name]
                pr_boxes = pr_f[["X", "Y", "Width", "Height"]].values.reshape(1, 4)
            elif len(pr_f):
                pr_ids = pr_f.index.tolist()
                pr_boxes = pr_f[["X", "Y", "Width", "Height"]].values
            else:
                pr_ids, pr_boxes = [], np.empty((0, 4))

            gt_total += len(gt_ids)
            pr_total += len(pr_ids)

            if not gt_ids or not pr_ids:
                continue

            cost = np.ones((len(gt_ids), len(pr_ids)))
            for i, g in enumerate(gt_boxes):
                for j, p in enumerate(pr_boxes):
                    cost[i, j] = 1 - iou(g, p)

            rows, cols = linear_sum_assignment(cost)
            for r, c in zip(rows, cols):
                if 1 - cost[r, c] >= alpha:
                    matches.append((gt_ids[r], pr_ids[c], frame))

        TP = len(matches)
        FP = pr_total - TP
        FN = gt_total - TP

        if TP == 0:
            hota_scores.append(0.0)
            continue

        DetA = TP / (TP + FP + FN)

        gt_frames = defaultdict(set)
        pr_frames = defaultdict(set)
        pair_frames = defaultdict(set)

        for g, p, f in matches:
            gt_frames[g].add(f)
            pr_frames[p].add(f)
            pair_frames[(g, p)].add(f)

        assa = 0.0
        for (g, p), tpa_frames in pair_frames.items():
            TPA = len(tpa_frames)
            FPA = len(pr_frames[p] - tpa_frames)
            FNA = len(gt_frames[g] - tpa_frames)
            assa += TPA / (TPA + FPA + FNA)

        AssA = assa / TP
        hota_scores.append(np.sqrt(DetA * AssA))

    return float(np.mean(hota_scores))


# ============================================================
# MAIN METRIC FUNCTION
# ============================================================

def compute_mot_metrics(gt_file: str, pred_file: str):
    gt = load_motchallenge(gt_file)
    pred = load_motchallenge(pred_file)

    summary = {}
    
    acc = mm.MOTAccumulator(auto_id=True)
    '''
    if gt.empty:
        mh = mm.metrics.create()
        return mh.compute(mm.MOTAccumulator(), metrics=[], name="tracker"), acc

    for frame in sorted(gt.index.get_level_values(0).unique()):
        gt_f = gt.loc[frame]
        pr_f = pred.loc[frame] if frame in pred.index else []

        if isinstance(gt_f, pd.Series):
            gt_ids = [gt_f.name]
            gt_boxes = gt_f[["X", "Y", "Width", "Height"]].values.reshape(1, 4)
        else:
            gt_ids = gt_f.index.tolist()
            gt_boxes = gt_f[["X", "Y", "Width", "Height"]].values

        if isinstance(pr_f, pd.Series):
            pr_ids = [pr_f.name]
            pr_boxes = pr_f[["X", "Y", "Width", "Height"]].values.reshape(1, 4)
        elif len(pr_f):
            pr_ids = pr_f.index.tolist()
            pr_boxes = pr_f[["X", "Y", "Width", "Height"]].values
        else:
            pr_ids, pr_boxes = [], np.empty((0, 4))

        distances = mm.distances.iou_matrix(gt_boxes, pr_boxes, max_iou=0.5)
        acc.update(gt_ids, pr_ids, distances)

    mh = mm.metrics.create()
    summary = mh.compute(
        acc,
        metrics=[
            "mota", "motp",
            "idf1", "idp", "idr",
            "num_switches",
            "num_false_positives",
            "num_misses",
        ],
        name="tracker"
    )
    '''
    # ---- Manual HOTA ----
    summary["hota"] = compute_hota_manual(gt, pred)

    return summary, acc
