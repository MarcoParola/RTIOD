import os
import hydra
import numpy as np
import pandas as pd
from math import sqrt
from collections import defaultdict

from src.utils.utils import iou, load_motchallenge


@hydra.main(config_path="../config", config_name="config", version_base="1.3")
def compute_tracking_statistics(args):

    model_name = args.modelCheckpoint.split("/")[-1].split(".")[0]
    gt_dir = os.path.join(args.tracking.trackingData, args.submission.type)

    fps = getattr(args.tracking, "fps", 30)

    per_track_records = []

    for seq_file in os.listdir(gt_dir):
        if not seq_file.endswith(".txt"):
            continue

        gt_file = os.path.join(gt_dir, seq_file)
        df = load_motchallenge(gt_file)

        if df.empty:
            continue

        # ---- sanity check for class info ----
        if "class" not in df.columns:
            raise ValueError(
                "Expected column 'class' in MOT data. "
                "Check load_motchallenge implementation."
            )

        for obj_id in df.index.get_level_values("id").unique():
            track = df.xs(obj_id, level="id").sort_index()

            frames = track.index.values
            boxes = track[["X", "Y", "Width", "Height"]].values
            cls = track["class"].iloc[0]

            # --------------------------------------------------
            # Track length
            # --------------------------------------------------
            track_length = len(frames)

            # --------------------------------------------------
            # Track fragmentation
            # (# of gaps > 1 frame)
            # --------------------------------------------------
            fragmentation = np.sum(np.diff(frames) > 1)

            # --------------------------------------------------
            # Speed + inter-frame IoU
            # --------------------------------------------------
            speeds = []
            ious = []

            for i in range(len(frames) - 1):
                if frames[i + 1] != frames[i] + 1:
                    continue

                x1, y1, w1, h1 = boxes[i]
                x2, y2, w2, h2 = boxes[i + 1]

                c1 = (x1 + w1 / 2, y1 + h1 / 2)
                c2 = (x2 + w2 / 2, y2 + h2 / 2)

                dist = sqrt((c2[0] - c1[0]) ** 2 + (c2[1] - c1[1]) ** 2)
                speeds.append(dist * fps)

                ious.append(iou(boxes[i], boxes[i + 1]))

            per_track_records.append({
                "sequence": seq_file,
                "object_id": obj_id,
                "class": cls,
                "track_length": track_length,
                "fragmentation": fragmentation,
                "mean_speed_px_s": np.mean(speeds) if speeds else 0.0,
                "mean_inter_frame_iou": np.mean(ious) if ious else 0.0,
            })

    stats_df = pd.DataFrame(per_track_records)

    # ======================================================
    # GLOBAL METRICS (all classes)
    # ======================================================
    global_metrics = stats_df.agg({
        "track_length": "mean",
        "fragmentation": "mean",
        "mean_speed_px_s": "mean",
        "mean_inter_frame_iou": "mean"
    }).to_frame(name="global").T

    # ======================================================
    # PER-CLASS METRICS
    # ======================================================
    class_metrics = stats_df.groupby("class").agg({
        "track_length": "mean",
        "fragmentation": "mean",
        "mean_speed_px_s": "mean",
        "mean_inter_frame_iou": "mean"
    })

    print("\n=== Global tracking statistics ===")
    print(global_metrics)

    print("\n=== Per-class tracking statistics ===")
    print(class_metrics)

    return {
        "per_track": stats_df,
        "global": global_metrics,
        "per_class": class_metrics,
    }


if __name__ == "__main__":
    compute_tracking_statistics()
