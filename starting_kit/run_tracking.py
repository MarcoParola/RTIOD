import os
import json
import hydra
from collections import defaultdict

from src.trackers.base_tracker import get_tracker

@hydra.main(config_path='config', config_name='config', version_base="1.3")
def run_tracking(args):

    model_name = args.modelCheckpoint.split('/')[-1].split('.')[0]
    det_dir = os.path.join(
        args.dataDir, 
        args.tracking.precomputedDetectionFolder,
        args.submission.type + '_' + model_name + '_th' + str(args.confidence_threshold).replace('.','_'))

    output_dir = os.path.join(
        args.tracking.trackingPredictionFolder, 
        args.tracking.tracker + '_' + model_name + '_th' + str(args.confidence_threshold).replace('.','_'),
        args.submission.type)
    os.makedirs(output_dir, exist_ok=True)
    
    tracker = get_tracker(args.tracking.tracker)

    for det_file in os.listdir(det_dir):
        if not det_file.endswith(".json"):
            continue

        with open(os.path.join(det_dir, det_file), "r") as f:
            data = json.load(f)

        sequence = data["sequence"]
        detections = data["detections"]

        tracker.reset()

        frames = defaultdict(list)
        for det in detections:
            frames[det["frame"]].append(det)

        out_path = os.path.join(output_dir, f"{sequence}.txt")

        with open(out_path, "w") as f:
            for frame in sorted(frames.keys()):
                tracks = tracker.update(frames[frame])

                for trk in tracks:
                    x, y, w, h = trk["bbox"]
                    f.write(
                        f"{frame},{trk['id']},{x},{y},{w},{h},-1,-1,-1,-1\n"
                    )


if __name__ == "__main__":
    run_tracking()
