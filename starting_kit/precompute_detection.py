from ultralytics import YOLO
import os
import hydra
import torch
import json
from collections import defaultdict

from src.datasets.dataset import COCODataset
from src.utils.motchallenge_utils import parse_sequence_name, parse_frame_id


@hydra.main(config_path='config', config_name='config', version_base="1.3")
def main(args):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(args.modelCheckpoint).to(device)

    if args.submission.type == 'val':
        annotation_file = args.valAnnFile
    elif args.submission.type == 'test':
        annotation_file = args.testAnnFile
    else:
        raise ValueError("submission.type must be 'val' or 'test'")

    annotation_file = os.path.join(args.currentDir, annotation_file)
    dataset = COCODataset(args.dataDir, annotation_file, numClass=args.numClass)

    model_name = args.modelCheckpoint.split('/')[-1].split('.')[0]
    precomputedDetFolder = os.path.join(
        args.dataDir, 
        args.tracking.precomputedDetectionFolder,
        args.submission.type + '_' + model_name)

    os.makedirs(precomputedDetFolder, exist_ok=True)

    sequences = defaultdict(list)

    for idx in range(len(dataset.ids)):
        print(idx)
        img_info = dataset.coco.imgs[dataset.ids[idx]]
        file_name = img_info["file_name"]

        img_path = os.path.join(args.dataDir, file_name)
        if not os.path.exists(img_path):
            continue

        seq_name = parse_sequence_name(file_name)
        frame_id = parse_frame_id(file_name)
        
        results = model(img_path, imgsz=320, conf=0.01, verbose=False)[0]

        if results.boxes is None:
            continue

        boxes = results.boxes.xywh.cpu().numpy()     # x, y, w, h
        scores = results.boxes.conf.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy().astype(int)

        for box, score, cls in zip(boxes, scores, classes):

            x_center, y_center, w, h = box.tolist()
            x = x_center - w/2
            y = y_center - h/2

            sequences[seq_name].append({
                "frame": frame_id,
                "bbox": [x, y, w, h],
                "score": float(score),
                "class": int(cls)
            })

    for seq_name, dets in sequences.items():
        dets.sort(key=lambda x: x["frame"])

        out_file = os.path.join(precomputedDetFolder, f"{seq_name}.json")
        with open(out_file, "w") as f:
            json.dump({
                "sequence": seq_name,
                "detections": dets},
                f, indent=2)

    print(f"Saved predictions for {len(sequences)} sequences")


if __name__ == "__main__":
    main()
