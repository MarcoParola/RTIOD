import os
from typing import List, Tuple, Dict
import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from torch import Tensor
import src.utils.transforms as T
import hydra
from collections import defaultdict
import re
import shutil

from src.utils.motchallenge_utils import parse_sequence_name, parse_frame_id


class COCODataset():
    def __init__(self, root: str, annotation: str, numClass: int):
        self.root = root
        self.coco = COCO(annotation)
        self.ids = list(self.coco.imgs.keys())
        self.numClass = numClass
        self.transforms = T.Compose([T.ToTensor()])
        self.newIndex = {}

        for i, (k, _) in enumerate(self.coco.cats.items()):
            self.newIndex[k] = i

    def get_img_info(self, idx: int):
        imgID = self.ids[idx]
        return self.coco.imgs[imgID], imgID


def load_datasets(args):
    num_classes = args.numClass
    data_folder = os.path.join(args.currentDir, args.dataDir)
    train_file = os.path.join(args.currentDir, args.trainAnnFile)
    val_file = os.path.join(args.currentDir, args.valAnnFile)

    train_dataset = COCODataset(data_folder, train_file, num_classes)
    val_dataset = COCODataset(data_folder, val_file, num_classes)
    return train_dataset, val_dataset



def convert_split_to_mot(dataset, output_dir):
    """
    Writes one MOTChallenge .txt file per sequence
    """
    os.makedirs(output_dir, exist_ok=True)

    sequences = defaultdict(list)

    for idx in range(len(dataset.ids)):
        img_info, img_id = dataset.get_img_info(idx)
        file_name = img_info["file_name"]

        seq_name = parse_sequence_name(file_name)
        frame_id = parse_frame_id(file_name)

        annotations = dataset.coco.imgToAnns[img_id]

        for ann in annotations:
            uid = ann["uid"]              # object id
            x, y, w, h = ann["bbox"]      # COCO bbox (absolute pixels)
            category_id = ann["category_id"]

            sequences[seq_name].append(
                (frame_id, uid, category_id, x, y, w, h)
            )

    # Write MOT files
    for seq_name, entries in sequences.items():
        entries.sort(key=lambda x: x[0])  # sort by frame number

        mot_file = os.path.join(output_dir, f"{seq_name}.txt")
        with open(mot_file, "w") as f:
            for frame, uid, category_id, x, y, w, h in entries:
                    # Write in MOTChallenge expected column order:
                    # frame, id, x, y, width, height, -1, -1, -1, -1
                    f.write(
                        f"{frame},{uid},{x},{y},{w},{h},-1,-1,-1,-1\n"
                    )


@hydra.main(config_path="../config", config_name="config", version_base="1.3")
def main(args):

    train_dataset, val_dataset = load_datasets(args)

    tracking_root = args.tracking.trackingData
    os.makedirs(tracking_root, exist_ok=True)

    mot_train_dir = os.path.join(tracking_root, "train")
    mot_val_dir = os.path.join(tracking_root, "val")
    os.makedirs(mot_train_dir, exist_ok=True)
    os.makedirs(mot_val_dir, exist_ok=True)

    convert_split_to_mot(train_dataset, mot_train_dir)
    print("Train MOT conversion done!")

    convert_split_to_mot(val_dataset, mot_val_dir)
    print("Val MOT conversion done!")


if __name__ == "__main__":
    main()

