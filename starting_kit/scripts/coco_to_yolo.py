import os
from typing import List, Tuple, Dict
import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from torch import Tensor
import src.utils.transforms as T
import hydra

from src.datasets.dataset import load_datasets

'''
class COCODataset():
    def __init__(self, root: str, annotation: str, numClass: int):
        self.root = root
        self.coco = COCO(annotation)
        self.ids = list(self.coco.imgs.keys())
        self.numClass = numClass
        self.transforms = T.Compose([T.ToTensor()])
        self.newIndex = {}
        classes = []
        for i, (k, v) in enumerate(self.coco.cats.items()):
            self.newIndex[k] = i
            classes.append(v['name'])

    def get_item_for_yolo(self, idx: int) -> Tuple[Tensor, dict]:
        imgID = self.ids[idx]
        imgInfo = self.coco.imgs[imgID]        
        imgPath = imgInfo['file_name']
        annotations = self.loadAnnotations(imgID, imgInfo['width'], imgInfo['height'])
        if len(annotations) == 0:
            targets = {
                'boxes': torch.zeros(1, 4, dtype=torch.float32),
                'labels': torch.as_tensor([self.numClass], dtype=torch.int64),}
        else:
            targets = {
                'boxes': torch.as_tensor(annotations[..., :-1], dtype=torch.float32),
                'labels': torch.as_tensor(annotations[..., -1], dtype=torch.int64),}

        return imgPath, targets


    def loadAnnotations(self, imgID: int, imgWidth: int, imgHeight: int) -> np.ndarray:
        ans = []
        for annotation in self.coco.imgToAnns[imgID]:
            cat = self.newIndex[annotation['category_id']]
            bbox = annotation['bbox']
            bbox = [val / imgHeight if i % 2 else val / imgWidth for i, val in enumerate(bbox)]
            ans.append(bbox + [cat])

        return np.asarray(ans)


def load_datasets(args):
    num_classes = args.numClass
    data_folder = args.dataDir
    train_file = args.trainAnnFile
    val_file = args.valAnnFile
    data_folder = os.path.join(args.currentDir, data_folder)
    train_file = os.path.join(args.currentDir, train_file)
    val_file = os.path.join(args.currentDir, val_file)

    train_dataset = COCODataset(data_folder, train_file, num_classes)
    val_dataset = COCODataset(data_folder, val_file, num_classes)
    return train_dataset, val_dataset
'''

@hydra.main(config_path='../config', config_name='config', version_base="1.3")
def main(args):

    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import shutil

    num_classes = 4
    data_folder = args.dataDir
    width, height = 384, 288

    # create labels and images folders
    labels_folder = os.path.join(data_folder, 'labels')
    images_folder = os.path.join(data_folder, 'images')
    os.makedirs(labels_folder, exist_ok=True)
    os.makedirs(images_folder, exist_ok=True)

    train, val = load_datasets(args)
    print(f'Train size: {len(train.ids)}')
    print(f'Val size: {len(val.ids)}')

    # CONVERT TRAIN
    # create a subfolder in labels and images with name 'train'
    labels_train_folder = os.path.join(labels_folder, 'train')
    images_train_folder = os.path.join(images_folder, 'train')
    os.makedirs(labels_train_folder, exist_ok=True)
    os.makedirs(images_train_folder, exist_ok=True)

    for i in range(len(train.ids)):
        imgPath, target = train.get_item_for_yolo(i)
        #print(imgPath, target)

        img_Path = os.path.join(data_folder, imgPath)
        yolo_name = f'{imgPath.replace("/", "_")}'
        yolo_file_name = os.path.join(images_train_folder, yolo_name)
        
        if os.path.exists(img_Path) and target is not None:
            shutil.copy(img_Path, yolo_file_name)
            # create a txt file with the same name in labels folder
            yolo_label_name = yolo_name.replace('.jpg', '.txt')
            yolo_label_file = os.path.join(labels_train_folder, yolo_label_name)
            with open(yolo_label_file, 'w') as f:
                for box, lbl in zip(target['boxes'].numpy(), target['labels'].numpy()):
                    x_center = (box[0] + box[2] / 2) 
                    y_center = (box[1] + box[3] / 2) 
                    w = box[2]
                    h = box[3]
                    f.write(f'{lbl-1} {x_center} {y_center} {w} {h}\n')
        else:
            print(f'File {img_Path} does not exist')
            continue
    print('Train conversion done!')

    # CONVERT VAL
    # create a subfolder in labels and images with name 'val'
    labels_val_folder = os.path.join(labels_folder, 'val')
    images_val_folder = os.path.join(images_folder, 'val')
    os.makedirs(labels_val_folder, exist_ok=True)
    os.makedirs(images_val_folder, exist_ok=True)

    for i in range(len(val.ids)):
        imgPath, target = val.get_item_for_yolo(i)
        img_Path = os.path.join(data_folder, imgPath)
        yolo_name = f'{imgPath.replace("/", "_")}'
        yolo_file_name = os.path.join(images_val_folder, yolo_name)

        if os.path.exists(img_Path) and target is not None:
            shutil.copy(img_Path, yolo_file_name)
            # create a txt file with the same name in labels folder
            yolo_label_name = yolo_name.replace('.jpg', '.txt')
            yolo_label_file = os.path.join(labels_val_folder, yolo_label_name)

            with open(yolo_label_file, 'w') as f:
                for box, lbl in zip(target['boxes'].numpy(), target['labels'].numpy()):
                    x_center = (box[0] + box[2] / 2)
                    y_center = (box[1] + box[3] / 2)
                    w = box[2]
                    h = box[3]
                    f.write(f'{lbl-1} {x_center} {y_center} {w} {h}\n')
        else:
            print(f'File {img_Path} does not exist')
            continue
    print('Val conversion done!')

    
if __name__ == '__main__':
    main()