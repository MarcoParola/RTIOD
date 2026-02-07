import os
from typing import List, Tuple, Dict
import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from torch import Tensor
from torch.utils.data.dataset import Dataset
import src.utils.transforms as T
import hydra

'''
class COCODataset(Dataset):
    def __init__(self, root: str, annotation: str, numClass: int):
        self.root = root
        self.coco = COCO(annotation)
        self.ids = list(self.coco.imgs.keys())
        self.numClass = numClass
        self.transforms = T.Compose([
            T.ToTensor()
        ])

        self.newIndex = {}
        classes = []
        for i, (k, v) in enumerate(self.coco.cats.items()):
            self.newIndex[k] = i
            classes.append(v['name'])

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> Tuple[Tensor, dict]:
        imgID = self.ids[idx]
        imgInfo = self.coco.imgs[imgID]        
        imgPath = os.path.join(self.root, imgInfo['file_name'])
        image = Image.open(imgPath).convert('RGB')

        annotations = self.loadAnnotations(imgID, imgInfo['width'], imgInfo['height'])

        if len(annotations) == 0:
            targets = {
                'boxes': torch.zeros(1, 4, dtype=torch.float32),
                'labels': torch.as_tensor([self.numClass], dtype=torch.int64),}
        else:
            targets = {
                'boxes': torch.as_tensor(annotations[..., :-1], dtype=torch.float32),
                'labels': torch.as_tensor(annotations[..., -1], dtype=torch.int64),}

        image, targets = self.transforms(image, targets)
        return image, targets

    def get_img_path(self, idx: int) -> str:
        imgID = self.ids[idx]
        imgInfo = self.coco.imgs[imgID]        
        imgPath = os.path.join(self.root, imgInfo['file_name'])
        return imgPath

    def loadAnnotations(self, imgID: int, imgWidth: int, imgHeight: int) -> np.ndarray:
        ans = []
        for annotation in self.coco.imgToAnns[imgID]:
            cat = self.newIndex[annotation['category_id']]
            bbox = annotation['bbox']
            bbox = [val / imgHeight if i % 2 else val / imgWidth for i, val in enumerate(bbox)]
            ans.append(bbox + [cat])

        return np.asarray(ans)


# MARK: - collate functions for dataloaders
def collate_fn_coco(batch: List[Tuple[Tensor, dict]]) -> Tuple[Tensor, Tuple[Dict[str, Tensor]]]:
    batch = tuple(zip(*batch))
    return torch.stack(batch[0]), batch[1]


def load_datasets(args):
    num_classes = args.numClass
    data_folder = args.dataDir
    train_file = args.trainAnnFile
    val_file = args.valAnnFile
    test_file = args.testAnnFile

    data_folder = os.path.join(args.currentDir, data_folder)
    train_file = os.path.join(args.currentDir, train_file)
    val_file = os.path.join(args.currentDir, val_file)
    test_file = os.path.join(args.currentDir, test_file)

    train_dataset = COCODataset(data_folder, train_file, num_classes)
    val_dataset = COCODataset(data_folder, val_file, num_classes)
    test_dataset = COCODataset(data_folder, test_file, num_classes)
    collate_fn = collate_fn_coco

    return train_dataset, val_dataset, test_dataset, collate_fn
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

# TEST MAIN
@hydra.main(config_path='../../config', config_name='config', version_base="1.3")
def main(args):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from torch.utils.data import DataLoader

    num_classes = 4
    data_folder = 'data'
    data_file = 'data/TestNoLabels.json'
    data_folder = os.path.join(args.currentDir, data_folder)
    data_file = os.path.join(args.currentDir, data_file)
    dataset = COCODataset(data_folder, data_file, num_classes)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn_coco)
    print(dataset.__len__())


    for i in range(5):
        img, target = dataset.__getitem__(i)
        
        img = img.permute(1, 2, 0).numpy()
        fig, ax = plt.subplots(1)
        ax.imshow(img)
        plt.show()
    

    '''
    for images, targets in dataloader:
        for i in range(len(images)):
            img = images[i].permute(1, 2, 0).numpy()
            fig, ax = plt.subplots(1)
            ax.imshow(img)

            boxes = targets[i]['boxes'].numpy()
            labels = targets[i]['labels'].numpy()
            for j in range(boxes.shape[0]):
                if labels[j] == num_classes:
                    continue
                box = boxes[j]
                x, y, w, h = box
                x *= img.shape[1]
                y *= img.shape[0]
                w *= img.shape[1]
                h *= img.shape[0]
                rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(rect)

            plt.show()
        break
    '''  
if __name__ == '__main__':
    main()