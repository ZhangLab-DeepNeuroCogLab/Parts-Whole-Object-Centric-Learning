import json
import pytorch_lightning as pl
import os
import torch
import torchvision
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import VisionDataset
from typing import List, Optional, Callable, Tuple, Any
from pycocotools.coco import COCO
from torchvision.transforms import ToTensor, Compose, Resize, Normalize, RandomCrop, RandomHorizontalFlip, CenterCrop

class CocoDataModule(pl.LightningDataModule):

    def __init__(self,
                 args):
        super().__init__()
        self.root = args.data_root
        self.num_workers = args.num_workers
        self.batch_size = args.batch_size
        self.shuffle = False
        train_image_transforms = Compose([Resize(256, interpolation=3),
                                          CenterCrop(224),
                                          RandomHorizontalFlip(),
                                          ToTensor(),
                                          Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                          ])
        val_image_transforms = Compose([CenterCrop(224),
                                        Resize(320),
                                        ToTensor(),
                                        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                        ])
        val_target_transforms = Compose([CenterCrop(224),
                                         Resize(320, interpolation=torchvision.transforms.InterpolationMode.NEAREST),
                                         ToTensor(),
                                         ])
        self.coco_train = COCOSegmentation(self.root, image_set="train2017", train_transform=train_image_transforms)
        self.coco_val = COCOSegmentation(self.root, image_set="val2017", val_transform=val_image_transforms, target_transform=val_target_transforms)

        print(f"Train size {len(self.coco_train)}")
        print(f"Val size {len(self.coco_val)}")

    def __len__(self):
        return len(self.file_list)

    def train_dataloader(self):
        return DataLoader(self.coco_train, batch_size=self.batch_size,
                          shuffle=self.shuffle, num_workers=self.num_workers,
                          drop_last=True, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.coco_val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,
                          drop_last=False, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.coco_val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,
                          drop_last=False, pin_memory=True)


class COCOSegmentation(VisionDataset):

    def __init__(
            self,
            root: str,
            image_set: str = "train2017",
            train_transform: Optional[Callable] = None,
            val_transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
    ):
        super(COCOSegmentation, self).__init__(root, train_transform, val_transform, target_transform)
        self.root = root
        self.image_set = image_set
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.target_transform = target_transform
        assert self.image_set in ["train2017", "val2017"]

        self.coco = COCO(os.path.join(root, "annotations", "instances_" + image_set + ".json"))
        self.image_ids = list(self.coco.imgs.keys())

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img_id = self.image_ids[index]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.root, self.image_set, img_info['file_name'])
        img = Image.open(img_path).convert("RGB")

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        annotations = self.coco.loadAnns(ann_ids)

        masks = [self.coco.annToMask(ann) for ann in annotations]
        merged_mask = np.transpose(np.zeros(img.size[:2]))
        for i, mask in enumerate(masks):
            merged_mask = merged_mask + mask * i
        mask = Image.fromarray(merged_mask.astype(np.uint8))
        assert img.size == mask.size

        if self.image_set == 'val2017':
            if self.val_transform is not None:
                img = self.val_transform(img)
            if self.target_transform is not None:
                mask = self.target_transform(mask)
            return {
                'image': img,
                'mask': mask,
            }
        elif self.image_set == "train2017":
            if self.train_transform is not None:
                img = self.train_transform(img)
            return {
                'image': img,
            }