import os
import pytorch_lightning as pl

from typing import Optional, Callable
from PIL import Image
import torch.nn as nn
import glob
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
from typing import Tuple, Any
from torchvision import transforms
from torchvision.transforms import ToTensor, Compose, Resize, Normalize, RandomCrop
import random

class MOVIDataset(VisionDataset):

    def __init__(
            self,
            root: str,
            split: str,
            image_set: str = "train",
            train_transform: Optional[Callable] = None,
            val_transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None
    ):
        super(MOVIDataset, self).__init__(root, train_transform, val_transform, target_transform)
        self.image_set = image_set
        self.root = os.path.join(root, split, self.image_set)
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.target_transform = target_transform
        if self.image_set == "train":
            self.images = []
            clips = glob.glob(self.root + "/rgb/" + "*")
            for c in clips:
                self.images.extend(random.sample(glob.glob(c + "/*.jpg"), 9))
        elif self.image_set == "validation":
            self.images = glob.glob(self.root + "/rgb/" + "*/*.jpg")
            self.masks = glob.glob(self.root + "/seg/" + "*/*.png")
        else:
            raise ValueError(f"No support for image set {self.image_set}")

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img = Image.open(self.images[index]).convert('RGB')
        if self.image_set == "validation":
            mask = Image.open(self.images[index].replace('rgb', 'seg').replace(".jpg", ".png"))
            if self.val_transform is not None:
                img = self.val_transform(img)
            if self.target_transform is not None:
                mask = self.target_transform(mask)
            mask = (mask * 255).int()
            return {
                'image': img,
                'mask': mask,
            }
        elif "train" in self.image_set:
            if self.train_transform is not None:
                img = self.train_transform(img)
            return {
                'image': img
            }

    def __len__(self) -> int:
        return len(self.images)

class MOVIDataModule(pl.LightningDataModule):

    def __init__(self,
                 args):
        super().__init__()
        self.root = args.data_root
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.drop_last = False
        self.shuffle = False
        self.split_name = args.split_name

        train_image_transforms = Compose([RandomCrop(224),
                                          ToTensor(),
                                          Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                          ])

        val_image_transforms = Compose([Resize(128),
                                        ToTensor(),
                                        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                        ])
        val_target_transforms = Compose([Resize(128),
                                         ToTensor(),
                                         ])
        self.movi_train = MOVIDataset(root=self.root, split=self.split_name, image_set='train', train_transform=train_image_transforms)
        self.movi_val = MOVIDataset(root=self.root, split=self.split_name, image_set='validation', val_transform=val_image_transforms, target_transform=val_target_transforms)

    def train_dataloader(self):
        return DataLoader(self.movi_train, batch_size=self.batch_size,
                          shuffle=self.shuffle, num_workers=self.num_workers,
                          drop_last=self.drop_last, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.movi_val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,
                          drop_last=self.drop_last, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.movi_val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,
                          drop_last=self.drop_last, pin_memory=True)