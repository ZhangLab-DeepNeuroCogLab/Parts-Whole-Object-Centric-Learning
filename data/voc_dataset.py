import os
import pytorch_lightning as pl

from typing import Optional, Callable
from PIL import Image
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
from typing import Tuple, Any
import torchvision
from torchvision.transforms import ToTensor, Compose, Resize, Normalize, RandomCrop, RandomHorizontalFlip, CenterCrop

class VOCDataModule(pl.LightningDataModule):

    CLASS_IDX_TO_NAME = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
                         'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
                         'train', 'tvmonitor']

    def __init__(self,
                 args):
        """
        Data module for PVOC data. "trainaug" and "train" are valid train_splits.
        If return_masks is set train_image_transform should be callable with imgs and masks or None.
        """
        super().__init__()
        self.root = os.path.join(args.data_root, "VOCSegmentation")
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.return_masks = False
        self.drop_last = False
        self.shuffle = False

        train_image_transforms = Compose([Resize(256, interpolation=3),
                                          RandomCrop(224),
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
        self.voc_train = VOCDataset(root=self.root, image_set='trainaug', transforms=train_image_transforms,
                                    return_masks=self.return_masks)
        self.voc_val = VOCDataset(root=self.root, image_set='val', transform=val_image_transforms,
                                  target_transform=val_target_transforms)

    def __len__(self):
        return len(self.voc_train)

    def class_id_to_name(self, i: int):
        return self.CLASS_IDX_TO_NAME[i]

    def setup(self, stage: Optional[str] = None):
        print(f"Train size {len(self.voc_train)}")
        print(f"Val size {len(self.voc_val)}")

    def train_dataloader(self):
        return DataLoader(self.voc_train, batch_size=self.batch_size,
                          shuffle=self.shuffle, num_workers=self.num_workers,
                          drop_last=self.drop_last, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.voc_val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,
                          drop_last=self.drop_last, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.voc_val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,
                          drop_last=self.drop_last, pin_memory=True)


class TrainXVOCValDataModule(pl.LightningDataModule):
    # wrapper class to allow for training on a different data set

    def __init__(self, train_datamodule: pl.LightningDataModule, val_datamodule: VOCDataModule):
        super().__init__()
        self.train_datamodule = train_datamodule
        self.val_datamodule = val_datamodule

    def setup(self, stage: str = None):
        self.train_datamodule.setup(stage)
        self.val_datamodule.setup(stage)

    def class_id_to_name(self, i: int):
        return self.val_datamodule.class_id_to_name(i)

    def __len__(self):
        return len(self.train_datamodule)

    def train_dataloader(self):
        return self.train_datamodule.train_dataloader()

    def val_dataloader(self):
        return self.val_datamodule.val_dataloader()


class VOCDataset(VisionDataset):

    def __init__(
            self,
            root: str,
            image_set: str = "trainaug",
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            transforms: Optional[Callable] = None,
            return_masks: bool = False
    ):
        super(VOCDataset, self).__init__(root, transforms, transform, target_transform)
        self.image_set = image_set
        if self.image_set == "trainaug" or self.image_set == "train":
            seg_folder = "SegmentationClassAug"
        elif self.image_set == "val":
            seg_folder = "SegmentationClass"
        else:
            raise ValueError(f"No support for image set {self.image_set}")
        seg_dir = os.path.join(root, seg_folder)
        image_dir = os.path.join(root, 'images')
        if not os.path.isdir(seg_dir) or not os.path.isdir(image_dir) or not os.path.isdir(root):
            raise RuntimeError('Dataset not found or corrupted.')
        splits_dir = os.path.join(root, 'sets')
        split_f = os.path.join(splits_dir, self.image_set.rstrip('\n') + '.txt')

        with open(os.path.join(split_f), "r") as f:
            file_names = [x.strip() for x in f.readlines()]

        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]
        self.masks = [os.path.join(seg_dir, x + ".png") for x in file_names]
        self.return_masks = return_masks

        assert all([Path(f).is_file() for f in self.masks]) and all([Path(f).is_file() for f in self.images])

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img = Image.open(self.images[index]).convert('RGB')
        if self.image_set == "val":
            mask = Image.open(self.masks[index])
            if self.transforms:
                img, mask = self.transforms(img, mask)
                mask = (mask * 255).int()
            return {
                'image': img,
                'mask': mask,
            }
        elif "train" in self.image_set:
            if self.transforms:
                if self.return_masks:
                    mask = Image.open(self.masks[index])
                    res = self.transforms(img, mask)
                    img = res[0]
                    mask = (res[1] * 255).int()
                    return {
                        'image': img,
                        'mask': mask,
                    }
                else:
                    res = self.transforms(img)
                    img = res
                    return {
                        'image': img
                    }
            return {
                'image': img
            }

    def __len__(self) -> int:
        return len(self.images)