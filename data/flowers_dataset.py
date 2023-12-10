import os
import sys
root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-2])
sys.path.append(root_path)

import numpy as np
import pytorch_lightning as pl
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from scipy import io

from modules.utils import rescale


class FlowersDataset(Dataset):
    def __init__(
        self, 
        data_root: str,
        data_split='train',
        use_rescale=True,
        use_flip=False,
    ):
        super(FlowersDataset, self).__init__()
        self.files =  io.loadmat(os.path.join(data_root, "setid.mat"))
        if data_split == 'train':
            self.files = self.files.get('tstid')[0]
        elif data_split == 'val':
            self.files = self.files.get('valid')[0]
        else:
            self.files = self.files.get('trnid')[0]
        self.use_flip = use_flip

        self.transform_seg = transforms.Compose([
            transforms.Resize((128, 128), interpolation=Image.NEAREST),
            transforms.ToTensor(),
        ])
        trans = [
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ]
        if use_rescale:
            trans.append(transforms.Lambda(rescale))
        self.transform = transforms.Compose(trans)
        self.datapath = data_root

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        imgname = "image_%05d.jpg" % self.files[idx]
        segname = "segmim_%05d.jpg" % self.files[idx]
        img = self.transform(Image.open(os.path.join(self.datapath, "jpg", imgname)))
        seg = np.array(Image.open(os.path.join(self.datapath, "segmim", segname)))
        seg = 1 - ((seg[:,:,0:1] == 0) + (seg[:,:,1:2] == 0) + (seg[:,:,2:3] == 254))
        seg = (seg * 255).astype('uint8').repeat(3,axis=2)
        seg = self.transform_seg(Image.fromarray(seg))[:1]

        return {
            'image': img,
            'mask': seg, 
        }  


class FlowersDataModule(pl.LightningDataModule):
    def __init__(
        self,
        args,
    ):
        super().__init__()
        self.data_root = args.data_root
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers

        self.train_dataset = FlowersDataset(args.data_root, 'train')
        self.val_dataset = FlowersDataset(args.data_root, 'val',)
        self.test_dataset = FlowersDataset(args.data_root, 'test')

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )