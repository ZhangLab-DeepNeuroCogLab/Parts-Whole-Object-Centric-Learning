import sys
sys.path.append("..")
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import os
import argparse
from torch.cuda import amp
from torch.cuda.amp import autocast as autocast
from modules.utils import average_ari, iou_and_dice, average_segcover

from data.birds_dataset import BirdsDataModule
from data.dogs_dataset import DogsDataModule
from data.cars_dataset import CarsDataModule
from data.flowers_dataset import FlowersDataModule
from data.voc_dataset import VOCDataModule
from data.coco_dataset import CocoDataModule
from data.movi_dataset import MOVIDataModule
from models.sacrw import SACRW
import warnings
warnings.filterwarnings("ignore")

data_paths = {
    'birds': '/home/ziyu/Datasets/Birds',
    'dogs': '/home/ziyu/Datasets/Dogs',
    'cars': '/home/ziyu/Datasets/Cars',
    'flowers': '/home/ziyu/Datasets/Flowers',
    'voc': '/home/ziyu/Datasets/PascalVOC',
    'coco': '/home/ziyu/Datasets/mscoco',
    'movi': '/home/ziyu/Datasets/MOVI',
}

dataset_factory = {
    'birds': BirdsDataModule,
    'dogs': DogsDataModule,
    'cars': CarsDataModule,
    'flowers': FlowersDataModule,
    'voc': VOCDataModule,
    'coco': CocoDataModule,
    'movi': MOVIDataModule,
}

model_factory = {
    "sacrw": SACRW,
}

def parse_args():
    parser = argparse.ArgumentParser('Unsupervised object-centric learning for image.')

    # Training config
    parser.add_argument('--dataset', default="voc", help="birds | dogs | cars | flowers | voc")
    parser.add_argument('--log_dir', default='./logs/', help="path where to save, empty for no saving.")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--max_steps', type=int, default=250000)
    parser.add_argument('--max_epochs', type=int, default=50000)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--lr', type=float, default=4e-4)
    parser.add_argument('--warmup_steps', type=int, default=5000)
    parser.add_argument('--decay_steps', type=int, default=50000)
    parser.add_argument('--model_name', default='sacrw', help="sacrw")

    # Evaluation
    parser.add_argument('--task', type=str, default='od', help='od (object discovery) | fe (foreground extraction) | ss (semantic segmentation)')
    parser.add_argument('--evaluate_interval', type=int, default=1)
    parser.add_argument('--monitor', type=str, default='avg_ari_fg', help='avg_ari_fg or avg_iou')

    # Backbone ViT
    parser.add_argument('--vit_arch', type=str, default='vit_small')
    parser.add_argument('--vit_model_path', type=str, default='./pretrained_models/dino_deitsmall8_pretrain.pth')
    parser.add_argument('--vit_patch_size', type=int, default=8)
    parser.add_argument('--vit_feature_size', type=int, default=384)

    # Slot attention
    parser.add_argument('--num_iterations', type=int, default=3)
    parser.add_argument('--slot_size', type=int, default=384)
    parser.add_argument('--num_slots', type=int, default=2, help="Number of slots")
    parser.add_argument('--num_slot_heads', type=int, default=1)
    parser.add_argument('--mlp_hidden_size', type=int, default=384)

    # Contrastive random work
    parser.add_argument('--temperature', type=float, default=0.1)
    parser.add_argument('--threshold', type=float, default=0.7)
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--beta', type=float, default=10)

    args = parser.parse_args()
    return args

def run(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    datamodule = dataset_factory[args.dataset](args)
    model = model_factory[args.model_name](args).cuda()
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)
        model.to(device)

    log_dir = os.path.join(args.log_dir, args.model_name, args.dataset)
    model.load_state_dict(torch.load(os.path.join(log_dir, 'best_model.pt')))
    model.eval()

    print("==================== Testing ====================")
    test_results_list = {'ari': [], 'ari_fg': [], 'msc_fg': [], 'iou': [], 'dice': []}
    for batch, data in enumerate(datamodule.test_dataloader()):
        image = data['image'].to(device)
        masks_gt = data['mask'].to(device)
        if args.model_name == 'sacrw':
            with autocast():
                loss, masks, log_dict = model(image)
        else:
            raise NotImplementedError
        if args.task == 'od':
            m = masks.detach().argmax(dim=1)
            ari, _ = average_ari(m, masks_gt)
            ari_fg, _ = average_ari(m, masks_gt, True)
            msc_fg, _ = average_segcover(masks_gt, m, True)
            test_results_list['ari'].append(ari)
            test_results_list['ari_fg'].append(ari_fg)
            test_results_list['msc_fg'].append(msc_fg)
        elif args.task == 'fe':
            m = F.one_hot(masks.argmax(dim=1), args.num_slots).permute(0, 4, 1, 2, 3)
            iou, dice = iou_and_dice(m[:, 0], masks_gt)
            for i in range(1, args.num_slots):
                iou1, dice1 = iou_and_dice(m[:, i], masks_gt)
                iou = torch.max(iou, iou1)
                dice = torch.max(dice, dice1)
            iou = iou.mean()
            dice = dice.mean()
            test_results_list['iou'].append(iou)
            test_results_list['dice'].append(dice)

        if (batch + 1) % args.log_interval == 0:
            print('Test Batch: [{:5}]'.format(batch + 1))

    test_results = {}
    for k, v in test_results_list.items():
        if len(v) > 0:
            test_results['avg_' + k] = torch.stack(v).mean()
    str = ""
    for k, v in test_results.items():
        str = str + "\t{}:{:F}".format(k, v.item())
    print(str)

def main():
    args = parse_args()
    args.data_root = data_paths[args.dataset]
    run(args)

if __name__ == '__main__':
    main()
