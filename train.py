import sys
sys.path.append("..")
import torch
import torch.nn.functional as F
import os
import argparse
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from torch.cuda import amp
from torch.cuda.amp import autocast as autocast
from modules.utils import cosine_anneal, lr_scheduler_warm, lr_scheduler_no_warm, average_ari, iou_and_dice, average_segcover, miou, clustering, get_parameter_number

from data.birds_dataset import BirdsDataModule
from data.dogs_dataset import DogsDataModule
from data.cars_dataset import CarsDataModule
from data.flowers_dataset import FlowersDataModule
from data.voc_dataset import VOCDataModule

from models.sacrw import SACRW
import warnings
warnings.filterwarnings("ignore")

data_paths = {
    'birds': '/home/ziyu/Datasets/Birds',
    'dogs': '/home/ziyu/Datasets/Dogs',
    'cars': '/home/ziyu/Datasets/Cars',
    'flowers': '/home/ziyu/Datasets/Flowers',
    'voc': '/home/ziyu/Datasets/PascalVOC',
}

dataset_factory = {
    'birds': BirdsDataModule,
    'dogs': DogsDataModule,
    'cars': CarsDataModule,
    'flowers': FlowersDataModule,
    'voc': VOCDataModule,
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    scaler = amp.GradScaler()

    datamodule = dataset_factory[args.dataset](args)
    model = model_factory[args.model_name](args).cuda()
    print(get_parameter_number(model))
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)
        model.to(device)

    if args.model_name == 'sacrw':
        optimizer = Adam(model.parameters(), lr=args.lr)
    else:
        raise NotImplementedError

    global_step = 0
    best = 0.0
    best_epoch = 0
    for epoch in range(args.max_epochs):
        print("==================== Training ====================")
        if global_step >= args.max_steps:
            break
        model.train()
        for batch, data in enumerate(datamodule.train_dataloader()):
            image = data['image'].to(device)
            if args.model_name == 'sacrw':
                factor_main = lr_scheduler_warm(global_step, args.warmup_steps, args.decay_steps)
                optimizer.param_groups[0]['lr'] = factor_main * args.lr
                optimizer.zero_grad()
                with autocast():
                    loss, masks, log_dict = model(image)
            else:
                raise NotImplementedError

            loss = loss.mean()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), args.grad_clip, 'inf')
            scaler.step(optimizer)
            scaler.update()
            global_step = global_step + 1

            if (batch + 1) % args.log_interval == 0:
                with torch.no_grad():
                    str = 'Train Epoch: {:3} [{:5}]'.format(epoch, batch + 1)
                    for k, v in log_dict.items():
                        str = str + "\t{}:{:F}".format(k, v.mean().item())
                    print(str)

        if (epoch + 1) % args.evaluate_interval == 0:
            print("==================== Evaluation ====================")
            with torch.no_grad():
                model.eval()
                val_step = 0
                val_results_list = {'ari': [], 'ari_fg': [], 'msc_fg': [], 'iou': [], 'dice': []}
                for batch, data in enumerate(datamodule.val_dataloader()):
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
                        val_results_list['ari'].append(ari)
                        val_results_list['ari_fg'].append(ari_fg)
                        val_results_list['msc_fg'].append(msc_fg)
                    elif args.task == 'fe':
                        m = F.one_hot(masks.argmax(dim=1), args.num_slots).permute(0, 4, 1, 2, 3)
                        iou, dice = iou_and_dice(m[:, 0], masks_gt)
                        for i in range(1, args.num_slots):
                            iou1, dice1 = iou_and_dice(m[:, i], masks_gt)
                            iou = torch.max(iou, iou1)
                            dice = torch.max(dice, dice1)
                        iou = iou.mean()
                        dice = dice.mean()
                        val_results_list['iou'].append(iou)
                        val_results_list['dice'].append(dice)
                    val_step = val_step + 1
                    if (batch + 1) % args.log_interval == 0:
                        print('Evaluate Epoch: {:3} [{:5}]'.format(epoch, batch + 1))
                val_results = {}
                if (args.task == 'od') or (args.task == 'fe'):
                    for k, v in val_results_list.items():
                        if len(v) > 0:
                            val_results['avg_' + k] = torch.stack(v).mean()

                log_dir = os.path.join(args.log_dir, args.model_name, args.dataset)
                if not os.path.exists(log_dir):
                    os.makedirs(log_dir)
                with open(os.path.join(log_dir, "log.txt"), 'a') as f:
                    str = '====> Epoch: {:3}'.format(epoch)
                    for k, v in val_results.items():
                        str = str + "\t{}:{:F}".format(k, v.item())
                    f.write(str + '\n')
                if best < val_results[args.monitor]:
                    best = val_results[args.monitor]
                    best_epoch = epoch
                    torch.save(model.state_dict(), os.path.join(log_dir, 'best_model.pt'))
                print('====> Best {}: {:F} @ Epoch {}'.format(args.monitor, best, best_epoch))

def main():
    args = parse_args()
    args.data_root = data_paths[args.dataset]
    run(args)

if __name__ == '__main__':
    main()
