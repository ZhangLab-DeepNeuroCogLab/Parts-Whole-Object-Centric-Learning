from typing import TypeVar
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import adjusted_rand_score
import math
from joblib import Parallel
from joblib.parallel import delayed
from scipy.optimize import linear_sum_assignment
from kmeans_pytorch import kmeans

Tensor = TypeVar("Tensor")
T = TypeVar("T")
TK = TypeVar("TK")
TV = TypeVar("TV")

def to_rgb_from_tensor(x: Tensor):
    return (x * 0.5 + 0.5).clamp(0, 1)

def build_grid(resolution):
    ranges = [torch.linspace(0.0, 1.0, steps=res) for res in resolution]
    grid = torch.meshgrid(*ranges, indexing='ij')
    grid = torch.stack(grid, dim=-1)
    grid = torch.reshape(grid, [resolution[0], resolution[1], -1])
    grid = grid.unsqueeze(0)
    return torch.cat([grid, 1.0 - grid], dim=-1)

def build_grid_3d(resolution):
    x, y, z = resolution
    x = torch.linspace(0.0, 1.0, steps=x)
    y = torch.linspace(0.0, 1.0, steps=y)
    z = torch.linspace(0.0, 1.0, steps=z)
    lenx, leny, lenz = x.size(0), y.size(0), z.size(0)
    x = x.unsqueeze(1).unsqueeze(2).repeat(1, leny, lenz)
    y = y.unsqueeze(0).unsqueeze(2).repeat(lenx, 1, lenz)
    z = z.unsqueeze(0).unsqueeze(1).repeat(lenx, leny, 1)
    grid = torch.stack([x, y, z], dim=-1)
    grid = grid.unsqueeze(0)
    return torch.cat([grid, 1.0 - grid], dim=-1)

class LinearPositionEncoding(nn.Module):
    def __init__(self, hidden_size: int, resolution):
        super().__init__()
        self.dense = nn.Linear(4, out_features=hidden_size)
        self.register_buffer("grid", build_grid(resolution))

    def forward(self, inputs):
        emb_proj = self.dense(self.grid).permute(0, 3, 1, 2)

        return inputs + emb_proj

class LinearPositionEncoding3D(nn.Module):
    def __init__(self, hidden_size: int, resolution):
        super().__init__()
        self.resolution = resolution
        self.dense = nn.Linear(6, out_features=hidden_size)
        #self.register_buffer("grid", build_grid_3d(resolution))

    def forward(self, inputs):
        grid = build_grid_3d(self.resolution).type_as(inputs)
        emb_proj = self.dense(grid).permute(0, 4, 1, 2, 3)

        return inputs + emb_proj

class ConvTransposeBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride):
        super().__init__()
        output_padding = 0 if stride == 1 else 1
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=kernel_size,
                               stride=stride, padding=kernel_size // 2, output_padding=output_padding),
            nn.ReLU(),
        )

    def forward(self, x):
        out = self.conv(x)
        return out

def gumbel_max(logits, dim=-1):
    eps = torch.finfo(logits.dtype).tiny
    gumbels = -(torch.empty_like(logits).exponential_() + eps).log()
    gumbels = logits + gumbels
    return gumbels.argmax(dim)


def gumbel_softmax(logits, tau=1., hard=False, dim=-1):
    eps = torch.finfo(logits.dtype).tiny
    gumbels = -(torch.empty_like(logits).exponential_() + eps).log()
    gumbels = (logits + gumbels) / tau
    y_soft = F.softmax(gumbels, dim)
    if hard:
        index = y_soft.argmax(dim, keepdim=True)
        y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.)
        return y_hard - y_soft.detach() + y_soft
    else:
        return y_soft


def log_prob_gaussian(value, mean, std):
    var = std ** 2
    if isinstance(var, float):
        return -0.5 * (((value - mean) ** 2) / var + math.log(var) + math.log(2 * math.pi))
    else:
        return -0.5 * (((value - mean) ** 2) / var + var.log() + math.log(2 * math.pi))


def conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0,
           dilation=1, groups=1, bias=True, padding_mode='zeros',
           weight_init='xavier'):
    m = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
    if weight_init == 'kaiming':
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
    else:
        nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.zeros_(m.bias)
    return m


class Conv2dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()

        self.m = conv2d(in_channels, out_channels, kernel_size, stride, padding,
                        bias=False, weight_init='kaiming')
        self.weight = nn.Parameter(torch.ones(out_channels))
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x):
        x = self.m(x)
        return F.relu(F.group_norm(x, 1, self.weight, self.bias))


def linear(in_features, out_features, bias=True, weight_init='xavier', gain=1.):
    m = nn.Linear(in_features, out_features, bias)
    if weight_init == 'kaiming':
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
    else:
        nn.init.xavier_uniform_(m.weight, gain)

    if bias:
        nn.init.zeros_(m.bias)
    return m


def gru_cell(input_size, hidden_size, bias=True):
    m = nn.GRUCell(input_size, hidden_size, bias)
    nn.init.xavier_uniform_(m.weight_ih)
    nn.init.orthogonal_(m.weight_hh)
    if bias:
        nn.init.zeros_(m.bias_ih)
        nn.init.zeros_(m.bias_hh)
    return m


class OneHotDictionary(nn.Module):
    def __init__(self, vocab_size, emb_size):
        super().__init__()
        self.dictionary = nn.Embedding(vocab_size, emb_size)

    def forward(self, x):
        """
        x: B, N, vocab_size
        """

        tokens = torch.argmax(x, dim=-1)  # batch_size x N
        token_embs = self.dictionary(tokens)  # batch_size x N x emb_size
        return token_embs

    def get_embedding(self):
        return self.dictionary.weight


def cosine_anneal(step, start_value, final_value, start_step, final_step):
    assert start_value >= final_value
    assert start_step <= final_step

    if step < start_step:
        value = start_value
    elif step >= final_step:
        value = final_value
    else:
        a = 0.5 * (start_value - final_value)
        b = 0.5 * (start_value + final_value)
        progress = (step - start_step) / (final_step - start_step)
        value = a * math.cos(math.pi * progress) + b

    return value

def linear_warmup(step, start_value, final_value, start_step, final_step):

    assert start_value <= final_value
    assert start_step <= final_step

    if step < start_step:
        value = start_value
    elif step >= final_step:
        value = final_value
    else:
        a = final_value - start_value
        b = start_value
        progress = (step + 1 - start_step) / (final_step - start_step)
        value = a * progress + b

    return value

def lr_scheduler_warm(step, warmup_steps, decay_steps):
    if step < warmup_steps:
        factor = step / warmup_steps
    else:
        factor = 1
    factor *= 0.5 ** (step / decay_steps)
    return factor


def lr_scheduler_no_warm(step, decay_steps):
    factor = 0.5 ** (step / decay_steps)
    return factor


def iou_and_dice(mask, mask_gt):
    B, _, H, W = mask.shape

    pred = mask > 0.5
    gt = mask_gt > 0.5

    eps = 1e-8  # to prevent NaN
    insertion = (pred * gt).view(B, -1).sum(dim=-1)
    union = ((pred + gt) > 0).view(B, -1).sum(dim=-1) + eps
    pred_plus_gt = pred.view(B, -1).sum(dim=-1) + gt.view(B, -1).sum(dim=-1) + eps

    iou = (insertion + eps) / union
    dice = (2 * insertion + eps) / pred_plus_gt

    return iou, dice


def iou(mask, mask_gt):
    eps = 1e-8  # to prevent NaN
    insertion = (mask * mask_gt).view(-1).sum()
    union = ((mask + mask_gt) > 0).view(-1).sum() + eps
    iou = (insertion + eps) / union
    return iou


def iou_binary(mask_A, mask_B, debug=False):
    if debug:
        assert mask_A.shape == mask_B.shape
        assert mask_A.dtype == torch.bool
        assert mask_B.dtype == torch.bool
    intersection = (mask_A * mask_B).sum((1, 2, 3))
    union = (mask_A + mask_B).sum((1, 2, 3))
    # Return -100 if union is zero, else return IOU
    return torch.where(union == 0, torch.tensor(-100.0).to(mask_A.device),
                       intersection.float() / union.float())


def average_ari(masks, masks_gt, foreground_only=False):
    ari = []
    assert masks.shape[0] == masks_gt.shape[0], f"The number of masks is not equal to the number of masks_gt"
    # Loop over elements in batch
    for i in range(masks.shape[0]):
        m = masks[i].cpu().numpy().flatten()
        m_gt = masks_gt[i].cpu().numpy().flatten()
        if foreground_only:
            m = m[np.where(m_gt > 0)]
            m_gt = m_gt[np.where(m_gt > 0)]
        score = adjusted_rand_score(m, m_gt)
        ari.append(score)
    return torch.Tensor(ari).mean(), ari


def average_segcover(segA, segB, ignore_background=False):
    """
    Covering of segA by segB
    segA.shape = [batch size, 1, img_dim1, img_dim2]  ground truth
    segB.shape = [batch size, 1, img_dim1, img_dim2]
    scale: If true, take weighted mean over IOU values proportional to the
           the number of pixels of the mask being covered.
    Assumes labels in segA and segB are non-negative integers.
    Negative labels will be ignored.
    """

    assert segA.shape == segB.shape, f"{segA.shape} - {segB.shape}"
    assert segA.shape[1] == 1 and segB.shape[1] == 1
    bsz = segA.shape[0]
    nonignore = (segA >= 0)

    mean_scores = torch.tensor(bsz * [0.0]).to(segA.device)
    N = torch.tensor(bsz * [0]).to(segA.device)
    scaled_scores = torch.tensor(bsz * [0.0]).to(segA.device)
    scaling_sum = torch.tensor(bsz * [0]).to(segA.device)

    # Find unique label indices to iterate over
    if ignore_background:
        iter_segA = torch.unique(segA[segA > 0]).tolist()
    else:
        iter_segA = torch.unique(segA[segA >= 0]).tolist()
    iter_segB = torch.unique(segB[segB >= 0]).tolist()
    # Loop over segA
    for i in iter_segA:
        binaryA = segA == i
        if not binaryA.any():
            continue
        max_iou = torch.tensor(bsz * [0.0]).to(segA.device)
        # Loop over segB to find max IOU
        for j in iter_segB:
            # Do not penalise pixels that are in ignore regions
            binaryB = (segB == j) * nonignore
            if not binaryB.any():
                continue
            iou = iou_binary(binaryA, binaryB)
            max_iou = torch.where(iou > max_iou, iou, max_iou)
        # Accumulate scores
        mean_scores += max_iou
        N = torch.where(binaryA.sum((1, 2, 3)) > 0, N + 1, N)
        scaled_scores += binaryA.sum((1, 2, 3)).float() * max_iou
        scaling_sum += binaryA.sum((1, 2, 3))

    # Compute coverage
    mean_sc = mean_scores / torch.max(N, torch.tensor(1)).float()
    scaled_sc = scaled_scores / torch.max(scaling_sum, torch.tensor(1)).float()

    # Return mean over batch dimension
    return mean_sc.mean(0), scaled_sc.mean(0)

def rescale(x: Tensor) -> Tensor:
    return x * 2 - 1

def miou(masks, masks_gt, num_clusters, num_classes):
    # masks (B, 1, H, W)
    # masks_gt (B, 1, H, W)
    B, _, H, W = masks.size()
    tp = [0] * num_classes
    fp = [0] * num_classes
    fn = [0] * num_classes
    jac = [0] * num_classes
    all_preds = np.zeros(B * H * W, dtype=np.float32)
    all_gt = np.zeros(B * H * W, dtype=np.float32)
    offset_ = 0
    for i in range(B):
        mask = masks[i].squeeze(0).cpu().numpy()
        target = masks_gt[i].squeeze(0).cpu().numpy()
        valid = (target != 255)
        n_valid = np.sum(valid)
        all_gt[offset_:offset_ + n_valid] = target[valid]
        all_preds[offset_:offset_ + n_valid, ] = mask[valid]
        all_gt[offset_:offset_ + n_valid, ] = target[valid]
        offset_ += n_valid
    all_preds = all_preds[:offset_, ]
    all_gt = all_gt[:offset_, ]
    num_elems = offset_
    if num_clusters == num_classes:
        print('Using hungarian algorithm for matching')
        match = hungarian_match(all_preds, all_gt, preds_k=num_clusters, targets_k=num_classes, metric='iou')
    else:
        print('Using majority voting for matching')
        match = majority_vote(all_preds, all_gt, preds_k=num_clusters, targets_k=num_classes)

    reordered_preds = np.zeros(num_elems, dtype=all_preds.dtype)
    for pred_i, target_i in match:
        reordered_preds[all_preds == int(pred_i)] = int(target_i)
    for i_part in range(0, num_classes):
        tmp_all_gt = (all_gt == i_part)
        tmp_pred = (reordered_preds == i_part)
        tp[i_part] += np.sum(tmp_all_gt & tmp_pred)
        fp[i_part] += np.sum(~tmp_all_gt & tmp_pred)
        fn[i_part] += np.sum(tmp_all_gt & ~tmp_pred)
    for i_part in range(0, num_classes):
        jac[i_part] = float(tp[i_part]) / max(float(tp[i_part] + fp[i_part] + fn[i_part]), 1e-8)
    mean_iou = np.mean(jac)
    return mean_iou

def hungarian_match(flat_preds, flat_targets, preds_k, targets_k, metric='acc', n_jobs=16):
    assert (preds_k == targets_k)  # one to one
    num_k = preds_k

    # perform hungarian matching
    print('Using iou as metric')
    results = Parallel(n_jobs=n_jobs, backend='multiprocessing')(delayed(get_iou)(
        flat_preds, flat_targets, c1, c2) for c2 in range(num_k) for c1 in range(num_k))
    results = np.array(results)
    results = results.reshape((num_k, num_k)).T
    match = linear_sum_assignment(flat_targets.shape[0] - results)
    match = np.array(list(zip(*match)))
    res = []
    for out_c, gt_c in match:
        res.append((out_c, gt_c))

    return res


def majority_vote(flat_preds, flat_targets, preds_k, targets_k, n_jobs=16):
    iou_mat = Parallel(n_jobs=n_jobs, backend='multiprocessing')(delayed(get_iou)(
        flat_preds, flat_targets, c1, c2) for c2 in range(targets_k) for c1 in range(preds_k))
    iou_mat = np.array(iou_mat)
    results = iou_mat.reshape((targets_k, preds_k)).T
    results = np.argmax(results, axis=1)
    match = np.array(list(zip(range(preds_k), results)))
    return match


def get_iou(flat_preds, flat_targets, c1, c2):
    tp = 0
    fn = 0
    fp = 0
    tmp_all_gt = (flat_preds == c1)
    tmp_pred = (flat_targets == c2)
    tp += np.sum(tmp_all_gt & tmp_pred)
    fp += np.sum(~tmp_all_gt & tmp_pred)
    fn += np.sum(tmp_all_gt & ~tmp_pred)
    jac = float(tp) / max(float(tp + fp + fn), 1e-8)
    return jac

def clustering(masks, region_vectors, num_clusters):
    # masks(B, 1, H, W)
    # region_vectors (B, num_slots, c)
    masks = masks.squeeze(1)
    B, num_slots, c = region_vectors.size()
    vectors = region_vectors.reshape(-1, c)
    cluster_ids_x, cluster_centers = kmeans(X=vectors, num_clusters=num_clusters, distance='cosine', device=vectors.get_device())
    cluster_ids_x = cluster_ids_x.reshape(B, num_slots).cuda()
    masks_kmeans = torch.zeros_like(masks)
    for i in range(B):
        for j in range(num_slots):
            masks_kmeans[i][masks[i] == j] = cluster_ids_x[i][j]
    return masks_kmeans.unsqueeze(1)

def color_map(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap

def positionalencoding2d(d_model, height, width):
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

    return pe

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num * 1e-3, 'Trainable': trainable_num * 1e-3}