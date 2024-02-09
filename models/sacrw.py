import torch
import torch.nn as nn
import torch.nn.functional as F
import modules.vision_transformer as vit
import math
from modules.slot_attention import SlotAttentionEncoder
from torch.cuda.amp import autocast as autocast
from modules.utils import positionalencoding2d

class SACRW(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        dino = vit.__dict__[args.vit_arch](img_size=[224], patch_size=args.vit_patch_size, num_classes=0)
        for p in dino.parameters():
            p.requires_grad = False
        dino.eval()
        state_dict = torch.load(args.vit_model_path, map_location="cpu")
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        msg = dino.load_state_dict(state_dict, strict=False)
        self.vit_encoder = dino
        self.slot_attn = SlotAttentionEncoder(num_iterations=args.num_iterations,
                                              num_slots=args.num_slots,
                                              input_channels=args.vit_feature_size,
                                              slot_size=args.slot_size,
                                              mlp_hidden_size=args.mlp_hidden_size,
                                              num_heads=args.num_slot_heads)

    @autocast()
    def topk_filter(self, x, ratio=0.5, fill=-1e4):
        n = x.size(-1)
        a, _ = x.topk(k=int(n * ratio), dim=-1)
        a_min = torch.min(a, dim=-1).values
        a_min = a_min.unsqueeze(-1).repeat(1, 1, n)
        ge = torch.ge(x, a_min)
        neg = torch.ones_like(x) * fill
        x = torch.where(ge, x, neg)
        return x

    @autocast()
    def forward(self, image, return_slots=False):
        B, C, H, W = image.size()

        features = self.vit_encoder.forward_feats(image)[:, 1:].detach()
        H_enc = W_enc = int(math.sqrt(features.size(1)))

        f = F.normalize(features, p=2, dim=-1)  # (B, N, C)
        ff_corr = torch.matmul(f, f.permute(0, 2, 1))
        ff_corr = torch.where(ff_corr >= self.args.threshold, ff_corr, -1e4) / self.args.temperature
        ff_corr = ff_corr.softmax(dim=-1)
        if self.args.additional_position:
            pos_emb = positionalencoding2d(self.args.vit_feature_size, H_enc, W_enc).cuda().flatten(1, 2).permute(1, 0)
            p = F.normalize(pos_emb, p=2, dim=-1)
            pp_corr = torch.matmul(p, p.permute(1, 0)) / self.args.temperature
            pp_corr = pp_corr.softmax(dim=-1)
            ff_corr = 0.5 * (ff_corr + pp_corr)

        slots, attns = self.slot_attn(features) #(B, K, C)
        k = slots.size(1)
        s = F.normalize(slots, p=2, dim=-1)
        ss_corr = torch.eye(k).expand(B, k, k).cuda()

        fs_corr = torch.matmul(f, s.permute(0, 2, 1))
        sf_corr = fs_corr.permute(0, 2, 1)
        fs_corr = (fs_corr / self.args.temperature).softmax(dim=-1)
        sf_corr = (sf_corr / self.args.temperature).softmax(dim=-1)


        transition1 = torch.matmul(sf_corr, fs_corr)
        wpw_loss = (torch.log(transition1 + 1e-4).flatten(0, 1) * ss_corr.flatten(0, 1) * (-1)).mean()
        transition2 = torch.matmul(fs_corr, sf_corr)
        pwp_loss = (torch.log(transition2 + 1e-4).flatten(0, 1) * ff_corr.flatten(0, 1) * (-1)).mean()

        attns = fs_corr
        attns = attns.permute(0, 2, 1).reshape(B, -1, H_enc, W_enc)
        attns = F.interpolate(attns, size=(H, W), mode='bilinear').unsqueeze(2)

        log_dict = {'wpw_loss': self.args.alpha * wpw_loss, 'pwp_loss': self.args.beta * pwp_loss}
        if return_slots:
            return self.args.alpha * wpw_loss + self.args.beta * pwp_loss, attns, log_dict, torch.matmul(sf_corr, f)
        else:
            return self.args.alpha * wpw_loss + self.args.beta * pwp_loss, attns, log_dict