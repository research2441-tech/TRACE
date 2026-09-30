"""Shared definitions: data splits, encoder backbone and acquisition-style augmentation."""
import numpy as np, json, torch, torch.nn as nn, torch.nn.functional as F

DATA = "../data"
RES = "../res"


def splits():
    """Patient-level partition of the development cohort (one index exam per woman)."""
    p = np.random.default_rng(2026).permutation(6000)
    return dict(train=p[:3600], val=p[3600:4200], cal=p[4200:5000], test=p[5000:])


def load_meta(c):
    return json.load(open(f"{DATA}/{c}_meta.json"))


def conv_block(ci, co, s):
    return nn.Sequential(nn.Conv2d(ci, co, 3, s, 1, bias=False), nn.GroupNorm(8, co), nn.GELU(),
                         nn.Conv2d(co, co, 3, 1, 1, bias=False), nn.GroupNorm(8, co), nn.GELU())


class Encoder(nn.Module):
    """Compact convolutional backbone: 96x96 view -> 12x12 grid of C-dim features."""

    def __init__(self, C=64):
        super().__init__()
        self.net = nn.Sequential(conv_block(1, 24, 2), conv_block(24, 48, 2), conv_block(48, C, 2), conv_block(C, C, 1))
        self.C = C

    def forward(self, x):
        return self.net(x)


def acq_augment(x, g):
    """Acquisition-style photometric perturbation on [0,1] images (B,1,H,W)."""
    B = x.shape[0]
    gam = torch.exp(torch.empty(B, 1, 1, 1).uniform_(-0.45, 0.45, generator=g))
    slope = torch.empty(B, 1, 1, 1).uniform_(2.5, 9.0, generator=g)
    cen = torch.empty(B, 1, 1, 1).uniform_(0.35, 0.6, generator=g)
    m = (x > 0.02).float()
    y = x.clamp(0, 1) ** gam
    y = torch.sigmoid(slope * (y - cen))
    lo = (y * m + (1 - m) * 9).flatten(1).min(1)[0].view(B, 1, 1, 1)
    hi = (y * m - (1 - m) * 9).flatten(1).max(1)[0].view(B, 1, 1, 1)
    y = (y - lo) / (hi - lo + 1e-6)
    k = torch.empty(B, 1, 1, 1).uniform_(0, 1.0, generator=g)
    bl = F.avg_pool2d(y, 5, 1, 2)
    y = y + k * (y - bl)
    y = y + torch.randn(y.shape, generator=g) * torch.empty(B, 1, 1, 1).uniform_(0, .04, generator=g)
    return (y * m).clamp(0, 1.2)
