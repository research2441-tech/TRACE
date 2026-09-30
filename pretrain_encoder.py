"""Pretraining of the mammography encoder on a DISJOINT pretraining cohort (8,000 women, vendors
V1-V3, current round only).  Objective: report-level weak supervision (image-level finding category
none / benign-appearing / suspicious, and density category) through multiple-instance pooling over
the 12x12 feature grid, plus an InfoNCE term between acquisition-style views of the same image.
No development, calibration, test or external exam is used here."""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, time, json
from common import Encoder, acq_augment, load_meta, DATA

torch.manual_seed(0)
torch.set_num_threads(2)
img = np.load(f"{DATA}/pre_img.npy", mmap_mode="r")
meta = load_meta("pre")
pool, lab, dens = [], [], []
for i, m in enumerate(meta):
    for j in range(4):
        L = m["lab"]["LR"[j // 2]]
        pool.append((i, j)); lab.append(2 if L["cancer"] else (1 if L["stable_benign"] else 0)); dens.append(m["density"])
lab, dens = np.array(lab), np.array(dens)
print("pretraining views", len(pool), np.bincount(lab))
enc = Encoder(64)
cell = nn.Conv2d(64, 3, 1)                       # per-cell finding logits
dhead = nn.Linear(64, 4)
proj = nn.Sequential(nn.Linear(64, 128), nn.GELU(), nn.Linear(128, 64))
params = list(enc.parameters()) + list(cell.parameters()) + list(dhead.parameters()) + list(proj.parameters())
ITERS, B, TAU = 1500, 96, 0.2
opt = torch.optim.AdamW(params, 2e-3, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=ITERS, pct_start=0.1)
g = torch.Generator().manual_seed(1)
rng = np.random.default_rng(0)


def geo(x):
    Bn = x.shape[0]
    th = torch.zeros(Bn, 2, 3)
    s = torch.empty(Bn).uniform_(0.92, 1.08, generator=g)
    th[:, 0, 0] = s; th[:, 1, 1] = s
    th[:, :, 2] = torch.empty(Bn, 2).uniform_(-0.06, 0.06, generator=g)
    return F.grid_sample(x, F.affine_grid(th, x.shape, align_corners=False), align_corners=False)


def mil(logits):  # [B,3,12,12] -> image-level logits via log-sum-exp pooling (smooth max)
    return torch.logsumexp(logits.flatten(2) * 2.0, -1) / 2.0 - np.log(144) / 2.0


log = []
t0 = time.time()
for it in range(ITERS):
    idx = rng.choice(len(pool), B, replace=False)
    x = torch.from_numpy(np.stack([img[pool[i][0], 0, pool[i][1]] for i in idx])).float().div(255).unsqueeze(1)
    v1, v2 = acq_augment(geo(x), g), acq_augment(geo(x), g)
    f = enc(torch.cat([v1, v2]))
    yl = torch.from_numpy(lab[idx]).repeat(2)
    yd = torch.from_numpy(dens[idx]).repeat(2)
    l_find = F.cross_entropy(mil(cell(f)), yl)
    l_dens = F.cross_entropy(dhead(f.mean((2, 3))), yd)
    z = F.normalize(proj(f.mean((2, 3))), dim=1)
    sim = z @ z.t() / TAU
    sim.fill_diagonal_(-1e9)
    l_con = F.cross_entropy(sim, torch.cat([torch.arange(B, 2 * B), torch.arange(0, B)]))
    loss = l_find + 0.5 * l_dens + 0.3 * l_con
    opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    if it % 50 == 0:
        acc = (mil(cell(f)).argmax(1) == yl).float().mean()
        log.append(dict(it=it, find=float(l_find.detach()), dens=float(l_dens.detach()), con=float(l_con.detach()),
                        acc=float(acc), t=time.time() - t0)); print(log[-1], flush=True)
torch.save(enc.state_dict(), "../res/encoder_ssl.pt")
json.dump(log, open("../res/pretrain_log.json", "w"))
