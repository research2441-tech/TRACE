"""Proposed longitudinal change-aware model and the compared architectures.

Tensor conventions
  fc : [B, 4, C, 12, 12]   current views (LCC, LMLO, RCC, RMLO), frozen-encoder features
  fp : [B, 2, 4, C, 12, 12] prior rounds t-1, t-2
  av : [B, 2]              prior availability (bool)
Every model returns a dict with breast-level logits mal [B,2], cat [B,2,4], risk [B,2],
localisation logits heat [B,4,144] and breast embeddings emb [B,2,D].
"""
import math, torch, torch.nn as nn, torch.nn.functional as F

G = 12
SIDE = torch.tensor([0, 0, 1, 1])
VTYPE = torch.tensor([0, 1, 0, 1])
DISP = torch.tensor([(dx, dy) for dy in range(-2, 3) for dx in range(-2, 3)], dtype=torch.float32)  # 25 x 2


def base_grid(n, device):
    ys, xs = torch.meshgrid(torch.linspace(-1, 1, G, device=device), torch.linspace(-1, 1, G, device=device), indexing="ij")
    return torch.stack([xs, ys], -1).unsqueeze(0).expand(n, G, G, 2)


def warp(x, flow):
    """Sample x at p + flow(p); flow in grid cells [N,2,G,G]."""
    g = base_grid(x.shape[0], x.device) + flow.permute(0, 2, 3, 1) * (2.0 / (G - 1))
    return F.grid_sample(x, g, align_corners=True, padding_mode="border")


def gauss_smooth(x, sigma):
    if sigma <= 0:
        return x
    r = int(math.ceil(2.5 * sigma))
    t = torch.arange(-r, r + 1, dtype=x.dtype, device=x.device)
    k = torch.exp(-t ** 2 / (2 * sigma ** 2)); k = k / k.sum()
    C = x.shape[1]
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="replicate"), k.view(1, 1, 1, -1).repeat(C, 1, 1, 1), groups=C)
    return F.conv2d(F.pad(x, (0, 0, r, r), mode="replicate"), k.view(1, 1, -1, 1).repeat(C, 1, 1, 1), groups=C)


def _corr_index():
    idx, valid = torch.zeros(G * G, 25, dtype=torch.long), torch.zeros(G * G, 25, dtype=torch.bool)
    for p in range(G * G):
        y, x = divmod(p, G)
        for k, (dy, dx) in enumerate([(dy, dx) for dy in range(-2, 3) for dx in range(-2, 3)]):
            yy, xx = y + dy, x + dx
            if 0 <= yy < G and 0 <= xx < G:
                idx[p, k], valid[p, k] = yy * G + xx, True
    return idx, valid


CIDX, CVALID = _corr_index()


def correlate(a, b):
    """Local correlation volume over a 5x5 displacement window. a,b: [N,Dk,G,G] -> [N,25,G,G].
    Computed from the full affinity matrix and gathered at the local displacements (zero outside the grid)."""
    N, Dk = a.shape[:2]
    A = torch.bmm(a.flatten(2).transpose(1, 2), b.flatten(2))              # [N,144,144]
    c = A.gather(2, CIDX.unsqueeze(0).expand(N, -1, -1)) * CVALID           # [N,144,25]
    return c.transpose(1, 2).reshape(N, 25, G, G) / math.sqrt(Dk)


class Adapter(nn.Module):
    def __init__(self, C, D, style_norm=False):
        super().__init__()
        self.style_norm = style_norm
        self.inp = nn.Conv2d(C, D, 1)
        self.res = nn.Sequential(nn.GroupNorm(8, D), nn.Conv2d(D, D, 3, 1, 1, groups=D), nn.GELU(), nn.Conv2d(D, D, 1))

    def forward(self, x):
        if self.style_norm:   # remove per-view channel statistics (acquisition "style")
            x = (x - x.mean((2, 3), keepdim=True)) / (x.std((2, 3), keepdim=True) + 1e-3)
        x = self.inp(x)
        return x + self.res(x)


class Correspondence(nn.Module):
    """Context-regularised, cycle-checked deformable correspondence between a current and a prior view.
    Matching uses the L2-normalised FROZEN encoder features (stable, task-independent descriptors);
    the displacement is then applied to the trainable adapter features."""

    def __init__(self, D, sigma_s=1.5, align=True, gate=True, tau0=0.03):
        super().__init__()
        self.log_tau = nn.Parameter(torch.tensor(math.log(tau0)))
        self.refine = nn.Sequential(nn.Conv2d(25, 16, 1), nn.GELU(), nn.Conv2d(16, 2, 3, 1, 1))
        nn.init.zeros_(self.refine[2].weight); nn.init.zeros_(self.refine[2].bias)
        self.log_sc = nn.Parameter(torch.tensor(0.0))
        self.sigma_s, self.align, self.gate = sigma_s, align, gate

    def flow(self, ea, eb):
        cost = correlate(F.normalize(ea, dim=1), F.normalize(eb, dim=1)) * math.sqrt(ea.shape[1])   # cosine
        w = torch.softmax(cost / self.log_tau.exp(), 1)
        f = torch.einsum("nkhw,kc->nchw", w, DISP.to(ea.device))
        f = f + 0.5 * torch.tanh(self.refine(cost))
        ent = -(w * torch.log(w + 1e-8)).sum(1, keepdim=True) / math.log(25)
        return gauss_smooth(f, self.sigma_s), ent

    def forward(self, a, b, ea, eb):
        if not self.align:
            z = torch.zeros(a.shape[0], 2, G, G, device=a.device)
            bw, fab = b, z
            c = torch.ones(a.shape[0], 1, G, G, device=a.device)
            cyc = torch.zeros_like(c)
        else:
            fab, ent_ab = self.flow(ea, eb)
            fba, _ = self.flow(eb, ea)
            bw = warp(b, fab)
            cyc = (fab + warp(fba, fab)).pow(2).sum(1, keepdim=True)          # forward-backward error
            c = torch.exp(-cyc / self.log_sc.exp()) * gauss_smooth(1 - ent_ab, 1.0)
        if not self.gate:
            c = torch.ones_like(c)
        R = a - bw
        return dict(R=R, c=c, D=c * R, flow=fab, cyc=cyc, bw=bw)


class TokenEmb(nn.Module):
    def __init__(self, D, n_time=3):
        super().__init__()
        self.pos6 = nn.Parameter(torch.randn(36, D) * .02)
        self.pos12 = nn.Parameter(torch.randn(144, D) * .02)
        self.pos3 = nn.Parameter(torch.randn(9, D) * .02)
        self.view = nn.Parameter(torch.randn(2, D) * .02)
        self.side = nn.Parameter(torch.randn(2, D) * .02)
        self.time = nn.Parameter(torch.randn(n_time, D) * .02)
        self.cls = nn.Parameter(torch.randn(2, D) * .02)


class Heads(nn.Module):
    def __init__(self, D):
        super().__init__()
        self.mlp = nn.Sequential(nn.LayerNorm(2 * D), nn.Linear(2 * D, D), nn.GELU(), nn.Dropout(0.1))
        self.mal, self.cat, self.risk = nn.Linear(D, 1), nn.Linear(D, 4), nn.Linear(D, 1)

    def forward(self, h):  # h [B,2,D]
        hx = torch.cat([h, h - h.flip(1)], -1)          # own breast + bilateral asymmetry
        z = self.mlp(hx)
        return self.mal(z).squeeze(-1), self.cat(z), self.risk(z).squeeze(-1), z


class LongitudinalChangeModel(nn.Module):
    """temporal: 'change' (proposed) | 'fusion' (prior tokens, no correspondence) | 'diff' (unaligned
    feature subtraction) | 'none' (current exam only).  crossview=False restricts attention to within-view."""

    def __init__(self, C=64, D=64, K=3, temporal="change", align=True, gate=True, crossview=True, layers=2,
                 sigma_s=1.5, dann=False, style_norm=False):
        super().__init__()
        self.D, self.K, self.temporal, self.crossview, self.dann = D, K, temporal, crossview, dann
        self.adapter = Adapter(C, D, style_norm)
        self.emb = TokenEmb(D)
        if temporal == "change":
            self.corr = Correspondence(D, sigma_s, align, gate)
            self.tok = nn.Linear(2 * D + 1, D)
            self.sal = nn.Conv2d(D + 1, 1, 1)                                  # change saliency
            self.dproj = nn.Conv2d(2 * D + 1, D, 1)                            # dense confidence-conditioned fusion
        if temporal == "diff":
            self.dproj = nn.Conv2d(D, D, 1)
        if temporal == "fusion":
            self.pproj = nn.Linear(D, D)
        layer = nn.TransformerEncoderLayer(D, 4, 2 * D, 0.1, batch_first=True, norm_first=True, activation="gelu")
        self.tf = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.heads = Heads(D)
        self.proj = nn.Sequential(nn.Linear(D, D), nn.GELU(), nn.Linear(D, D))    # projection for L_inv
        self.loc = nn.Sequential(nn.Conv2d(2 * D + 1, D, 1), nn.GELU(), nn.Conv2d(D, 1, 1))
        if dann:
            self.disc = nn.Sequential(nn.Linear(D, D), nn.GELU(), nn.Linear(D, 3))

    def pooled_tokens(self, z):  # z [B,4,D,G,G] -> [B,4*36,D]
        B = z.shape[0]
        p = F.avg_pool2d(z.flatten(0, 1), 2).view(B, 4, self.D, 36).permute(0, 1, 3, 2)
        e = self.emb
        p = p + e.pos6[None, None] + e.view[VTYPE][None, :, None] + e.side[SIDE][None, :, None]
        return p.reshape(B, 144, self.D)

    def forward(self, fc, fp, av):
        B = fc.shape[0]
        dev = fc.device
        zc = self.adapter(fc.flatten(0, 1).float()).view(B, 4, self.D, G, G)
        e = self.emb
        cur = self.pooled_tokens(zc if self.temporal != "diff" else zc)
        extra, extra_mask, aux = [], [], {}
        mag = torch.zeros(B, 4, 1, G, G, device=dev)
        if self.temporal in ("change", "diff", "fusion"):
            zp = self.adapter(fp.flatten(0, 2).float()).view(B, 2, 4, self.D, G, G)
        if self.temporal == "change":
            Ds, cs, cycs, sals, Rs = [], [], [], [], []
            for p in range(2):
                out = self.corr(zc.flatten(0, 1), zp[:, p].flatten(0, 1), fc.flatten(0, 1).float(), fp[:, p].flatten(0, 1).float())
                D_ = out["D"].view(B, 4, self.D, G, G)
                c_ = out["c"].view(B, 4, 1, G, G)
                m = (D_.pow(2).sum(2, keepdim=True) + 1e-6).sqrt() * av[:, p].float().view(B, 1, 1, 1, 1)
                mag = mag + m / 2
                sal = self.sal(torch.cat([D_, c_], 2).flatten(0, 1)).view(B, 4, 1, G, G)
                idx = sal.flatten(2).topk(self.K, -1).indices                     # [B,4,K]
                Dg = D_.flatten(3).gather(3, idx.unsqueeze(2).expand(-1, -1, self.D, -1))
                Zg = zc.flatten(3).gather(3, idx.unsqueeze(2).expand(-1, -1, self.D, -1))
                cg = c_.flatten(3).gather(3, idx.unsqueeze(2))
                # dense path: every current token receives the pooled, gated change of its cells
                R_ = out["R"].view(B, 4, self.D, G, G)
                dz = self.dproj(torch.cat([R_, D_, c_], 2).flatten(0, 1)).view(B, 4, self.D, G, G)
                pd = F.avg_pool2d(dz.flatten(0, 1), 2).view(B, 4, self.D, 36).permute(0, 1, 3, 2).reshape(B, 144, self.D)
                cur = cur + 0.5 * pd * av[:, p].float().view(B, 1, 1)
                sg = torch.sigmoid(sal.flatten(2).gather(2, idx)).unsqueeze(-1)        # [B,4,K,1]
                t = self.tok(torch.cat([Dg, Zg, cg], 2).permute(0, 1, 3, 2)) * sg      # [B,4,K,D]
                t = t + e.pos12[idx] + e.view[VTYPE][None, :, None] + e.side[SIDE][None, :, None] + e.time[p + 1]
                extra.append(t.reshape(B, 4 * self.K, self.D))
                extra_mask.append((~av[:, p]).view(B, 1).expand(B, 4 * self.K))
                Ds.append(D_); cs.append(c_); cycs.append(out["cyc"].view(B, 4, 1, G, G)); sals.append(sal)
                Rs.append(out["R"].view(B, 4, self.D, G, G))
                aux[f"flow{p}"] = out["flow"].view(B, 4, 2, G, G)
                aux[f"bw{p}"] = out["bw"].view(B, 4, self.D, G, G)
            aux.update(D=torch.stack(Ds, 1), c=torch.stack(cs, 1), cyc=torch.stack(cycs, 1), zc=zc, zp=zp,
                       sal=torch.stack(sals, 1), R=torch.stack(Rs, 1))
        elif self.temporal == "diff":
            for p in range(2):
                d = (zc - zp[:, p]) * av[:, p].float().view(B, 1, 1, 1, 1)
                m = (d.pow(2).sum(2, keepdim=True) + 1e-6).sqrt()
                mag = mag + m / 2
                cur = cur + 0.5 * self.pooled_tokens(self.dproj(d.flatten(0, 1)).view(B, 4, self.D, G, G)) \
                    * av[:, p].float().view(B, 1, 1)
        elif self.temporal == "fusion":
            for p in range(2):     # prior views as 3x3 pooled tokens per view (no correspondence)
                t = F.avg_pool2d(zp[:, p].flatten(0, 1), 4).view(B, 4, self.D, 9).permute(0, 1, 3, 2)
                t = t + e.pos3[None, None] + e.view[VTYPE][None, :, None] + e.side[SIDE][None, :, None]
                extra.append(self.pproj(t.reshape(B, 36, self.D)) + e.time[p + 1])
                extra_mask.append((~av[:, p]).view(B, 1).expand(B, 36))
        cls = e.cls[None].expand(B, 2, self.D) + e.side[None]
        toks = torch.cat([cls, cur] + extra, 1)
        kpm = torch.cat([torch.zeros(B, 146, dtype=torch.bool, device=dev)] + extra_mask, 1)
        attn = None
        if not self.crossview:
            attn = self.view_mask(toks.shape[1], dev)
        h = self.tf(toks, mask=attn, src_key_padding_mask=kpm)
        mal, cat, risk, z = self.heads(h[:, :2])
        hc = h[:, 2:146].reshape(B, 4, 36, self.D).permute(0, 1, 3, 2).reshape(B * 4, self.D, 6, 6)
        hc = F.interpolate(hc, size=(G, G), mode="nearest")
        heat = self.loc(torch.cat([zc.flatten(0, 1), hc, mag.flatten(0, 1)], 1)).view(B, 4, 144)
        out = dict(mal=mal, cat=cat, risk=risk, heat=heat, emb=z, aux=aux, proj=self.proj(h[:, :2]))
        if self.dann:
            out["dom"] = self.disc(GradReverse.apply(z.mean(1), 1.0))
        return out

    def view_mask(self, n, dev):
        """Block attention to within-view tokens; CLS tokens see only their own breast."""
        view_of = torch.full((n,), -1, dtype=torch.long)
        view_of[2:146] = torch.arange(4).repeat_interleave(36)
        if self.temporal == "change":
            k = 4 * self.K
            for p in range(2):
                view_of[146 + p * k:146 + (p + 1) * k] = torch.arange(4).repeat_interleave(self.K)
        elif self.temporal == "fusion":
            for p in range(2):
                view_of[146 + p * 36:146 + (p + 1) * 36] = torch.arange(4).repeat_interleave(9)
        allow = view_of[:, None] == view_of[None, :]
        for s in range(2):
            own = (view_of // 2 == s) & (view_of >= 0)
            allow[s, own] = True
            allow[own, s] = True
            allow[s, s] = True
        return (~allow).to(dev)


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, l):
        ctx.l = l
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.l * g, None


class SingleViewMIL(nn.Module):
    """Per-view patch-logit multiple-instance model (whole-image classifier family)."""

    def __init__(self, C=64, D=64):
        super().__init__()
        self.adapter = Adapter(C, D)
        self.patch = nn.Conv2d(D, 1, 1)
        self.heads = Heads(D)
        self.D = D

    def forward(self, fc, fp, av):
        B = fc.shape[0]
        z = self.adapter(fc.flatten(0, 1).float())
        pl = self.patch(z).view(B, 4, 144)
        vlog = pl.topk(4, -1).values.mean(-1)                       # view logit
        g = z.view(B, 4, self.D, 144).amax(-1)                       # view descriptor
        hb = g.view(B, 2, 2, self.D).amax(2)
        mal, cat, risk, emb = self.heads(hb)
        mal = mal + vlog.view(B, 2, 2).amax(2)
        return dict(mal=mal, cat=cat, risk=risk, heat=pl, emb=emb, aux={})


class MultiViewLate(nn.Module):
    """Four-view late fusion: pooled CC/MLO descriptors of each breast plus the contralateral breast."""

    def __init__(self, C=64, D=64):
        super().__init__()
        self.adapter = Adapter(C, D)
        self.fuse = nn.Sequential(nn.Linear(8 * D, 2 * D), nn.GELU(), nn.Linear(2 * D, D))
        self.heads = Heads(D)
        self.loc = nn.Conv2d(D, 1, 1)
        self.D = D

    def forward(self, fc, fp, av):
        B = fc.shape[0]
        z = self.adapter(fc.flatten(0, 1).float())
        g = torch.cat([z.mean((2, 3)), z.amax((2, 3))], 1).view(B, 2, 2 * 2 * self.D)
        hb = self.fuse(torch.cat([g, g.flip(1)], -1))
        mal, cat, risk, emb = self.heads(hb)
        return dict(mal=mal, cat=cat, risk=risk, heat=self.loc(z).view(B, 4, 144), emb=emb, aux={})


class AsymTemporalGlobal(nn.Module):
    """Global spatiotemporal-asymmetry model: breast embeddings at each round, with bilateral and
    temporal differences (global-embedding family of longitudinal risk models)."""

    def __init__(self, C=64, D=64):
        super().__init__()
        self.adapter = Adapter(C, D)
        self.bre = nn.Sequential(nn.Linear(4 * D, D), nn.GELU())
        self.fuse = nn.Sequential(nn.Linear(3 * D, D), nn.GELU(), nn.Linear(D, D))
        self.heads = Heads(D)
        self.loc = nn.Conv2d(D, 1, 1)
        self.D = D

    def emb_round(self, f):
        B = f.shape[0]
        z = self.adapter(f.flatten(0, 1).float())
        g = torch.cat([z.mean((2, 3)), z.amax((2, 3))], 1).view(B, 2, 4 * self.D)
        return self.bre(g), z

    def forward(self, fc, fp, av):
        B = fc.shape[0]
        ec, z = self.emb_round(fc)
        w = av.float()
        ep = sum(self.emb_round(fp[:, p])[0] * w[:, p].view(B, 1, 1) for p in range(2)) / (w.sum(1).view(B, 1, 1) + 1e-6)
        ep = torch.where(w.sum(1).view(B, 1, 1) > 0, ep, ec)
        hb = self.fuse(torch.cat([ec, ec - ep, ec - ec.flip(1)], -1))
        mal, cat, risk, emb = self.heads(hb)
        return dict(mal=mal, cat=cat, risk=risk, heat=self.loc(z).view(B, 4, 144), emb=emb, aux={})


def build(name):
    """Registry of every configuration reported in the experiments."""
    R = {
        "SV-MIL": lambda: SingleViewMIL(),
        "MV-Late": lambda: MultiViewLate(),
        "MV-Transformer": lambda: LongitudinalChangeModel(temporal="none"),
        "Prior-Fusion": lambda: LongitudinalChangeModel(temporal="fusion"),
        "Prior-Fusion+DANN": lambda: LongitudinalChangeModel(temporal="fusion", dann=True),
        "Siamese-Diff": lambda: LongitudinalChangeModel(temporal="diff"),
        "Asym-Temporal": lambda: AsymTemporalGlobal(),
        "Proposed": lambda: LongitudinalChangeModel(temporal="change"),
        "Proposed-IN": lambda: LongitudinalChangeModel(temporal="change", style_norm=True),
        "w/o alignment": lambda: LongitudinalChangeModel(temporal="change", align=False),
        "w/o gating": lambda: LongitudinalChangeModel(temporal="change", gate=False),
        "w/o cross-view": lambda: LongitudinalChangeModel(temporal="change", crossview=False),
        "w/o L_acc": lambda: LongitudinalChangeModel(temporal="change"),
        "w/o L_inv": lambda: LongitudinalChangeModel(temporal="change"),
        "w/o L_acc+L_inv": lambda: LongitudinalChangeModel(temporal="change"),
        "w/o counterfactuals": lambda: LongitudinalChangeModel(temporal="change"),
        "w/o multitask": lambda: LongitudinalChangeModel(temporal="change"),
        "K=1": lambda: LongitudinalChangeModel(temporal="change", K=1),
        "K=6": lambda: LongitudinalChangeModel(temporal="change", K=6),
        "sigma=0": lambda: LongitudinalChangeModel(temporal="change", sigma_s=0.0),
        "sigma=3": lambda: LongitudinalChangeModel(temporal="change", sigma_s=3.0),
    }
    for lam in ("0.25", "0.5", "2", "4"):
        R[f"lam_acc={lam}"] = lambda: LongitudinalChangeModel(temporal="change")
    return R[name]()
