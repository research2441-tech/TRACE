"""Train one configuration with one seed on cached frozen-encoder features and store predictions.
usage: python train.py "<config name>" <seed> [epochs]"""
import sys, time, json, numpy as np, torch, torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from common import splits, load_meta, DATA, RES
from models import build

if __name__ == "__main__":
    NAME, SEED = sys.argv[1], int(sys.argv[2])
    EPOCHS = int(sys.argv[3]) if len(sys.argv) > 3 else 8
else:                                   # imported for its helpers
    NAME, SEED, EPOCHS = "Proposed", 0, 8
torch.manual_seed(SEED); np.random.seed(SEED)
torch.set_num_threads(1)

# ---- loss switches per configuration
LAM = dict(loc=1.0, cat=0.5, risk=1.0, corr=0.2, acc=1.0, inv=0.1, sal=0.5)
if NAME in ("w/o L_acc", "w/o L_acc+L_inv", "w/o counterfactuals"):
    LAM["acc"] = 0.0
if NAME in ("w/o L_inv", "w/o L_acc+L_inv", "w/o counterfactuals"):
    LAM["inv"] = 0.0
if NAME.startswith("lam_acc="):
    LAM["acc"] = float(NAME.split("=")[1])
if NAME == "w/o multitask":
    LAM["cat"] = 0.0; LAM["loc"] = 0.0
PROPOSED_FAMILY = not NAME.split("+")[0] in ("SV-MIL", "MV-Late", "MV-Transformer", "Prior-Fusion", "Siamese-Diff", "Asym-Temporal")
if not PROPOSED_FAMILY:
    LAM["acc"] = LAM["inv"] = LAM["corr"] = LAM["sal"] = 0.0


def labels(meta, locs):
    N = len(meta)
    y = dict(mal=np.zeros((N, 2), np.float32), cat=np.zeros((N, 2), np.int64), risk=np.zeros((N, 2), np.float32),
             av=np.zeros((N, 2), bool), vend=np.zeros(N, np.int64), dens=np.zeros(N, np.int64),
             sw=np.zeros(N, bool), cell=np.full((N, 4), -1, np.int64))
    for i, m in enumerate(meta):
        for s, S in enumerate("LR"):
            y["mal"][i, s] = m["lab"][S]["cancer"]; y["cat"][i, s] = m["lab"][S]["cat"]; y["risk"][i, s] = m["lab"][S]["risk"]
        y["av"][i] = [m["n_prior"] >= 1, m["n_prior"] >= 2]
        y["vend"][i] = m["vendors"][0]; y["dens"][i] = m["density"]
        y["sw"][i] = m["n_prior"] >= 1 and m["vendors"][1] != m["vendors"][0]
        for j in range(4):
            if locs[i, j, 0] >= 0:
                cx, cy = int(np.clip(locs[i, j, 0] // 8, 0, 11)), int(np.clip(locs[i, j, 1] // 8, 0, 11))
                y["cell"][i, j] = cy * 12 + cx
    return y


def batch(feat, idx, y, cf=None, mode="native"):
    idx = np.sort(idx)
    x = torch.from_numpy(np.asarray(feat[idx]))
    if mode == "switch":
        xc = torch.from_numpy(np.asarray(cf[idx])); x = torch.cat([x[:, :1], xc[:, 1:]], 1)
    elif mode == "switchu":            # cf = (devu features, position map) for unseen-chain priors
        arr, pos = cf
        xc = torch.from_numpy(np.stack([np.asarray(arr[pos[i]]) for i in idx])); x = torch.cat([x[:, :1], xc[:, 1:]], 1)
    elif mode == "cf":
        x = torch.from_numpy(np.asarray(cf[idx]))
    t = {k: torch.from_numpy(v[idx]) for k, v in y.items()}
    return x[:, 0], x[:, 1:], t, idx


def sup_loss(o, t):
    L = {}
    L["mal"] = F.binary_cross_entropy_with_logits(o["mal"], t["mal"], pos_weight=torch.tensor(3.0))
    L["cat"] = F.cross_entropy(o["cat"].reshape(-1, 4), t["cat"].reshape(-1))
    neg = (t["mal"] == 0).float()
    L["risk"] = (F.binary_cross_entropy_with_logits(o["risk"], t["risk"], pos_weight=torch.tensor(4.0), reduction="none") * neg).sum() / neg.sum().clamp(min=1)
    cell = t["cell"].reshape(-1)
    m = cell >= 0
    L["loc"] = F.cross_entropy(o["heat"].reshape(-1, 144)[m], cell[m]) if m.any() else o["heat"].sum() * 0
    return L


def corr_loss(o):
    """Self-supervised correspondence on detached features: robust residual + cycle term."""
    a = o["aux"]
    if "zc" not in a:
        return torch.zeros(())
    from models import warp
    zc = a["zc"].detach()
    B = zc.shape[0]
    r = 0
    for p in range(2):
        bw = warp(a["zp"][:, p].detach().flatten(0, 1), a[f"flow{p}"].flatten(0, 1)).view_as(zc)
        r = r + (zc - bw).pow(2).mean(2).add(1e-3).sqrt().mean()
    return r / 2 + 0.1 * a["cyc"].mean()


def sal_loss(o, t):
    """Change-saliency supervision: the lesion cell of current cancers should rank first among cells."""
    a = o["aux"]
    if "sal" not in a:
        return torch.zeros(())
    cell = t["cell"]                                         # [B,4]
    L, n = 0, 0
    for p in range(2):
        m = (cell >= 0) & t["av"][:, p:p + 1]
        if m.any():
            L = L + F.cross_entropy(a["sal"][:, p].reshape(-1, 144)[m.reshape(-1)], cell.reshape(-1)[m.reshape(-1)]); n += 1
    return L / max(n, 1)


def train():
    meta = load_meta("dev"); locs = np.load(f"{DATA}/dev_loc.npy")
    y = labels(meta, locs)
    feat = np.load(f"{DATA}/dev_img_feat.npy", mmap_mode="r")
    cf = np.load(f"{DATA}/dev_cf_feat.npy", mmap_mode="r")
    sp = splits()
    model = build(NAME)
    opt = torch.optim.AdamW(model.parameters(), 1e-3, weight_decay=0.02)
    B = 32
    steps = EPOCHS * (len(sp["train"]) // B)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=steps, pct_start=0.1)
    rng = np.random.default_rng(SEED)
    best, best_state, hist = -1, None, []
    t0 = time.time()
    for ep in range(EPOCHS):
        model.train()
        perm = rng.permutation(sp["train"])
        for b in range(len(perm) // B):
            idx = perm[b * B:(b + 1) * B]
            fc, fp, t, idx = batch(feat, idx, y)
            o = model(fc, fp, t["av"])
            L = sup_loss(o, t)
            loss = L["mal"] + LAM["cat"] * L["cat"] + LAM["risk"] * L["risk"] + LAM["loc"] * L["loc"]
            # label-preserving acquisition counterfactuals (all models): switched priors or full re-rendering
            half = len(idx) // 2
            if NAME == "w/o counterfactuals":
                loss = loss + LAM["corr"] * corr_loss(o) + LAM["sal"] * sal_loss(o, t)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                opt.step(); sch.step()
                continue
            fc_s, fp_s, _, _ = batch(feat, idx[:half], y, cf, "switch")
            fc_c, fp_c, _, _ = batch(feat, idx[half:], y, cf, "cf")
            o2 = model(torch.cat([fc_s, fc_c]), torch.cat([fp_s, fp_c]), t["av"])
            L2 = sup_loss(o2, t)
            loss = loss + L2["mal"] + LAM["cat"] * L2["cat"] + LAM["risk"] * L2["risk"] + LAM["loc"] * L2["loc"]
            loss = loss / 2
            if LAM["corr"] > 0:
                loss = loss + LAM["corr"] * corr_loss(o)
            if LAM["sal"] > 0 and "sal" in o["aux"]:
                loss = loss + LAM["sal"] * 0.5 * (sal_loss(o, t) + sal_loss(o2, t))
            if LAM["acc"] > 0 and "R" in o["aux"]:
                # acquisition-counterfactual consistency of change residuals, weighted by (detached) confidence
                Rn, Rs = o["aux"]["R"][:half], o2["aux"]["R"][:half]
                w = o["aux"]["c"][:half].detach() * t["av"][:half].float().view(half, 2, 1, 1, 1, 1)
                den = (Rn.detach().pow(2) * w).mean() + 1e-3
                acc = ((Rn - Rs).pow(2) * w).mean() / den
                loss = loss + LAM["acc"] * acc
            if LAM["inv"] > 0 and "proj" in o:
                za = F.normalize(o["proj"][half:].reshape(-1, o["proj"].shape[-1]), dim=1)
                zb = F.normalize(o2["proj"][half:].reshape(-1, o["proj"].shape[-1]), dim=1)
                s = za @ zb.t() / 0.2
                tg = torch.arange(s.shape[0])
                loss = loss + LAM["inv"] * 0.5 * (F.cross_entropy(s, tg) + F.cross_entropy(s.t(), tg))
            if "dom" in o:
                vend_all = torch.cat([t["vend"][:half], t["vend"][half:]])
                loss = loss + 0.3 * F.cross_entropy(o["dom"], t["vend"]) + 0.3 * F.cross_entropy(o2["dom"][half:], torch.from_numpy(np.array([meta[i]["vcf"] for i in idx[half:]])))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step(); sch.step()
        pv = predict(model, feat, sp["val"], y)
        auc_m = roc_auc_score(y["mal"][sp["val"]].max(1), pv["mal"].max(1))
        neg = y["mal"][sp["val"]] == 0
        auc_r = roc_auc_score(y["risk"][sp["val"]][neg], pv["risk"][neg])
        score = 0.5 * auc_m + 0.5 * auc_r
        hist.append(dict(ep=ep, auc_mal=auc_m, auc_risk=auc_r, t=time.time() - t0))
        print(NAME, SEED, hist[-1], flush=True)
        if score > best:
            best = score
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, feat, cf, y, sp, hist


def unseen_prior_source(name="devu"):
    arr = np.load(f"{DATA}/{name}_cf_feat.npy", mmap_mode="r")
    pos = {int(i): k for k, i in enumerate(np.load(f"{DATA}/devu_idx.npy"))}
    return arr, pos


@torch.no_grad()
def predict(model, feat, idx, y, cf=None, mode="native"):
    model.eval()
    out = {k: [] for k in ("mal", "cat", "risk", "cell", "emb")}
    for s in range(0, len(idx), 64):
        ii = idx[s:s + 64]
        order = np.argsort(ii)
        fc, fp, t, _ = batch(feat, ii, y, cf, mode)
        o = model(fc, fp, t["av"])
        inv = np.argsort(order)
        out["mal"].append(o["mal"].numpy()[inv]); out["cat"].append(o["cat"].numpy()[inv])
        out["risk"].append(o["risk"].numpy()[inv]); out["cell"].append(o["heat"].argmax(-1).numpy()[inv])
        out["emb"].append(o["emb"].numpy()[inv])
    model.train()
    return {k: np.concatenate(v) for k, v in out.items()}


if __name__ == "__main__":
    t0 = time.time()
    model, feat, cf, y, sp, hist = train()
    P = {}
    for split in ("val", "cal", "test"):
        for k, v in predict(model, feat, sp[split], y).items():
            P[f"{split}_{k}"] = v
    for k, v in predict(model, feat, sp["test"], y, cf, "switch").items():
        P[f"testsw_{k}"] = v
    for src, key in (("devu", "testswu"), ("devw20", "testw20"), ("devw30", "testw30")):
        for k, v in predict(model, feat, sp["test"], y, unseen_prior_source(src), "switchu").items():
            P[f"{key}_{k}"] = v
    meta_e = load_meta("ext"); ye = labels(meta_e, np.load(f"{DATA}/ext_loc.npy"))
    fe = np.load(f"{DATA}/ext_img_feat.npy", mmap_mode="r")
    for k, v in predict(model, fe, np.arange(len(meta_e)), ye).items():
        P[f"ext_{k}"] = v
    tag = NAME.replace("/", "").replace(" ", "_").replace("+", "p").replace("=", "")
    np.savez_compressed(f"{RES}/pred_{tag}_s{SEED}.npz", **P)
    nparam = sum(p.numel() for p in model.parameters())
    json.dump(dict(name=NAME, seed=SEED, hist=hist, params=nparam, train_time=time.time() - t0),
              open(f"{RES}/log_{tag}_s{SEED}.json", "w"))
    torch.save(model.state_dict(), f"{RES}/model_{tag}_s{SEED}.pt")
    print("done", NAME, SEED, time.time() - t0)
