"""Classical baseline: hand-crafted intensity, blob (LoG), texture, bilateral-asymmetry and
rigid-temporal-subtraction descriptors with gradient-boosted trees (one model per target)."""
import numpy as np, json, sys, time
from scipy import ndimage as ndi
from sklearn.ensemble import HistGradientBoostingClassifier
from common import splits, load_meta, DATA, RES
from train import labels


def view_feats(x, xp):
    x = x.astype(np.float32) / 255
    m = x > 0.03
    core = ndi.binary_erosion(m, iterations=4)
    f = [x[m].mean() if m.any() else 0, x[m].std() if m.any() else 0, (x[m] > .6).mean() if m.any() else 0]
    for s in (1.5, 2.5, 4.0):
        d = -ndi.gaussian_laplace(x, s) * s ** 2
        d[~core] = 0
        f += [d.max(), np.sort(d.ravel())[-10:].mean()]
    if xp is not None:
        xp = xp.astype(np.float32) / 255
        # rigid alignment by centre of mass, then smoothed subtraction
        c1, c2 = ndi.center_of_mass(m), ndi.center_of_mass(xp > .03)
        xs = ndi.shift(xp, (c1[0] - c2[0], c1[1] - c2[1]), order=1)
        d = ndi.gaussian_filter(x, 1.5) - ndi.gaussian_filter(xs, 1.5)
        d[~core] = 0
        f += [d.max(), np.sort(d.ravel())[-10:].mean(), np.abs(d).mean(), 1.0]
    else:
        f += [0, 0, 0, 0.0]
    return f


def exam_feats(img, i):
    out = []
    for s in range(2):
        fs = []
        for j in (2 * s, 2 * s + 1):
            has_p = img[i, 1, j].max() > 0
            fs.append(view_feats(img[i, 0, j], img[i, 1, j] if has_p else None))
        out.append(np.array(fs).max(0).tolist() + np.array(fs).mean(0).tolist())
    out = np.array(out)
    return np.concatenate([out, out - out[::-1]], 1)     # own + bilateral difference


def build_X(img, idx):
    return np.stack([exam_feats(img, i) for i in idx])   # [n,2,F]


if __name__ == "__main__":
    t0 = time.time()
    sp = splits()
    meta = load_meta("dev"); y = labels(meta, np.load(f"{DATA}/dev_loc.npy"))
    img = np.load(f"{DATA}/dev_img.npy", mmap_mode="r")
    cf = np.load(f"{DATA}/dev_cf.npy", mmap_mode="r")
    X = {k: build_X(img, sp[k]) for k in ("train", "val", "cal", "test")}
    # vendor-switched test: current native + counterfactual priors
    class SW:
        def __getitem__(self, key):
            i, k, j = key
            return img[i, 0, j] if k == 0 else cf[i, k, j]
    X["testsw"] = np.stack([exam_feats(SW(), i) for i in sp["test"]])
    cfu = np.load(f"{DATA}/devu_cf.npy", mmap_mode="r")
    posu = {int(i): k for k, i in enumerate(np.load(f"{DATA}/devu_idx.npy"))}
    class SWU:
        def __getitem__(self, key):
            i, k, j = key
            return img[i, 0, j] if k == 0 else cfu[posu[i], k, j]
    X["testswu"] = np.stack([exam_feats(SWU(), i) for i in sp["test"]])
    for nm in ("devw20", "devw30"):
        arrw = np.load(f"{DATA}/{nm}_cf.npy", mmap_mode="r")
        class SWW:
            def __getitem__(self, key, a=arrw):
                i, k, j = key
                return img[i, 0, j] if k == 0 else a[posu[i], k, j]
        X["testw" + nm[4:]] = np.stack([exam_feats(SWW(), i) for i in sp["test"]])
    me = load_meta("ext"); ye = labels(me, np.load(f"{DATA}/ext_loc.npy"))
    X["ext"] = build_X(np.load(f"{DATA}/ext_img.npy", mmap_mode="r"), np.arange(len(me)))
    print("features", time.time() - t0, flush=True)
    for seed in range(3):
        P = {}
        Xtr = X["train"].reshape(-1, X["train"].shape[-1])
        mdl = {}
        for tgt in ("mal", "risk", "cat"):
            yt = y[tgt][sp["train"]].reshape(-1)
            mk = np.ones_like(yt, bool) if tgt != "risk" else (y["mal"][sp["train"]].reshape(-1) == 0)
            mdl[tgt] = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=seed).fit(Xtr[mk], yt[mk])
        for k in ("val", "cal", "test", "testsw", "testswu", "testw20", "testw30", "ext"):
            Xk = X[k].reshape(-1, X[k].shape[-1]); n = X[k].shape[0]
            pm = np.clip(mdl["mal"].predict_proba(Xk)[:, 1], 1e-5, 1 - 1e-5)
            pr = np.clip(mdl["risk"].predict_proba(Xk)[:, 1], 1e-5, 1 - 1e-5)
            P[f"{k}_mal"] = np.log(pm / (1 - pm)).reshape(n, 2)
            P[f"{k}_risk"] = np.log(pr / (1 - pr)).reshape(n, 2)
            P[f"{k}_cat"] = np.log(mdl["cat"].predict_proba(Xk) + 1e-6).reshape(n, 2, 4)
            P[f"{k}_cell"] = np.full((n, 4), 0)          # no localisation output
            P[f"{k}_emb"] = Xk.reshape(n, 2, -1)
        np.savez_compressed(f"{RES}/pred_Classical-HGB_s{seed}.npz", **P)
        json.dump(dict(name="Classical-HGB", seed=seed, train_time=time.time() - t0), open(f"{RES}/log_Classical-HGB_s{seed}.json", "w"))
    print("done", time.time() - t0)
