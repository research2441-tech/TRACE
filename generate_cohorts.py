"""Generate the development (vendors V1-V3) and external (unseen vendors V4/V5, denser
population) longitudinal cohorts.  Output: uint8 image tensors [N, 3 rounds, 4 views, H, W],
an acquisition-counterfactual copy for the development cohort, and per-breast labels."""
import numpy as np, json, os, sys
from multiprocessing import Pool
from simulator import Woman, render, random_warp, apply_warp, fwd_point, H, W, DEV_VENDORS, EXT_VENDORS

OUT = sys.argv[1] if len(sys.argv) > 1 else "../data"
VIEW_LIST = [("L", "CC"), ("L", "MLO"), ("R", "CC"), ("R", "MLO")]
VID = {"V1": 0, "V2": 1, "V3": 2, "V4": 3, "V5": 4}

COHORTS = {
    "dev": dict(n=6000, seed=11, dens=[.10, .40, .40, .10], vendors=DEV_VENDORS, p_same=.82,
                legacy=DEV_VENDORS, p_cancer=.16, p_seed=.12, p_benign=.22, p_newbenign=.10, cf=True),
    # disjoint pretraining cohort (current round only, report-level finding labels)
    "pre": dict(n=8000, seed=5, dens=[.10, .40, .40, .10], vendors=DEV_VENDORS, p_same=1.0,
                legacy=DEV_VENDORS, p_cancer=.20, p_seed=.0, p_benign=.25, p_newbenign=.0, cf=False),
    "ext": dict(n=2000, seed=97, dens=[.04, .24, .44, .28], vendors=EXT_VENDORS, p_same=.50,
                legacy=DEV_VENDORS, p_cancer=.16, p_seed=.12, p_benign=.26, p_newbenign=.12, cf=False),
}


PRIOR_WARP = 1.0                        # positioning-deformation strength of prior rounds (stress test)


def simulate(args):
    cname, i = args[:2]
    warp_s = args[2] if len(args) > 2 else PRIOR_WARP
    unseen_cf = cname == "devu"          # same women as "dev", counterfactual priors through unseen chains
    if unseen_cf or cname == "devw":
        cname_orig = cname
        cname = "dev"
    else:
        cname_orig = cname
    C = COHORTS[cname]
    rng = np.random.default_rng(C["seed"] * 100003 + i)
    dens = int(rng.choice(4, p=C["dens"]))
    n_prior = int(rng.choice(3, p=[.15, .25, .60])) if cname != "pre" else 0
    wm = Woman(rng, dens, n_prior, C["p_cancer"], C["p_seed"], C["p_benign"], C["p_newbenign"])
    vc = str(rng.choice(C["vendors"]))
    vend = [vc]
    for k in range(1, 3):
        if rng.random() < C["p_same"]:
            vend.append(vc)
        else:
            pool = [v for v in (C["vendors"] + C["legacy"]) if v != vc]
            vend.append(str(rng.choice(pool)))
    vcf = str(rng.choice([v for v in DEV_VENDORS if v != vc])) if C["cf"] else vc
    if unseen_cf:
        vcf = EXT_VENDORS[i % 2]
    img = np.zeros((3, 4, H, W), np.uint8)
    cf = np.zeros((3, 4, H, W), np.uint8) if C["cf"] else None
    loc = np.full((4, 3), -1.0, np.float32)      # per view: x, y, r of the malignant lesion (current round)
    for k in range(0, n_prior + 1):
        for j, (s, v) in enumerate(VIEW_LIST):
            T, mask, thick, gts = wm.tissue(s, v, k)
            wp = random_warp(rng, 1.0 if k == 0 else warp_s)
            Tw, Mw, Kw = apply_warp(T, wp), (apply_warp(mask, wp, 0) > .5).astype(np.float32), apply_warp(thick, wp)
            img[k, j] = render(Tw, Mw, Kw, vend[k], rng)
            if C["cf"]:
                cf[k, j] = render(Tw, Mw, Kw, vcf, rng)
            if k == 0:
                for g in gts:
                    if g["malignant"] and g["r"] >= 2.0:
                        x, y = fwd_point(wp, g["x"], g["y"])
                        loc[j] = (x, y, g["r"])
    lab = {}
    for s in "LR":
        les = [l for l in wm.lesions if l["side"] == s]
        cancer = any(l["malignant"] and l["visible_now"] for l in les)
        seed = any(l["malignant"] and not l["visible_now"] for l in les)
        stable_b = any((not l["malignant"]) and l.get("stable", False) for l in les)
        new_b = any((not l["malignant"]) and not l.get("stable", True) for l in les)
        if cancer:
            cat = 3
        elif new_b or (stable_b and n_prior == 0):
            cat = 2
        elif stable_b:
            cat = 1
        else:
            cat = 0
        lab[s] = dict(cancer=int(cancer), risk=int(seed and not cancer), cat=cat,
                      stable_benign=int(stable_b), new_benign=int(new_b),
                      calc=int(any(l["malignant"] and l["kind"] == "calc" for l in les)))
    meta = dict(density=dens, n_prior=n_prior, vendors=[VID[v] for v in vend], vcf=VID[vcf], lab=lab)
    return i, img, cf, loc, meta


def run(cname):
    C = COHORTS[cname]
    n = C["n"]
    os.makedirs(OUT, exist_ok=True)
    imgs = np.lib.format.open_memmap(f"{OUT}/{cname}_img.npy", "w+", np.uint8, (n, 3, 4, H, W))
    cfs = np.lib.format.open_memmap(f"{OUT}/{cname}_cf.npy", "w+", np.uint8, (n, 3, 4, H, W)) if C["cf"] else None
    locs = np.zeros((n, 4, 3), np.float32)
    metas = [None] * n
    with Pool(2) as p:
        for i, img, cf, loc, meta in p.imap_unordered(simulate, [(cname, i) for i in range(n)], chunksize=16):
            imgs[i] = img
            if cfs is not None:
                cfs[i] = cf
            locs[i] = loc
            metas[i] = meta
            if i % 500 == 0:
                print(cname, i, flush=True)
    imgs.flush()
    np.save(f"{OUT}/{cname}_loc.npy", locs)
    json.dump(metas, open(f"{OUT}/{cname}_meta.json", "w"))


def run_unseen_cf(idx):
    """Counterfactual renderings through V4/V5 for the given development women (test split)."""
    n = len(idx)
    cfs = np.lib.format.open_memmap(f"{OUT}/devu_cf.npy", "w+", np.uint8, (n, 3, 4, H, W))
    ref = np.load(f"{OUT}/dev_img.npy", mmap_mode="r")
    with Pool(2) as p:
        for k, (i, img, cf, loc, meta) in enumerate(p.imap(simulate, [("devu", int(i)) for i in idx], chunksize=8)):
            assert np.array_equal(img, ref[i]), "stream mismatch"
            cfs[k] = cf
    cfs.flush()
    np.save(f"{OUT}/devu_idx.npy", np.asarray(idx))


def run_warp(idx, strength):
    """Test women re-rendered with stronger positioning deformation of the priors (native chains)."""
    n = len(idx)
    out = np.lib.format.open_memmap(f"{OUT}/devw{int(strength * 10)}_cf.npy", "w+", np.uint8, (n, 3, 4, H, W))
    ref = np.load(f"{OUT}/dev_img.npy", mmap_mode="r")
    with Pool(2) as p:
        for k, (i, img, cf, loc, meta) in enumerate(p.imap(simulate, [("devw", int(i), strength) for i in idx], chunksize=8)):
            assert np.array_equal(img[0], ref[i, 0]), "stream mismatch"
            out[k] = img
    out.flush()


if __name__ == "__main__":
    if sys.argv[2] == "devw":
        from common import splits
        for s_ in sys.argv[3:]:
            run_warp(splits()["test"], float(s_))
    elif sys.argv[2:] == ["devu"]:
        from common import splits
        run_unseen_cf(splits()["test"])
    else:
        for c in (sys.argv[2:] or ["dev", "ext"]):
            run(c)
