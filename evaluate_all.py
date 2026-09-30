"""Aggregate every stored prediction file into the result tables used in the manuscript."""
import numpy as np, json, glob, os, re
from collections import defaultdict
from sklearn.metrics import roc_auc_score
from common import splits, load_meta, DATA, RES
from train import labels
import metrics as M

sp = splits()
meta = load_meta("dev"); Y = labels(meta, np.load(f"{DATA}/dev_loc.npy")); LOC = np.load(f"{DATA}/dev_loc.npy")
me = load_meta("ext"); YE = labels(me, np.load(f"{DATA}/ext_loc.npy")); LOCE = np.load(f"{DATA}/ext_loc.npy")


def sub(y, idx):
    return {k: v[idx] for k, v in y.items()}


SETS = {"test": (sub(Y, sp["test"]), LOC[sp["test"]]), "testsw": (sub(Y, sp["test"]), LOC[sp["test"]]),
        "testswu": (sub(Y, sp["test"]), LOC[sp["test"]]), "testw20": (sub(Y, sp["test"]), LOC[sp["test"]]),
        "testw30": (sub(Y, sp["test"]), LOC[sp["test"]]), "ext": (YE, LOCE)}
YV, YC = sub(Y, sp["val"]), sub(Y, sp["cal"])


def load(tag):
    out = {}
    for f in sorted(glob.glob(f"{RES}/pred_{tag}_s*.npz")):
        s = int(re.findall(r"_s(\d+)\.npz", f)[0])
        out[s] = dict(np.load(f))
    return out


def run_metrics(P):
    T = M.fit_temperature(P["val_mal"].max(1), YV["mal"].max(1))
    r = {}
    for k, (y, loc) in SETS.items():
        if f"{k}_mal" not in P:
            continue
        r[k] = M.core_metrics(P, k, y, loc, T)
        # operating point transferred from the calibration split (90% specificity there)
        pc = M.sig(P["cal_mal"].max(1) / T)
        thr = np.quantile(pc[YC["mal"].max(1) == 0], 0.9)
        pe = M.sig(P[f"{k}_mal"].max(1) / T); ye = y["mal"].max(1)
        r[k]["op_sens"] = float((pe[ye == 1] >= thr).mean()); r[k]["op_spec"] = float((pe[ye == 0] < thr).mean())
    r["T"] = T
    return r


def strata(P, T):
    """AUC per clinically relevant stratum (seed-level)."""
    out = {}
    def auc(pre, y, mask):
        ye = y["mal"].max(1)[mask]; pe = P[f"{pre}_mal"].max(1)[mask]
        return float(roc_auc_score(ye, pe)) if 0 < ye.mean() < 1 else np.nan
    def rauc(pre, y, mask):
        neg = (y["mal"][mask] == 0).reshape(-1)
        return float(roc_auc_score(y["risk"][mask].reshape(-1)[neg], P[f"{pre}_risk"][mask].reshape(-1)[neg]))
    y = SETS["test"][0]
    n_prior = y["av"].sum(1)
    for name, m in [("dens_ab", y["dens"] <= 1), ("dens_cd", y["dens"] >= 2), ("prior0", n_prior == 0),
                    ("prior1", n_prior == 1), ("prior2", n_prior == 2), ("same_vendor", (~y["sw"]) & (n_prior > 0)),
                    ("switched_vendor", y["sw"])]:
        out["test_" + name] = auc("test", y, m); out["test_" + name + "_risk"] = rauc("test", y, m)
    ye = YE
    n_prior = ye["av"].sum(1)
    for name, m in [("V4", ye["vend"] == 3), ("V5", ye["vend"] == 4), ("dens_ab", ye["dens"] <= 1),
                    ("dens_cd", ye["dens"] >= 2), ("prior0", n_prior == 0), ("legacy_prior", ye["sw"]),
                    ("same_prior", (~ye["sw"]) & (n_prior > 0))]:
        out["ext_" + name] = auc("ext", ye, m); out["ext_" + name + "_risk"] = rauc("ext", ye, m)
    # static-lesion false positives: cancer-free breasts carrying a stable benign mass with priors
    pb = M.sig(P["test_mal"] / T)
    stable = (y["cat"] == 1) & (y["mal"] == 0)
    newb = (y["cat"] == 2) & (y["mal"] == 0)
    clean = (y["cat"] == 0) & (y["mal"] == 0)
    thr = np.quantile(M.sig(P["test_mal"].max(1) / T)[y["mal"].max(1) == 0], 0.9)
    out["fp_stable_benign"] = float((pb[stable] >= thr).mean())
    out["fp_new_benign"] = float((pb[newb] >= thr).mean())
    out["fp_clean"] = float((pb[clean] >= thr).mean())
    return out


def summarise(tags):
    table = {}
    for tag in tags:
        runs = load(tag)
        if not runs:
            continue
        per = {s: run_metrics(P) for s, P in runs.items()}
        st = {s: strata(P, per[s]["T"]) for s, P in runs.items()}
        agg = {}
        for k in [k for k in SETS if k in per[next(iter(per))]]:
            for m in per[next(iter(per))][k]:
                v = np.array([per[s][k][m] for s in per])
                agg[f"{k}.{m}"] = (float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0)
        for m in st[next(iter(st))]:
            v = np.array([st[s][m] for s in st])
            agg[f"strata.{m}"] = (float(np.nanmean(v)), float(np.nanstd(v, ddof=1)) if len(v) > 1 else 0.0)
        logs = [json.load(open(f)) for f in glob.glob(f"{RES}/log_{tag}_s*.json")]
        agg["params"] = logs[0].get("params", 0) if logs else 0
        agg["train_time"] = float(np.mean([l["train_time"] for l in logs])) if logs else 0
        agg["n_seeds"] = len(runs)
        table[tag] = agg
    return table


def ensemble(tag, key):
    runs = load(tag)
    return np.mean([M.sig(P[key].max(1)) for P in runs.values()], 0)


def ensemble_risk(tag, key, neg):
    runs = load(tag)
    return np.mean([M.sig(P[key]).reshape(-1)[neg] for P in runs.values()], 0)


def significance(ref, others):
    res = {}
    for k in ("test", "testsw", "testswu", "testw30", "ext"):
        y = SETS[k][0]
        ye = y["mal"].max(1)
        neg = y["mal"].reshape(-1) == 0
        yr = y["risk"].reshape(-1)[neg]
        a = ensemble(ref, f"{k}_mal"); ar = ensemble_risk(ref, f"{k}_risk", neg)
        rows, pv, pvr = [], [], []
        for o in others:
            if not load(o):
                continue
            b = ensemble(o, f"{k}_mal"); br = ensemble_risk(o, f"{k}_risk", neg)
            a1, a2, z, p = M.delong(ye, a, b)
            r1, r2, zr, pr = M.delong(yr, ar, br)
            ci = M.boot_diff(ye, a, b, n=1000)
            rows.append(dict(model=o, auc_ref=a1, auc=a2, diff=a1 - a2, ci=ci.tolist(), z=z, p=p,
                             risk_ref=r1, risk=r2, rdiff=r1 - r2, pr=pr))
            pv.append(p); pvr.append(pr)
        adj, adjr = M.holm(pv), M.holm(pvr)
        for r, q, qr in zip(rows, adj, adjr):
            r["p_holm"] = float(q); r["pr_holm"] = float(qr)
        res[k] = rows
    return res


def conformal(tags, alpha=0.05, delta=0.05):
    out = {}
    for tag in tags:
        runs = load(tag)
        if not runs:
            continue
        rr = defaultdict(list)
        for s, P in runs.items():
            T = M.fit_temperature(P["val_mal"].max(1), YV["mal"].max(1))
            Pv = {k[4:]: v for k, v in P.items() if k.startswith("val_")}
            Pc = {k[4:]: v for k, v in P.items() if k.startswith("cal_")}
            for k in ("test", "testsw", "testswu", "ext"):
                Pt = {kk[len(k) + 1:]: v for kk, v in P.items() if kk.startswith(k + "_")}
                for mode in ("marginal", "mondrian", "proposed"):
                    r = M.decision_layer(Pv, Pc, Pt, YV["mal"], YC["mal"], SETS[k][0]["mal"], alpha, delta, T, mode)
                    rr[(k, mode)].append(r)
        out[tag] = {f"{k}|{mode}": {m: (float(np.nanmean([d[m] for d in v])), float(np.nanstd([d[m] for d in v])))
                                    for m in v[0]} for (k, mode), v in rr.items()}
    return out


if __name__ == "__main__":
    tags = sorted({re.sub(r"_s\d+\.npz$", "", os.path.basename(f))[5:] for f in glob.glob(f"{RES}/pred_*.npz")})
    print(tags)
    T = summarise(tags)
    json.dump(T, open(f"{RES}/summary.json", "w"), indent=1)
    base = [t for t in tags if t in ("Classical-HGB", "SV-MIL", "MV-Late", "MV-Transformer", "Prior-Fusion",
                                     "Prior-FusionpDANN", "Siamese-Diff", "Asym-Temporal")]
    if "Proposed" in tags:
        json.dump(significance("Proposed", base), open(f"{RES}/significance.json", "w"), indent=1)
    json.dump(conformal([t for t in tags if t in base + ["Proposed", "wo_L_accpL_inv", "wo_counterfactuals", "wo_L_acc", "wo_L_inv"]]),
              open(f"{RES}/conformal.json", "w"), indent=1)
    for t, a in T.items():
        print(f"{t:22s} test {a['test.auc'][0]:.3f} sw {a['testsw.auc'][0]:.3f} swu {a.get('testswu.auc',(0,0))[0]:.3f} ext {a['ext.auc'][0]:.3f} | "
              f"risk {a['test.risk_auc'][0]:.3f}/{a['ext.risk_auc'][0]:.3f} | loc {a['test.loc'][0]:.3f} | ece {a['test.ece'][0]:.3f}/{a['ext.ece'][0]:.3f} n={a['n_seeds']}")
