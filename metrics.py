"""Evaluation metrics, calibration, conformal decision layer and statistical tests."""
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, cohen_kappa_score, roc_curve


def sig(x):
    return 1 / (1 + np.exp(-x))


def sens_at_spec(y, s, spec=0.9):
    fpr, tpr, _ = roc_curve(y, s)
    return float(np.interp(1 - spec, fpr, tpr))


def ece(y, p, bins=15):
    e = 0.0
    edges = np.linspace(0, 1, bins + 1)
    for a, b in zip(edges[:-1], edges[1:]):
        m = (p > a) & (p <= b)
        if m.any():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def fit_temperature(logit, y):
    best, bt = 1e9, 1.0
    for T in np.exp(np.linspace(np.log(0.25), np.log(6), 120)):
        p = np.clip(sig(logit / T), 1e-6, 1 - 1e-6)
        nll = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        if nll < best:
            best, bt = nll, T
    return bt


def loc_hit(cell, loc):
    """cell [N,4] predicted argmax cells, loc [N,4,3] (x,y,r) -> hit per exam in >=1 view with a lesion."""
    cx, cy = (cell % 12) * 8 + 4, (cell // 12) * 8 + 4
    has = loc[:, :, 0] >= 0
    d = np.sqrt((cx - loc[:, :, 0]) ** 2 + (cy - loc[:, :, 1]) ** 2)
    hit = (d <= loc[:, :, 2] + 8) & has
    return hit.any(1), has.any(1)


def exam_scores(P, pre):
    return P[f"{pre}_mal"].max(1)


def core_metrics(P, pre, y, loc, T=1.0):
    """y: dict of label arrays restricted to this evaluation set."""
    ye = y["mal"].max(1)
    le = exam_scores(P, pre) / T
    pe = sig(le)
    out = dict(auc=roc_auc_score(ye, pe), ap=average_precision_score(ye, pe), se90=sens_at_spec(ye, pe, 0.9),
               brier=float(np.mean((pe - ye) ** 2)), ece=ece(ye, pe))
    hit, has = loc_hit(P[f"{pre}_cell"], loc)
    pos = ye == 1
    out["loc"] = float(hit[pos & has].mean())
    thr = np.quantile(pe[~pos], 0.9)
    out["loc_se90"] = float(((pe >= thr) & hit)[pos].mean())
    cat_pred = P[f"{pre}_cat"].argmax(-1).reshape(-1)
    cat_true = y["cat"].reshape(-1)
    out["cat_f1"] = f1_score(cat_true, cat_pred, average="macro")
    out["cat_qwk"] = cohen_kappa_score(cat_true, cat_pred, weights="quadratic")
    neg = y["mal"].reshape(-1) == 0
    out["risk_auc"] = roc_auc_score(y["risk"].reshape(-1)[neg], P[f"{pre}_risk"].reshape(-1)[neg])
    out["risk_ap"] = average_precision_score(y["risk"].reshape(-1)[neg], P[f"{pre}_risk"].reshape(-1)[neg])
    return out


# ---------------------------------------------------------------- conformal decision layer
def mondrian_q(p_cal, y_cal, alpha):
    q = {}
    for c in (0, 1):
        s = 1 - (p_cal[y_cal == c] if c == 1 else 1 - p_cal[y_cal == c])
        n = len(s)
        k = int(np.ceil((n + 1) * (1 - alpha)))
        q[c] = np.sort(s)[min(k, n) - 1]
    return q


def marginal_q(p_cal, y_cal, alpha):
    s = np.where(y_cal == 1, 1 - p_cal, p_cal)
    n = len(s)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    v = np.sort(s)[min(k, n) - 1]
    return {0: v, 1: v}


def sets(p, q):
    return np.stack([(p <= q[0]), (1 - p <= q[1])], 1)   # include 0 if score(0)=p<=q0; include 1 if 1-p<=q1


def crc_threshold(p_cal, y_cal, alpha):
    """Conformal risk control of the false-negative rate: largest lambda with (sum L + 1)/(n+1) <= alpha."""
    pp = np.sort(p_cal[y_cal == 1])
    n = len(pp)
    best = 0.0
    for lam in np.concatenate([[0.0], pp]):
        risk = (np.sum(pp < lam) + 1) / (n + 1)
        if risk <= alpha:
            best = lam
    return best


def mahal_fit(E):
    mu = E.mean(0)
    C = np.cov(E.T) + 1e-3 * np.eye(E.shape[1])
    return mu, np.linalg.inv(C)


def mahal(E, mu, Ci):
    d = E - mu
    return np.einsum("ij,jk,ik->i", d, Ci, d)


def decision_layer(Pv, Pc, Pt, yv, yc, yt, alpha=0.05, delta=0.05, T=1.0, mode="proposed"):
    """Returns dict of referral / coverage / sensitivity statistics on the target set."""
    pc, pt = sig(Pc["mal"].max(1) / T), sig(Pt["mal"].max(1) / T)
    yc_e, yt_e = yc.max(1), yt.max(1)
    q = mondrian_q(pc, yc_e, alpha) if mode in ("proposed", "mondrian", "mondrian+msp") else marginal_q(pc, yc_e, alpha)
    S = sets(pt, q)
    size = S.sum(1)
    cover = S[np.arange(len(yt_e)), yt_e.astype(int)]
    refer = size != 1
    if mode == "proposed":
        mu, Ci = mahal_fit(Pv["emb"].mean(1))
        dc = mahal(Pc["emb"].mean(1), mu, Ci)
        tau = np.quantile(dc, 1 - delta)
        atyp = mahal(Pt["emb"].mean(1), mu, Ci) > tau
        refer = refer | atyp
    auto = ~refer
    pred1 = S[:, 1] & (size == 1)
    atyp_rate = float(atyp.mean()) if mode == "proposed" else np.nan
    res = dict(coverage=float(cover.mean()), cov_pos=float(cover[yt_e == 1].mean()), cov_neg=float(cover[yt_e == 0].mean()),
               size=float(size.mean()), refer=float(refer.mean()),
               auto_sens=float(pred1[auto & (yt_e == 1)].mean()) if (auto & (yt_e == 1)).any() else np.nan,
               auto_spec=float((~pred1)[auto & (yt_e == 0)].mean()),
               auto_fn_per1000=float(1000 * ((~pred1) & auto & (yt_e == 1)).sum() / len(yt_e)),
               ref_cancer_frac=float(yt_e[refer].mean()) if refer.any() else np.nan, atyp=atyp_rate)
    lam = crc_threshold(pc, yc_e, alpha * 2)
    res["crc_sens"] = float((pt[yt_e == 1] >= lam).mean())
    res["crc_spec"] = float((pt[yt_e == 0] < lam).mean())
    return res


# ---------------------------------------------------------------- statistics
def delong(y, s1, s2):
    """Two-sided DeLong test for correlated AUCs. Returns (auc1, auc2, z, p)."""
    y = np.asarray(y).astype(int)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    def comps(s):
        X, Y = s[pos], s[neg]
        V10 = np.array([(np.sum(x > Y) + 0.5 * np.sum(x == Y)) / len(Y) for x in X])
        V01 = np.array([(np.sum(X > yy) + 0.5 * np.sum(X == yy)) / len(X) for yy in Y])
        return V10.mean(), V10, V01
    a1, v10a, v01a = comps(s1)
    a2, v10b, v01b = comps(s2)
    S10 = np.cov(np.vstack([v10a, v10b]))
    S01 = np.cov(np.vstack([v01a, v01b]))
    S = S10 / len(pos) + S01 / len(neg)
    var = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    z = (a1 - a2) / np.sqrt(max(var, 1e-12))
    return a1, a2, z, 2 * (1 - stats.norm.cdf(abs(z)))


def boot_diff(y, s1, s2, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    d = []
    N = len(y)
    for _ in range(n):
        i = rng.integers(0, N, N)
        if y[i].min() == y[i].max():
            continue
        d.append(roc_auc_score(y[i], s1[i]) - roc_auc_score(y[i], s2[i]))
    return np.percentile(d, [2.5, 97.5])


def holm(pvals):
    p = np.asarray(pvals); o = np.argsort(p); m = len(p)
    adj = np.empty(m); run = 0
    for r, i in enumerate(o):
        run = max(run, (m - r) * p[i]); adj[i] = min(1, run)
    return adj
