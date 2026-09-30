"""Result figures (grayscale): ROC curves, stratified robustness, sensitivity, conformal behaviour."""
import numpy as np, json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
import evaluate_all as E
import metrics as M

plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif"], "mathtext.fontset": "stix",
                     "font.size": 7, "axes.linewidth": 0.6, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
                     "legend.frameon": False})
FIG = "../fig/"
MAIN = [("Proposed", "TRACE", "black", "-", 1.3), ("Siamese-Diff", "Siamese-Diff", "0.4", "-.", 0.9),
        ("Asym-Temporal", "Asym-Temporal", "0.6", (0, (5, 1, 1, 1, 1, 1)), 0.9), ("Prior-Fusion", "Prior-Fusion", "0.35", "--", 0.9),
        ("MV-Transformer", "MV-Transformer", "0.55", ":", 1.1), ("Classical-HGB", "Classical-HGB", "0.75", (0, (1, 2)), 0.9)]


def fig_roc():
    fig, axs = plt.subplots(1, 3, figsize=(7.1, 2.35))
    plt.subplots_adjust(left=0.06, right=0.99, top=0.9, bottom=0.34, wspace=0.28)
    titles = {"test": "(a) In-distribution test", "testswu": "(b) Priors from unseen chains", "ext": "(c) External (V4/V5)"}
    for ax, k in zip(axs, ("test", "testswu", "ext")):
        y = E.SETS[k][0]["mal"].max(1)
        for tag, lab, c, ls, lw in MAIN:
            if not E.load(tag):
                continue
            p = E.ensemble(tag, f"{k}_mal")
            f, t, _ = roc_curve(y, p)
            ax.plot(f, t, color=c, ls=ls, lw=lw, label=f"{lab} ({roc_auc_score(y, p):.3f})" if k == "test" else lab)
        ax.plot([0, 1], [0, 1], color="0.85", lw=0.5)
        ax.axvline(0.1, color="0.8", lw=0.5, ls="--")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.0)
        ax.set_xlabel("1 − specificity"); ax.set_title(titles[k], fontsize=7.5)
        if k == "test":
            ax.set_ylabel("Sensitivity")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, [x.split(" (")[0] for x in l], loc="lower center", ncol=6, fontsize=6.5, bbox_to_anchor=(0.5, 0.01),
               handlelength=2.6, columnspacing=1.4)
    fig.savefig(FIG + "fig4_roc.png", dpi=600)
    plt.close(fig)


STRATA = [("test_same_vendor", "Same-vendor\nprior (ID)"), ("test_switched_vendor", "Switched\nprior (ID)"),
          ("test_dens_cd", "Dense c/d\n(ID)"), ("ext_V4", "V4\n(external)"), ("ext_V5", "V5\n(external)"),
          ("ext_legacy_prior", "Cross-chain\nprior (ext.)"), ("ext_dens_cd", "Dense c/d\n(ext.)")]
BARS = [("MV-Transformer", "MV-Transformer", "white", ""), ("Prior-Fusion", "Prior-Fusion", "0.85", "////"),
        ("Asym-Temporal", "Asym-Temporal", "0.65", "...."), ("Siamese-Diff", "Siamese-Diff", "0.45", "\\\\\\\\"), ("Proposed", "TRACE", "0.1", "")]


def fig_strata():
    T = json.load(open("../res/summary.json"))
    fig, axs = plt.subplots(2, 1, figsize=(3.45, 3.2))
    plt.subplots_adjust(left=0.14, right=0.99, top=0.97, bottom=0.21, hspace=0.62)
    for ax, suf, yl in zip(axs, ("", "_risk"), ("Malignancy AUC", "Interval-risk AUC")):
        x = np.arange(len(STRATA))
        w = 0.16
        for b, (tag, lab, fc, hat) in enumerate(BARS):
            if tag not in T:
                continue
            m = [T[tag][f"strata.{s}{suf}"][0] for s, _ in STRATA]
            sd = [T[tag][f"strata.{s}{suf}"][1] for s, _ in STRATA]
            ax.bar(x + (b - 2) * w, m, w, yerr=sd, color=fc, edgecolor="black", lw=0.4, hatch=hat, label=lab,
                   error_kw=dict(lw=0.5, capsize=1.2))
        ax.set_xticks(x); ax.set_xticklabels([s[1] for s in STRATA], fontsize=5.8)
        ax.set_ylabel(yl)
        lo = 0.55 if suf else 0.7
        ax.set_ylim(lo, 0.97 if not suf else 0.9)
        ax.grid(axis="y", color="0.9", lw=0.4); ax.set_axisbelow(True)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=6.3, bbox_to_anchor=(0.55, 0.0), columnspacing=1.2)
    fig.savefig(FIG + "fig5_strata.png", dpi=600)
    plt.close(fig)


def fig_sensitivity():
    T = json.load(open("../res/summary.json"))
    def g(tag, key):
        return T[tag][key][0] if tag in T else np.nan
    panels = [("$\\lambda_{\\mathrm{acc}}$", ["lam_acc0.25", "Proposed", "lam_acc4"], [0.25, 1, 4], "log"),
              ("$K$ (change tokens / view)", ["K1", "Proposed", "K6"], [1, 3, 6], "linear"),
              ("$\\sigma_{s}$ (field smoothing)", ["sigma0", "Proposed"], [0, 1.5], "linear")]
    fig, axs = plt.subplots(1, 3, figsize=(3.45, 1.75))
    plt.subplots_adjust(left=0.13, right=0.99, top=0.95, bottom=0.42, wspace=0.45)
    for ax, (xl, tags, xs, sc) in zip(axs, panels):
        for key, lab, mk, c in (("test.auc", "ID AUC", "o", "black"), ("testsw.auc", "Switch AUC", "s", "0.45"),
                                ("ext.auc", "Ext. AUC", "^", "0.65")):
            ax.plot(xs, [g(t, key) for t in tags], marker=mk, ms=3, lw=0.8, color=c, label=lab, mfc="white" if c != "black" else "black")
        ax.set_xscale(sc); ax.set_xticks(xs); ax.set_xticklabels([str(v) for v in xs])
        ax.minorticks_off()
        ax.set_xlabel(xl, fontsize=6.5)
        ax.grid(color="0.9", lw=0.4)
        ax.tick_params(labelsize=6)
    axs[0].set_ylabel("AUC")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=6.3, bbox_to_anchor=(0.55, 0.0))
    fig.savefig(FIG + "fig6_sensitivity.png", dpi=600)
    plt.close(fig)


def fig_conformal():
    """Class-conditional coverage vs target and referral trade-off (seed 0..2 averaged)."""
    alphas = [0.02, 0.05, 0.1, 0.15, 0.2]
    runs = E.load("Proposed")
    fig, axs = plt.subplots(1, 2, figsize=(3.45, 1.9))
    plt.subplots_adjust(left=0.13, right=0.99, top=0.95, bottom=0.4, wspace=0.42)
    styles = {("test", "mondrian"): ("black", "-", "o", "ID, Mondrian"), ("ext", "mondrian"): ("0.5", "-", "s", "Ext., Mondrian"),
              ("ext", "marginal"): ("0.5", ":", "^", "Ext., marginal")}
    res = {}
    for (k, mode) in styles:
        cov1, ref = [], []
        for a in alphas:
            rr = []
            for s, P in runs.items():
                T = M.fit_temperature(P["val_mal"].max(1), E.YV["mal"].max(1))
                Pv = {q[4:]: v for q, v in P.items() if q.startswith("val_")}
                Pc = {q[4:]: v for q, v in P.items() if q.startswith("cal_")}
                Pt = {q[len(k) + 1:]: v for q, v in P.items() if q.startswith(k + "_")}
                rr.append(M.decision_layer(Pv, Pc, Pt, E.YV["mal"], E.YC["mal"], E.SETS[k][0]["mal"], a, 0.05, T,
                                           "proposed" if mode == "mondrian" else "marginal"))
            cov1.append(np.mean([r["cov_pos"] for r in rr])); ref.append(np.mean([r["refer"] for r in rr]))
        res[(k, mode)] = (cov1, ref)
        c, ls, mk, lab = styles[(k, mode)]
        axs[0].plot(1 - np.array(alphas), cov1, color=c, ls=ls, marker=mk, ms=3, lw=0.8, label=lab, mfc="white" if c != "black" else c)
        axs[1].plot(1 - np.array(alphas), ref, color=c, ls=ls, marker=mk, ms=3, lw=0.8, mfc="white" if c != "black" else c)
    axs[0].plot([0.78, 1], [0.78, 1], color="0.8", lw=0.5, ls="--")
    axs[0].set_xlabel("Target $1-\\alpha$", fontsize=6.5); axs[0].set_ylabel("Cancer coverage", fontsize=6.5)
    axs[1].set_xlabel("Target $1-\\alpha$", fontsize=6.5); axs[1].set_ylabel("Referral rate", fontsize=6.5)
    for ax in axs:
        ax.grid(color="0.9", lw=0.4); ax.tick_params(labelsize=6)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=6.2, bbox_to_anchor=(0.55, 0.0))
    fig.savefig(FIG + "fig7_conformal.png", dpi=600)
    plt.close(fig)
    json.dump({f"{k}|{m}": v for (k, m), v in res.items()}, open("../res/conformal_curve.json", "w"))


if __name__ == "__main__":
    import sys
    for f in sys.argv[1:] or ["roc", "strata", "sensitivity", "conformal"]:
        globals()["fig_" + f]()
