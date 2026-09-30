"""Fig. 2: one simulated breast rendered through the five acquisition chains (top) and across
screening rounds under a fixed chain (bottom)."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from simulator import Woman, render, random_warp, apply_warp, fwd_point

plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif"], "mathtext.fontset": "stix", "font.size": 7})
for seed in range(40, 400):
    rng = np.random.default_rng(seed)
    w = Woman(rng, 1, 2, 1.0, 0.0, 0.0, 0.0)
    les = [l for l in w.lesions if l["malignant"]]
    if les and les[0]["kind"] == "mass" and les[0]["radii"][0] > 4.2 and les[0]["radii"][2] > 1.2:
        break
side = les[0]["side"]
fig, ax = plt.subplots(2, 5, figsize=(3.5, 1.72))
plt.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.02, wspace=0.05, hspace=0.32)
T, mask, thick, gts = w.tissue(side, "CC", 0)
wp = random_warp(np.random.default_rng(1), 0.6)
Tw, Mw, Kw = apply_warp(T, wp), (apply_warp(mask, wp, 0) > .5).astype(np.float32), apply_warp(thick, wp)
g = [x for x in gts if x["malignant"]][0]
lx, ly = fwd_point(wp, g["x"], g["y"])
for j, v in enumerate(["V1", "V2", "V3", "V4", "V5"]):
    im = render(Tw, Mw, Kw, v, np.random.default_rng(5))
    ax[0, j].imshow(im, cmap="gray", vmin=0, vmax=255)
    ax[0, j].set_title(v + (" (unseen)" if v in ("V4", "V5") else ""), fontsize=6.3, pad=2)
    ax[0, j].add_patch(plt.Circle((lx, ly), g["r"] + 5, fill=False, color="white", lw=0.6, ls="--"))
labels = ["$t-2$ (V1)", "$t-1$ (V1)", "$t$ (V1)", "$t-1$ (V4)", "$t+1$ (V1)"]
rounds = [2, 1, 0, 1, -1]
vend = ["V1", "V1", "V1", "V4", "V1"]
for j, (k, v) in enumerate(zip(rounds, vend)):
    T, mask, thick, gts = w.tissue(side, "CC", k)
    wp = random_warp(np.random.default_rng(10 + (k % 5)), 0.6)
    Tw, Mw, Kw = apply_warp(T, wp), (apply_warp(mask, wp, 0) > .5).astype(np.float32), apply_warp(thick, wp)
    im = render(Tw, Mw, Kw, v, np.random.default_rng(6))
    ax[1, j].imshow(im, cmap="gray", vmin=0, vmax=255)
    ax[1, j].set_title(labels[j], fontsize=6.3, pad=2)
    g = [x for x in gts if x["malignant"]][0]
    x, y = fwd_point(wp, g["x"], g["y"])
    ax[1, j].add_patch(plt.Circle((x, y), max(g["r"], 1.5) + 5, fill=False, color="white", lw=0.6, ls="--"))
for a in ax.flat:
    a.set_xticks([]); a.set_yticks([])
    for s in a.spines.values():
        s.set_linewidth(0.4)
fig.savefig("../fig/fig2_cohort.png", dpi=600)
print("seed", seed, "radii", les[0]["radii"])
