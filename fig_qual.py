"""Fig. 3: correspondence-gated change maps of TRACE for a growing cancer and a stable benign mass,
with the native prior and with a vendor-switched counterfactual prior."""
import numpy as np, torch, json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch.nn.functional as F
from common import splits, load_meta, DATA
from models import build

plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif"], "mathtext.fontset": "stix", "font.size": 7})
torch.set_num_threads(1)
sp = splits(); meta = load_meta("dev")
img = np.load(f"{DATA}/dev_img.npy", mmap_mode="r"); cfimg = np.load(f"{DATA}/dev_cf.npy", mmap_mode="r")
feat = np.load(f"{DATA}/dev_img_feat.npy", mmap_mode="r"); cf = np.load(f"{DATA}/dev_cf_feat.npy", mmap_mode="r")
loc = np.load(f"{DATA}/dev_loc.npy")
m = build("Proposed"); m.load_state_dict(torch.load("../res/model_Proposed_s0.pt"), strict=False); m.eval()


def maps(i, switched):
    x = torch.from_numpy(np.asarray(feat[i:i + 1])).float()
    if switched:
        x = torch.cat([x[:, :1], torch.from_numpy(np.asarray(cf[i:i + 1, 1:])).float()], 1)
    av = torch.tensor([[meta[i]["n_prior"] >= 1, meta[i]["n_prior"] >= 2]])
    with torch.no_grad():
        o = m(x[:, 0], x[:, 1:], av)
    D = o["aux"]["D"][0, 0].pow(2).sum(1).sqrt()      # prior 1, [4,12,12]
    c = o["aux"]["c"][0, 0, :, 0]
    up = lambda a: F.interpolate(a[None, None], size=(96, 96), mode="bilinear", align_corners=False)[0, 0].numpy()
    return [up(D[j]) for j in range(4)], [up(c[j]) for j in range(4)], torch.sigmoid(o["mal"][0]).numpy()


def pick():
    cancer, benign = None, None
    for i in sp["test"]:
        mm = meta[i]
        if mm["n_prior"] < 1 or mm["vendors"][1] != mm["vendors"][0]:
            continue
        for s in "LR":
            L = mm["lab"][s]
            if cancer is None and L["cancer"] and not L["calc"] and not L["stable_benign"]:
                j = 0 if s == "L" else 2
                if loc[i, j, 2] > 4.0:
                    cancer = (i, j)
            if benign is None and L["stable_benign"] and not L["cancer"] and L["cat"] == 1:
                benign = (i, 0 if s == "L" else 2)
        if cancer and benign:
            return cancer, benign


cancer, benign = pick()
fig, ax = plt.subplots(2, 5, figsize=(3.5, 1.82))
plt.subplots_adjust(left=0.06, right=0.995, top=0.9, bottom=0.02, wspace=0.05, hspace=0.12)
cols = ["Current $t$", "Prior $t-1$", "$\\|\\boldsymbol{\\Delta}\\|$ native", "Confidence $c$", "$\\|\\boldsymbol{\\Delta}\\|$ switched"]
vmax = None
rowlab = ["Growing\ncancer", "Stable\nbenign"]
info = {}
for r, (i, j) in enumerate([cancer, benign]):
    Dn, cn, pn = maps(i, False)
    Ds, cs, ps = maps(i, True)
    vm = max(Dn[j].max(), Ds[j].max()) if r == 0 else vmax
    vmax = vm
    ax[r, 0].imshow(img[i, 0, j], cmap="gray", vmin=0, vmax=255)
    ax[r, 1].imshow(img[i, 1, j], cmap="gray", vmin=0, vmax=255)
    ax[r, 2].imshow(Dn[j], cmap="gray", vmin=0, vmax=vm)
    ax[r, 3].imshow(cn[j], cmap="gray", vmin=0, vmax=1)
    ax[r, 4].imshow(Ds[j], cmap="gray", vmin=0, vmax=vm)
    ax[r, 0].set_ylabel(rowlab[r], fontsize=6.3, labelpad=2)
    info[rowlab[r].replace("\n", " ")] = dict(exam=int(i), view=int(j), p_native=pn.tolist(), p_switched=ps.tolist(),
                                              vendor=meta[i]["vendors"], vcf=meta[i]["vcf"])
    if r == 0:
        for k in (0, 2, 4):
            ax[r, k].add_patch(plt.Circle((loc[i, j, 0], loc[i, j, 1]), loc[i, j, 2] + 6, fill=False, color="white", lw=0.6, ls="--"))
for k, t in enumerate(cols):
    ax[0, k].set_title(t, fontsize=6.2, pad=2)
for a in ax.flat:
    a.set_xticks([]); a.set_yticks([])
    for s in a.spines.values():
        s.set_linewidth(0.4)
fig.savefig("../fig/fig3_change_maps.png", dpi=600)
json.dump(info, open("../res/qual_info.json", "w"), indent=1)
print(info)
