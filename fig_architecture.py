"""Fig. 1: overall architecture of TRACE (grayscale, forward = solid, reverse-mode gradients = dashed)."""
from diagram import Diagram

d = Diagram(17.2, 6.8, (7.16, 2.95))
G1, G2, G3 = "0.93", "0.85", "0.97"
fs = 6.3
# inputs
d.box("cur", 0.1, 5.2, 1.85, 1.2, "Current exam $\\mathbf{x}^{(0)}$\n(LCC, LMLO,\nRCC, RMLO)", fs=fs)
d.box("pri", 0.1, 3.5, 1.85, 1.2, "Prior exams\n$\\mathbf{x}^{(1)}$, $\\mathbf{x}^{(2)}$ + mask $a_p$", fs=fs)
d.box("cfr", 0.1, 0.35, 1.85, 1.75, "Acquisition\ncounterfactuals\n(switched priors,\nfull re-rendering)\n[training only]", fs=fs, ls=(0, (3, 2)))
# encoding
d.box("enc", 2.45, 3.5, 1.45, 2.9, "Frozen\npretrained\nencoder $E$\n(shared,\nno gradient)", fc=G2, fs=fs)
d.box("ada", 4.35, 3.5, 1.3, 2.9, "Adapter\n$A_{\\phi}$", fc=G1, fs=fs)
# current-token path
d.box("tok", 6.15, 5.3, 4.45, 1.1, "Pooled current tokens (36 per view)\n+ view, side and position embeddings", fs=fs)
# correspondence path
d.box("cost", 6.15, 3.5, 1.35, 1.25, "Cost volume\n+ soft-argmax\n$\\mathbf{u}_{0\\rightarrow p}$", fs=fs, fc=G3)
d.box("cyc", 7.85, 3.5, 1.35, 1.25, "Smoothing +\ncycle check\n$c(\\mathbf{p})$", fs=fs, fc=G3)
d.box("gate", 9.55, 3.5, 1.05, 1.25, "Gated\nresidual\n$\\boldsymbol{\\Delta}^{(p)}$, $\\mathbf{r}^{(p)}$", fs=fs, fc=G3)
d.box("topk", 7.85, 2.2, 2.75, 0.85, "Saliency top-$K$ change tokens", fs=fs, fc=G3)
# reasoning and outputs
d.box("tf", 11.05, 2.2, 1.55, 4.2, "Cross-view\nrelational\ntransformer\n(ipsilateral +\nbilateral +\nchange\ntokens)", fc=G1, fs=fs)
d.box("heads", 13.05, 3.3, 1.75, 3.1, "Breast heads\n\nmalignancy\nassessment\ncategory\ninterval risk\n\n+ localisation", fs=fs)
d.box("conf", 15.25, 3.3, 1.85, 3.1, "Shift-aware\nconformal\nreferral\n\ntemperature,\nMondrian sets,\ntypicality gate", fc=G1, fs=fs)
d.box("out", 15.25, 0.35, 1.85, 1.3, "Auto-negative\nRecall\nRefer to reader", fs=fs)
# objectives (training)
d.box("lsup", 4.35, 0.35, 1.9, 0.95, "$\\mathcal{L}_{\\mathrm{sup}}$ (detection,\ncategory, risk, loc.)", fs=fs)
d.box("lcor", 6.6, 0.35, 1.9, 0.95, "$\\mathcal{L}_{\\mathrm{corr}}$ + $\\mathcal{L}_{\\mathrm{sal}}$\n(correspondence)", fs=fs)
d.box("lacc", 8.85, 0.35, 1.75, 0.95, "$\\mathcal{L}_{\\mathrm{acc}}$ (counter-\nfactual\nconsistency)", fs=fs)
d.box("linv", 10.95, 0.35, 1.65, 0.95, "$\\mathcal{L}_{\\mathrm{inv}}$ (invariant\ncontrastive)", fs=fs)
d.box("ltot", 13.05, 0.35, 1.75, 0.95, "Total objective\n$\\mathcal{L}$", fs=fs, bold=False, fc=G2)

# forward information flow (solid)
d.arrow(d.anchor("cur", "r"), (2.45, 5.8))
d.arrow(d.anchor("pri", "r"), (2.45, 4.1))
d.poly([d.anchor("cfr", "r", 0.8), (3.175, 0.35 + 1.75 * 0.8), d.anchor("enc", "b")])
d.arrow((3.9, 4.95), (4.35, 4.95))
d.arrow((5.65, 5.85), (6.15, 5.85))
# frozen descriptors (not the adapted features) feed the cost volume: routed below the adapter
d.poly([(3.7, 3.5), (3.7, 3.22), (6.5, 3.22), (6.5, 3.5)])
d.ax.text(5.25, 2.98, "frozen $\\hat{\\mathbf{f}}$", fontsize=5.8, ha="left", va="center", style="italic")
# adapted features z feed the residual (gate)
d.poly([(5.65, 5.02), (10.075, 5.02), (10.075, 4.75)])
d.arrow(d.anchor("cost", "r"), d.anchor("cyc", "l"))
d.arrow(d.anchor("cyc", "r"), d.anchor("gate", "l"))
d.arrow(d.anchor("gate", "b"), (10.075, 3.05))
d.arrow((10.6, 4.3), (11.05, 4.3))                                  # dense confidence-conditioned fusion
d.arrow((10.6, 2.625), (11.05, 2.625))
d.arrow((10.6, 5.85), (11.05, 5.85))
d.arrow((12.6, 4.85), (13.05, 4.85))
d.arrow((14.8, 4.85), (15.25, 4.85))
d.arrow(d.anchor("conf", "b"), d.anchor("out", "t"))
for k in ("lsup", "lcor", "lacc", "linv"):
    pass
d.arrow(d.anchor("linv", "r"), d.anchor("ltot", "l"))
for xg in (6.425, 8.675, 10.775):          # the objectives are summed
    d.ax.text(xg, 0.825, "+", ha="center", va="center", fontsize=8)

# reverse-mode gradients (dashed): one gradient bus from the total objective to every trainable module
yb = 1.72
d.bus([(13.3, 1.3), (13.3, yb), (5.0, yb)])
d.arrow((5.0, yb), (5.0, 3.5), dashed=True)                 # adapter
from matplotlib.patches import Rectangle
d.ax.add_patch(Rectangle((4.93, 3.12), 0.14, 0.2, fc="white", ec="none", zorder=5))
d.ax.plot([4.85, 5.15], [3.22, 3.22], color="black", lw=0.9, zorder=6)
d.arrow((6.825, yb), (6.825, 3.5), dashed=True)             # correspondence (cost volume / flow)
d.arrow((9.225, yb), (9.225, 2.2), dashed=True)             # change-token projection / saliency
d.arrow((11.825, yb), (11.825, 2.2), dashed=True)           # transformer
d.arrow((13.925, 1.3), (13.925, 3.3), dashed=True)          # heads
# legend (placed in free space, no overlap with boxes or paths)
d.ax.plot([0.15, 0.75], [3.05, 3.05], color="black", lw=0.9)
d.ax.annotate("", xy=(0.8, 3.05), xytext=(0.7, 3.05), arrowprops=dict(arrowstyle="-|>", lw=0.9, mutation_scale=7, color="black"))
d.ax.text(0.95, 3.05, "forward pass", fontsize=6.0, va="center")
d.ax.plot([0.15, 0.75], [2.6, 2.6], color="black", lw=0.9, ls=(0, (3.5, 2.2)))
d.ax.annotate("", xy=(0.8, 2.6), xytext=(0.7, 2.6), arrowprops=dict(arrowstyle="-|>", lw=0.9, mutation_scale=7, color="black"))
d.ax.text(0.95, 2.6, "reverse-mode gradient", fontsize=6.0, va="center")
if __name__ == "__main__":
    bad, loose = d.check(0.08)
    print("overlaps:", bad, "loose arrows:", loose, "text overflow:", d.text_fit())
    d.save("../fig/fig1_architecture.png", 600)
