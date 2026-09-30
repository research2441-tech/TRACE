"""Minimal grayscale block-diagram helper with automatic overlap and arrow-attachment checks."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif", "Times New Roman", "DejaVu Serif"], "mathtext.fontset": "stix"})


class Diagram:
    def __init__(self, w, h, figsize):
        self.fig = plt.figure(figsize=figsize)
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, w); self.ax.set_ylim(0, h); self.ax.axis("off")
        self.boxes, self.arrows, self.labels, self.texts = {}, [], [], {}
        self.buses = []

    def box(self, key, x, y, w, h, text, fc="white", lw=0.9, ls="-", fs=7, bold=False, round_=0.04):
        p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={round_}", fc=fc, ec="black", lw=lw, ls=ls)
        self.ax.add_patch(p)
        t = self.ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
                         fontweight="bold" if bold else "normal", linespacing=1.15)
        self.boxes[key] = (x, y, w, h)
        self.texts[key] = t

    def group(self, key, x, y, w, h, title, fs=7):
        p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.06", fc="none", ec="0.35", lw=0.8, ls=(0, (4, 2)))
        self.ax.add_patch(p)
        self.ax.text(x + 0.06, y + h - 0.05, title, ha="left", va="top", fontsize=fs, style="italic", color="0.2")
        self.labels.append((key, x, y, w, h))

    def anchor(self, key, side, frac=0.5):
        x, y, w, h = self.boxes[key]
        return {"l": (x, y + h * frac), "r": (x + w, y + h * frac), "t": (x + w * frac, y + h), "b": (x + w * frac, y)}[side]

    def arrow(self, a, b, dashed=False, lw=0.9, label=None, lab_off=(0, 0.08), fs=6.2, rad=0.0, ha="center"):
        style = (0, (3.5, 2.2)) if dashed else "-"
        ar = FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=7, lw=lw, color="black", ls=style,
                             connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0)
        self.ax.add_patch(ar)
        self.arrows.append((a, b, dashed))
        if label:
            mx, my = (a[0] + b[0]) / 2 + lab_off[0], (a[1] + b[1]) / 2 + lab_off[1]
            self.ax.text(mx, my, label, ha=ha, va="center", fontsize=fs,
                         bbox=dict(fc="white", ec="none", pad=0.4))

    def poly(self, pts, dashed=False, lw=0.9):
        """Orthogonal multi-segment arrow through the given points (head on the last segment)."""
        style = (0, (3.5, 2.2)) if dashed else "-"
        for p, q in zip(pts[:-2], pts[1:-1]):
            self.ax.plot([p[0], q[0]], [p[1], q[1]], color="black", lw=lw, ls=style, solid_capstyle="butt")
        self.arrow(pts[-2], pts[-1], dashed=dashed, lw=lw)
        self.arrows[-1] = (pts[0], pts[-1], dashed)

    def bus(self, pts, dashed=True, lw=0.9):
        """Plain (arrow-less) line segment path; arrows may start anywhere on it."""
        style = (0, (3.5, 2.2)) if dashed else "-"
        xs, ys = zip(*pts)
        self.ax.plot(xs, ys, color="black", lw=lw, ls=style)
        self.buses.append(pts)

    def on_bus(self, p):
        for pts in self.buses:
            for a, b in zip(pts[:-1], pts[1:]):
                if min(a[0], b[0]) - 1e-6 <= p[0] <= max(a[0], b[0]) + 1e-6 and min(a[1], b[1]) - 1e-6 <= p[1] <= max(a[1], b[1]) + 1e-6:
                    return True
        return False

    def text_fit(self, pad=0.04):
        """Return boxes whose label extends beyond the box outline."""
        self.fig.canvas.draw()
        r = self.fig.canvas.get_renderer()
        inv = self.ax.transData.inverted()
        bad = []
        for k, t in self.texts.items():
            bb = t.get_window_extent(r)
            (x0, y0), (x1, y1) = inv.transform((bb.x0, bb.y0)), inv.transform((bb.x1, bb.y1))
            x, y, w, h = self.boxes[k]
            if x0 < x + pad or x1 > x + w - pad or y0 < y + pad or y1 > y + h - pad:
                bad.append(k)
        return bad

    def check(self, margin=0.05):
        """Assert that no two boxes overlap (including a margin) and every arrow end touches a box edge."""
        items = list(self.boxes.items())
        bad = []
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                (k1, (x1, y1, w1, h1)), (k2, (x2, y2, w2, h2)) = items[i], items[j]
                if x1 < x2 + w2 + margin and x2 < x1 + w1 + margin and y1 < y2 + h2 + margin and y2 < y1 + h1 + margin:
                    bad.append((k1, k2))
        def on_edge(p):
            for k, (x, y, w, h) in self.boxes.items():
                if (abs(p[0] - x) < 1e-6 or abs(p[0] - x - w) < 1e-6) and y - 1e-6 <= p[1] <= y + h + 1e-6:
                    return k
                if (abs(p[1] - y) < 1e-6 or abs(p[1] - y - h) < 1e-6) and x - 1e-6 <= p[0] <= x + w + 1e-6:
                    return k
            return None
        loose = [(a, b) for a, b, _ in self.arrows if (on_edge(a) is None and not self.on_bus(a)) or on_edge(b) is None]
        return bad, loose

    def save(self, path, dpi=600):
        self.fig.savefig(path, dpi=dpi)
        plt.close(self.fig)
