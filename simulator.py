"""In-silico longitudinal, bilateral, four-view screening mammography simulator.

Each simulated woman owns a canonical (acquisition-free) tissue description:
power-law fibroglandular texture (spectral exponent ~3), a breast outline per view,
lesions placed in a shared 3-D breast frame so that CC and MLO projections agree on
nipple distance, and a lesion history across screening rounds.  A tissue map for a
given round is obtained by warping the canonical tissue with a positioning
deformation, and an image is produced by passing the tissue map through a
vendor-specific acquisition/processing chain.  Rendering the same tissue with two
chains yields an acquisition counterfactual pair.
"""
import numpy as np
from scipy import ndimage as ndi

H = W = 96
VIEWS = ("LCC", "LMLO", "RCC", "RMLO")

# acquisition / processing chains -------------------------------------------
VENDORS = {
    # sq: quantum noise, blur: MTF sigma (px), eq: peripheral equalisation,
    # slope/center: presentation LUT, gam: gamma, usm: unsharp amount,
    # bg: background level, grid: periodic detector pattern, rs: resample factor
    "V1": dict(sq=.015, blur=.6, eq=.30, slope=5.0, center=.45, gam=1.00, usm=.30, bg=.00, grid=.000, rs=1.0),
    "V2": dict(sq=.020, blur=.8, eq=.50, slope=7.0, center=.50, gam=0.80, usm=.60, bg=.03, grid=.000, rs=1.0),
    "V3": dict(sq=.012, blur=.5, eq=.20, slope=4.0, center=.40, gam=1.25, usm=.10, bg=.00, grid=.000, rs=1.0),
    # unseen at development time
    "V4": dict(sq=.022, blur=.7, eq=.70, slope=9.0, center=.56, gam=0.70, usm=1.0, bg=.06, grid=.018, rs=1.0),
    "V5": dict(sq=.035, blur=1.1, eq=.10, slope=3.4, center=.34, gam=1.45, usm=.00, bg=.00, grid=.000, rs=0.70),
}
DEV_VENDORS = ("V1", "V2", "V3")
EXT_VENDORS = ("V4", "V5")

_fy = np.fft.fftfreq(H)[:, None]
_fx = np.fft.fftfreq(W)[None, :]
_F2 = _fy ** 2 + _fx ** 2
_F2[0, 0] = 1.0
_YY, _XX = np.mgrid[0:H, 0:W].astype(np.float32)


def powerlaw(rng, beta=3.0):
    amp = _F2 ** (-beta / 4.0)
    amp[0, 0] = 0.0
    ph = rng.uniform(0, 2 * np.pi, (H, W))
    f = np.real(np.fft.ifft2(amp * np.exp(1j * ph)))
    return ((f - f.mean()) / (f.std() + 1e-8)).astype(np.float32)


def breast_geometry(rng):
    """Per-breast outline parameters for CC and MLO (chest wall on the left)."""
    a = rng.uniform(64, 84)
    return dict(a_cc=a, b_cc=rng.uniform(38, 45), a_mlo=a * rng.uniform(1.02, 1.10),
                b_mlo=rng.uniform(42, 47), pect_w=rng.uniform(18, 30), pect_h=rng.uniform(38, 60))


def outline(view, g):
    if view.endswith("CC"):
        a, b, cy = g["a_cc"], g["b_cc"], 48.0
    else:
        a, b, cy = g["a_mlo"], g["b_mlo"], 52.0
    r2 = (_XX / a) ** 2 + ((_YY - cy) / b) ** 2
    mask = (r2 <= 1.0).astype(np.float32)
    thick = np.sqrt(np.clip(1.0 - r2, 0, 1)).astype(np.float32)  # compressed-thickness proxy
    pect = np.zeros_like(mask)
    if view.endswith("MLO"):
        pect = ((_XX < g["pect_w"] * (1 - _YY / g["pect_h"])) & (_YY < g["pect_h"])).astype(np.float32) * mask
    return mask, thick, pect, a, b, cy


def lesion_xy(view, g, u, v, w):
    """Project a 3-D breast-frame location (u: chest-wall->nipple, v: lateral, w: vertical)."""
    if view.endswith("CC"):
        a, b, cy, lat = g["a_cc"], g["b_cc"], 48.0, v
    else:
        a, b, cy, lat = g["a_mlo"], g["b_mlo"], 52.0, np.clip(0.72 * w + 0.45 * v, -1, 1)
    x = u * a
    y = cy + lat * 0.78 * b * np.sqrt(max(0.0, 1 - u ** 2))
    return float(x), float(y)


def mass_profile(x0, y0, r, irregular, spic, blur, rng_state):
    rs = np.random.RandomState(rng_state)
    dx, dy = _XX - x0, _YY - y0
    d = np.sqrt(dx ** 2 + dy ** 2) + 1e-6
    th = np.arctan2(dy, dx)
    reff = r * (1 + irregular * sum(rs.uniform(.3, 1) * np.cos(k * th + rs.uniform(0, 6.28)) for k in (2, 3, 5)))
    m = 1.0 / (1.0 + np.exp(np.clip((d - reff) / blur, -30, 30)))
    if spic > 0:
        for _ in range(rs.randint(4, 8)):
            ang = rs.uniform(0, 2 * np.pi)
            L = r * rs.uniform(1.6, 2.6)
            proj = dx * np.cos(ang) + dy * np.sin(ang)
            perp = np.abs(-dx * np.sin(ang) + dy * np.cos(ang))
            s = ((proj > 0) & (proj < L)).astype(np.float32) * np.exp(-(perp ** 2) / 0.6) * (1 - proj / L)
            m = np.maximum(m, spic * s)
    return m.astype(np.float32)


def calc_profile(x0, y0, n, rng_state):
    rs = np.random.RandomState(rng_state)
    m = np.zeros((H, W), np.float32)
    for _ in range(n):
        cx, cy = x0 + rs.normal(0, 2.2), y0 + rs.normal(0, 2.2)
        m += np.exp(-((_XX - cx) ** 2 + (_YY - cy) ** 2) / (2 * rs.uniform(.35, .7) ** 2))
    return np.clip(m, 0, 1.2)


def random_warp(rng, strength=1.0):
    ang = np.deg2rad(rng.normal(0, 3.0) * strength)
    sc = 1 + rng.normal(0, .03) * strength
    tx, ty = rng.normal(0, 3.0) * strength, rng.normal(0, 3.0) * strength
    ex = ndi.gaussian_filter(rng.normal(0, 1, (H, W)), 10) * 60 * strength
    ey = ndi.gaussian_filter(rng.normal(0, 1, (H, W)), 10) * 60 * strength
    return dict(ang=ang, sc=sc, tx=tx, ty=ty, ex=ex.astype(np.float32), ey=ey.astype(np.float32))


def warp_coords(wp):
    """Inverse map: output pixel -> canonical coordinate."""
    c, s = np.cos(wp["ang"]), np.sin(wp["ang"])
    x0, y0 = _XX - wp["tx"], _YY - 48 - wp["ty"]
    xc = (c * x0 + s * y0) / wp["sc"] + wp["ex"]
    yc = (-s * x0 + c * y0) / wp["sc"] + 48 + wp["ey"]
    return xc, yc


def fwd_point(wp, x, y):
    """Approximate forward map of a canonical point (used for lesion ground truth)."""
    xc, yc = warp_coords(wp)
    d = (xc - x) ** 2 + (yc - y) ** 2
    i = np.unravel_index(np.argmin(d), d.shape)
    return float(i[1]), float(i[0])


def apply_warp(img, wp, order=1):
    xc, yc = warp_coords(wp)
    return ndi.map_coordinates(img, [yc, xc], order=order, mode="constant", cval=0.0).astype(np.float32)


def render(T, mask, thick, vendor, rng):
    """Acquisition + processing chain: tissue attenuation map -> 8-bit presentation image."""
    p = VENDORS[vendor]
    mu = 1.4
    S = np.exp(-mu * T * (0.35 + 0.65 * thick))          # transmitted signal
    S = S + p["sq"] * np.sqrt(np.clip(S, 1e-4, None)) * rng.normal(0, 1, S.shape)
    S = ndi.gaussian_filter(S, p["blur"])
    L = -np.log(np.clip(S, 1e-3, None))
    L = L + p["eq"] * (1 - thick) * mask * 0.6              # peripheral equalisation
    inside = mask > 0.5
    lo, hi = np.percentile(L[inside], 1), np.percentile(L[inside], 99.5)
    Ln = np.clip((L - lo) / (hi - lo + 1e-6), 0, 1)
    Ln = 1 / (1 + np.exp(-p["slope"] * (Ln - p["center"])))
    Ln = (Ln - Ln[inside].min()) / (Ln[inside].max() - Ln[inside].min() + 1e-6)
    Ln = np.clip(Ln, 0, 1) ** p["gam"]
    if p["usm"] > 0:
        Ln = Ln + p["usm"] * (Ln - ndi.gaussian_filter(Ln, 2.0))
    if p["grid"] > 0:
        Ln = Ln + p["grid"] * np.sin(_XX * 2.1) * mask
    if p["rs"] < 1:
        small = ndi.zoom(Ln, p["rs"], order=1)
        Ln = ndi.zoom(small, (H / small.shape[0], W / small.shape[1]), order=1)[:H, :W]
    img = np.where(mask > 0.5, Ln, p["bg"])
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
DENS_THR = {0: 1.25, 1: 0.55, 2: -0.05, 3: -0.75}   # BI-RADS a,b,c,d -> texture threshold


class Woman:
    """Canonical anatomy + lesion history for one simulated woman."""

    def __init__(self, rng, density, n_prior, p_cancer, p_seed, p_benign, p_newbenign, growth_scale=1.0):
        self.rng = rng
        self.density = density
        self.n_prior = n_prior
        self.geom = {s: breast_geometry(rng) for s in "LR"}
        shared = {v: powerlaw(rng) for v in ("CC", "MLO")}
        self.tex = {}
        for s in "LR":
            for v in ("CC", "MLO"):
                self.tex[s + v] = (0.7 * shared[v] + 0.71 * powerlaw(rng)) / 1.22
        self.fine = {k: powerlaw(rng, 2.2) for k in self.tex}
        self.lesions = []       # dicts with side,u,v,w,kind,radius per round, etc.
        self.labels = {}
        side_c = rng.choice(["L", "R"])
        if rng.random() < p_cancer:
            self._add_malignant(side_c, current=True, growth_scale=growth_scale)
        elif rng.random() < p_seed:
            self._add_malignant(side_c, current=False, growth_scale=growth_scale)
        if rng.random() < p_benign:
            self._add_benign(rng.choice(["L", "R"]), stable=True)
        if rng.random() < p_newbenign:
            self._add_benign(rng.choice(["L", "R"]), stable=False)

    def _loc(self):
        r = self.rng
        return r.uniform(.25, .82), r.uniform(-.8, .8), r.uniform(-.8, .8)

    def _add_malignant(self, side, current, growth_scale):
        r = self.rng
        u, v, w = self._loc()
        g = r.uniform(1.45, 1.95) * growth_scale     # radius growth factor per round
        kind = "calc" if r.random() < 0.22 else "mass"
        if current:
            r0 = r.uniform(2.6, 6.0)                 # radius at current round (px)
        else:
            r0 = r.uniform(1.4, 2.3)                 # sub-threshold seed now; visible next round
        radii = [r0 / g ** k for k in range(3)]      # current, prior-1, prior-2
        radii.append(r0 * g)                         # next round (future)
        self.lesions.append(dict(side=side, u=u, v=v, w=w, kind=kind, malignant=True,
                                 radii=radii, contrast=r.uniform(.36, .52), irregular=r.uniform(.18, .32),
                                 spic=r.uniform(.3, .6) if r.random() < .6 else 0.0, seed=int(r.integers(1e9)),
                                 visible_now=current))

    def _add_benign(self, side, stable):
        r = self.rng
        u, v, w = self._loc()
        r0 = r.uniform(2.6, 5.5)
        radii = [r0, r0, r0, r0] if stable else [r0, 0.0, 0.0, r0]
        self.lesions.append(dict(side=side, u=u, v=v, w=w, kind="mass", malignant=False, radii=radii,
                                 contrast=r.uniform(.26, .40), irregular=r.uniform(.0, .06), spic=0.0,
                                 seed=int(r.integers(1e9)), visible_now=True, stable=stable))

    # -------------------------------------------------------------------
    def tissue(self, side, view, k, involution=0.0):
        """Canonical tissue map of breast `side`, view `view` at round k (0=current, 1,2 priors, -1 next)."""
        g = self.geom[side]
        mask, thick, pect, a, b, cy = outline(side + view, g)
        rad = np.sqrt((_XX / a) ** 2 + ((_YY - cy) / (0.8 * b)) ** 2)
        prof = np.exp(-((rad - 0.45) ** 2) / 0.12)
        thr = DENS_THR[self.density] - 0.10 * k + involution
        gland = 1 / (1 + np.exp(-(self.tex[side + view] * 1.0 + 0.9 * prof - 0.45 - thr) / 0.55))
        T = mask * (0.28 + 0.40 * gland + 0.07 * self.fine[side + view]) + 0.30 * pect
        idx = {0: 0, 1: 1, 2: 2, -1: 3}[k]
        gts = []
        for les in self.lesions:
            if les["side"] != side:
                continue
            r = les["radii"][idx]
            x, y = lesion_xy(view, g, les["u"], les["v"], les["w"])
            if r >= 0.9:
                if les["kind"] == "mass":
                    m = mass_profile(x, y, r, les["irregular"], les["spic"], 0.9 if les["malignant"] else 0.45, les["seed"] + 7 * (view == "MLO"))
                    T = T + les["contrast"] * min(1.0, r / 2.2) * m * mask
                else:
                    n = int(np.clip(round(2.2 * r), 2, 16))
                    m = calc_profile(x, y, n, les["seed"] + 7 * (view == "MLO"))
                    T = T + 0.55 * m * mask
            gts.append(dict(x=x, y=y, r=r, malignant=les["malignant"], kind=les["kind"]))
        return T.astype(np.float32), mask, thick, gts
