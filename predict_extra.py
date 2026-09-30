"""Add predictions on the unseen-chain-prior test set (testswu) to already finished runs."""
import glob, re, numpy as np, torch
from common import splits, load_meta, DATA, RES
from models import build
import train as TRN

torch.set_num_threads(1)
sp = splits(); meta = load_meta("dev")
y = TRN.labels(meta, np.load(f"{DATA}/dev_loc.npy"))
feat = np.load(f"{DATA}/dev_img_feat.npy", mmap_mode="r")
SRC = [("devu", "testswu"), ("devw20", "testw20"), ("devw30", "testw30")]
NAMES = {}
for f in glob.glob(f"{RES}/log_*_s*.json"):
    import json
    d = json.load(open(f)); NAMES[f.split("log_")[1].rsplit("_s", 1)[0]] = d["name"]
for f in sorted(glob.glob(f"{RES}/model_*_s*.pt")):
    tag, seed = re.findall(r"model_(.+)_s(\d+)\.pt", f)[0]
    pf = f"{RES}/pred_{tag}_s{seed}.npz"
    P = dict(np.load(pf))
    todo = [(s, key) for s, key in SRC if f"{key}_mal" not in P]
    if not todo:
        continue
    m = build(NAMES[tag]); m.load_state_dict(torch.load(f), strict=False)
    for s, key in todo:
        for k, v in TRN.predict(m, feat, sp["test"], y, TRN.unseen_prior_source(s), "switchu").items():
            P[f"{key}_{k}"] = v
    np.savez_compressed(pf, **P)
    print("added", tag, seed, flush=True)
