"""Parameter count, analytic multiply-accumulate count and single-thread CPU latency per examination."""
import torch, time, json
from models import build

torch.set_num_threads(1)
out = {}
for name in ["SV-MIL", "MV-Late", "MV-Transformer", "Siamese-Diff", "Asym-Temporal", "Prior-Fusion", "Proposed"]:
    m = build(name).eval()
    trainable = sum(p.numel() for n, p in m.named_parameters() if not n.startswith(("proj.", "disc.")) and "pos3" not in n or name == "Prior-Fusion")
    B = 16
    fc = torch.randn(B, 4, 64, 12, 12); fp = torch.randn(B, 2, 4, 64, 12, 12); av = torch.ones(B, 2, dtype=torch.bool)
    with torch.no_grad():
        for _ in range(2):
            m(fc, fp, av)
        t = time.time()
        for _ in range(5):
            m(fc, fp, av)
        lat = (time.time() - t) / 5 / B * 1000
    # MACs via torch.utils.flop_counter
    from torch.utils.flop_counter import FlopCounterMode
    with FlopCounterMode(display=False) as fcm:
        with torch.no_grad():
            m(fc[:1], fp[:1], av[:1])
    out[name] = dict(params=trainable, ms_per_exam=lat, gflops=fcm.get_total_flops() / 1e9)
    print(name, out[name], flush=True)
# frozen encoder cost per view (shared by all models)
from common import Encoder
enc = Encoder(64).eval()
x = torch.randn(12, 1, 96, 96)
with torch.no_grad():
    for _ in range(2):
        enc(x)
    t = time.time()
    for _ in range(5):
        enc(x)
from torch.utils.flop_counter import FlopCounterMode
with FlopCounterMode(display=False) as fcm:
    with torch.no_grad():
        enc(x)
out["encoder_12_views"] = dict(params=sum(p.numel() for p in enc.parameters()), ms_per_exam=(time.time() - t) / 5 * 1000,
                               gflops=fcm.get_total_flops() / 1e9)
print(out["encoder_12_views"])
json.dump(out, open("../res/bench.json", "w"), indent=1)
