"""Encode every view with the frozen self-supervised encoder and cache 12x12 feature grids (fp16)."""
import numpy as np, torch, sys
from common import Encoder, DATA

torch.set_num_threads(2)
enc = Encoder(64)
enc.load_state_dict(torch.load("../res/encoder_ssl.pt"))
enc.eval()


def run(name, src):
    X = np.load(f"{DATA}/{src}.npy", mmap_mode="r")
    N = X.shape[0]
    out = np.lib.format.open_memmap(f"{DATA}/{name}_feat.npy", "w+", np.float16, (N, 3, 4, 64, 12, 12))
    B = 32
    with torch.no_grad():
        for s in range(0, N, B):
            x = torch.from_numpy(np.asarray(X[s:s + B])).float().div(255).reshape(-1, 1, 96, 96)
            f = enc(x).reshape(-1, 3, 4, 64, 12, 12)
            out[s:s + B] = f.numpy().astype(np.float16)
            if s % 1600 == 0:
                print(name, s, flush=True)
    out.flush()


if __name__ == "__main__":
    for a in sys.argv[1:]:
        run(a, a)
