"""Run a list of (config, seed, epochs) jobs with N parallel single-thread workers, skipping finished ones."""
import subprocess, sys, os, time, json
from concurrent.futures import ThreadPoolExecutor

jobs = [l.strip().rsplit(" ", 2) for l in open(sys.argv[1]) if l.strip() and not l.startswith("#")]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 2


def tag(name):
    return name.replace("/", "").replace(" ", "_").replace("+", "p").replace("=", "")


def run(j):
    name, seed, ep = j
    out = f"../res/pred_{tag(name)}_s{seed}.npz"
    if os.path.exists(out):
        return
    t = time.time()
    with open(f"../res/run_{tag(name)}_s{seed}.log", "w") as f:
        subprocess.run(["python3", "train.py", name, seed, ep], stdout=f, stderr=subprocess.STDOUT)
    print(name, seed, "finished", round(time.time() - t), flush=True)


with ThreadPoolExecutor(N) as ex:
    list(ex.map(run, jobs))
