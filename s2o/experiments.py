"""Parameter sweeps behind Figs. 3-5 and the numbers quoted in the text.

Usage: python -m s2o.experiments [--seeds 30] [--sweeps load model ...] [--procs N]
"""
from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from multiprocessing import Pool

import pandas as pd

from .config import DEFAULT, MIN
from .policies import ABLATIONS, POLICIES, run

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

SWEEPS = {
    # name: (parameter, values)
    "load": ("rate_per_node_h", [0.5, 1, 2, 3, 4, 6, 8]),
    "model": ("ops_b_gop", [50, 100, 200, 400, 800, 1600, 3200]),
    "deadline": ("deadline_s", [m * MIN for m in (30, 60, 90, 180, 360, 720)]),
    "miss": ("p_miss", [0.0, 0.1, 0.2, 0.3, 0.4]),
    "leogap": ("leo_gap_min", [10, 20, 30, 60, 90, 120]),
    # contact loss in the energy-frugal regime (6 h deadline), with the ablation
    "robust": ("p_miss", [0.0, 0.1, 0.2, 0.3, 0.4]),
}
FIXED = {"robust": dict(deadline_s=360 * MIN)}


def _job(args):
    warnings.simplefilter("ignore", RuntimeWarning)  # quantiles over undelivered (inf) tasks
    sweep, param, value, seed, policy = args
    p = DEFAULT.with_(**FIXED.get(sweep, {})).with_(**{param: value})
    rows = run(p, seed, policy)
    df = pd.DataFrame(rows)
    return dict(sweep=sweep, param=param, value=value, seed=seed, policy=policy,
                n=len(df),
                mean_min=df.latency_s[df.delivered].mean() / 60,
                p50_min=df.latency_s.quantile(0.5) / 60,
                p95_min=df.latency_s.quantile(0.95) / 60,
                on_time=df.on_time.mean(), delivered=df.delivered.mean(),
                energy_j=df.energy_j.mean(), replans=df.replans.mean(),
                b_where=json.dumps(df.where_b.value_counts(normalize=True).round(4).to_dict()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--sweeps", nargs="*", default=list(SWEEPS))
    ap.add_argument("--procs", type=int, default=os.cpu_count())
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    for name in a.sweeps:
        param, values = SWEEPS[name]
        pols = POLICIES + (ABLATIONS if name == "robust" else ())
        jobs = [(name, param, v, s, pol) for v in values for s in range(a.seeds) for pol in pols]
        t0 = time.time()
        with Pool(a.procs) as pool:
            res = pool.map(_job, jobs, chunksize=1)
        pd.DataFrame(res).to_csv(os.path.join(OUT, f"sweep_{name}.csv"), index=False)
        print(f"{name}: {len(jobs)} runs in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
