"""Prints every number quoted in the paper from results/sweep_*.csv.

Usage: python -m s2o.summary  (also written to results/summary.txt)
"""
from __future__ import annotations

import io
import json
import os

import pandas as pd

from .experiments import OUT


def _g(name):
    df = pd.read_csv(os.path.join(OUT, f"sweep_{name}.csv"))
    return df.groupby(["value", "policy"]).agg(
        on_time=("on_time", "mean"), energy_j=("energy_j", "mean"),
        p50_min=("p50_min", "mean"), p95_min=("p95_min", "mean")).reset_index()


def _v(g, value, policy, col):
    return float(g[(g.value == value) & (g.policy == policy)][col].iloc[0])


def main():
    buf = io.StringIO()
    w = lambda *a: print(*a, file=buf)
    m = _g("model")
    w("== Workload sweep (90-min deadline, 2 clips/h/node)")
    for v in sorted(m.value.unique()):
        w(f"  {int(v):5d} GOP  on-time  CA-CGR {_v(m, v, 'cacgr', 'on_time'):.3f}  on-sensor {_v(m, v, 'sensor', 'on_time'):.3f}"
          f"  statistical {_v(m, v, 'statopt', 'on_time'):.3f}  surface MEC {_v(m, v, 'surface', 'on_time'):.3f}"
          f"  cloud {_v(m, v, 'cloud', 'on_time'):.3f}")
    r = _v(m, 1600, "cacgr", "on_time") / _v(m, 1600, "sensor", "on_time")
    w(f"  1600 GOP ratio CA-CGR/on-sensor = {r:.2f}; p95 {_v(m, 1600, 'cacgr', 'p95_min'):.0f} vs {_v(m, 1600, 'sensor', 'p95_min'):.0f} min;"
      f" energy {_v(m, 1600, 'cacgr', 'energy_j'):.0f} vs {_v(m, 1600, 'sensor', 'energy_j'):.0f} J")
    w(f"  cloud-centric median delay {_v(m, 400, 'cloud', 'p50_min') / 60:.1f} h")
    ld = _g("load")
    w("== Load sweep (400 GOP)")
    w(f"  8 clips/h: CA-CGR {_v(ld, 8, 'cacgr', 'on_time'):.3f}  on-sensor {_v(ld, 8, 'sensor', 'on_time'):.3f}")
    d = _g("deadline")
    w("== Deadline sweep (400 GOP): seabed energy per result (J) and on-time ratio")
    for v in sorted(d.value.unique()):
        w(f"  {v / 3600:4.1f} h  CA-CGR {_v(d, v, 'cacgr', 'energy_j'):7.1f} J ({_v(d, v, 'cacgr', 'on_time'):.3f})"
          f"  on-sensor {_v(d, v, 'sensor', 'energy_j'):6.1f} J ({_v(d, v, 'sensor', 'on_time'):.3f})"
          f"  cloud {_v(d, v, 'cloud', 'energy_j'):4.1f} J ({_v(d, v, 'cloud', 'on_time'):.3f})")
    e12, es = _v(d, 43200, "cacgr", "energy_j"), _v(d, 43200, "sensor", "energy_j")
    w(f"  12 h: reduction vs on-sensor {100 * (1 - e12 / es):.1f} percent; 48 clips/day: {48 * es / 1e3:.1f} kJ -> {48 * e12 / 1e3:.2f} kJ")
    rb = _g("robust")
    w("== Contact loss with a 6-h deadline")
    for v in sorted(rb.value.unique()):
        w(f"  q={v:.1f}  CA-CGR {_v(rb, v, 'cacgr', 'on_time'):.3f} / {_v(rb, v, 'cacgr', 'energy_j'):5.1f} J"
          f"  no check {_v(rb, v, 'cacgr_nc', 'on_time'):.3f} / {_v(rb, v, 'cacgr_nc', 'energy_j'):5.1f} J"
          f"  on-sensor {_v(rb, v, 'sensor', 'on_time'):.3f} / {_v(rb, v, 'sensor', 'energy_j'):5.1f} J")
    ms = _g("miss")
    w("== Contact loss with the 90-min deadline")
    for v in sorted(ms.value.unique()):
        w(f"  q={v:.1f}  CA-CGR {_v(ms, v, 'cacgr', 'on_time'):.3f}  on-sensor {_v(ms, v, 'sensor', 'on_time'):.3f}")
    lg = _g("leogap")
    w("== LEO revisit gap (90-min deadline)")
    for v in sorted(lg.value.unique()):
        w(f"  {int(v):3d} min  CA-CGR {_v(lg, v, 'cacgr', 'on_time'):.3f}  on-sensor {_v(lg, v, 'sensor', 'on_time'):.3f}")
    text = buf.getvalue()
    print(text)
    with open(os.path.join(OUT, "summary.txt"), "w") as f:
        f.write(text)


if __name__ == "__main__":
    main()
