"""Figures 3-5 of the paper from results/sweep_*.csv. Usage: python -m s2o.plots"""
from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .experiments import OUT  # noqa: E402
from .policies import LABELS  # noqa: E402

FIG = os.path.join(os.path.dirname(OUT), "figures")

# Categorical slots in fixed order (validated: adjacent CVD dE >= 9.1, normal-vision dE >= 19.6).
# The light hues sit below 3:1 on white, so every series also carries a distinct
# marker shape and the legend (relief rule).
ORDER = ["cacgr", "sensor", "statopt", "surface", "cloud"]
COLOR = dict(cacgr="#2a78d6", sensor="#eb6834", statopt="#1baf7a",
             surface="#eda100", cloud="#e87ba4")
MARK = dict(cacgr="o", sensor="s", statopt="D", surface="^", cloud="v")
# Ordinal ramp for tiers, dark = deep (validated ordinal: monotone L, adjacent dL >= 0.06).
TIERS = [("seabed", "Seabed node"), ("auv", "AUV"), ("buoy", "Buoy"),
         ("uav", "UAV"), ("sat", "LEO satellite")]
TIER_COLOR = ["#0d366b", "#1c5cab", "#2a78d6", "#5598e7", "#86b6ef"]

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
W1 = 3.45  # IEEE column width (in)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "FreeSans", "Liberation Sans", "DejaVu Sans"],
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
    "axes.axisbelow": True, "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def _load(name):
    return pd.read_csv(os.path.join(OUT, f"sweep_{name}.csv"))


def _ci(x):
    x = np.asarray(x, float)
    return 1.96 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0


def _save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"), dpi=300)
    plt.close(fig)


def fig_workload():
    df = _load("model")
    fig, ax = plt.subplots(figsize=(W1, 2.25))
    for pol in ORDER:
        d = df[df.policy == pol].groupby("value").on_time.agg(["mean", _ci]).reset_index()
        ax.errorbar(d.value, d["mean"], yerr=d["_ci"], color=COLOR[pol], marker=MARK[pol],
                    ms=4.5, lw=1.6, capsize=0, elinewidth=0.8, label=LABELS[pol],
                    markeredgecolor="white", markeredgewidth=0.6, zorder=3 if pol == "cacgr" else 2)
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted(df.value.unique()))
    ax.set_xticklabels([f"{int(v)}" for v in sorted(df.value.unique())])
    ax.minorticks_off()
    ax.set_xlabel("Stage-B inference workload per clip (GOP)")
    ax.set_ylabel("Results delivered by the deadline")
    ax.set_ylim(-0.03, 1.03)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    handles = [plt.Line2D([], [], color=COLOR[p], marker=MARK[p], ms=4.5, lw=1.6,
                          markeredgecolor="white", markeredgewidth=0.6) for p in ORDER]
    ax.legend(handles, [LABELS[p] for p in ORDER], loc="lower center",
              bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, handlelength=2.0,
              columnspacing=0.9, borderaxespad=0.2)
    _save(fig, "fig3_workload")


def fig_deadline():
    df = _load("deadline")
    fig, ax = plt.subplots(figsize=(W1, 2.3))
    for pol in ORDER:
        d = (df[df.policy == pol].groupby("value")
             .agg(e=("energy_j", "mean"), ot=("on_time", "mean")).reset_index())
        h = d.value / 3600.0
        ax.plot(h, d.e, color=COLOR[pol], lw=1.6, zorder=2, label="_nolegend_")
        for x, y, ot in zip(h, d.e, d.ot):
            filled = ot >= 0.95
            ax.plot([x], [y], marker=MARK[pol], ms=5, color=COLOR[pol],
                    markerfacecolor=COLOR[pol] if filled else "white",
                    markeredgecolor=COLOR[pol] if not filled else "white",
                    markeredgewidth=0.9 if not filled else 0.6, ls="none", zorder=3)
        ax.plot([], [], color=COLOR[pol], marker=MARK[pol], ms=4.5, lw=1.6,
                markeredgecolor="white", markeredgewidth=0.6, label=LABELS[pol])
    ax.set_xscale("log")
    ticks = sorted(df.value.unique() / 3600.0)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:g}" for t in ticks])
    ax.minorticks_off()
    ax.set_yscale("log")
    ax.set_ylim(1.5, 1.2e4)
    ax.set_xlabel("Application deadline (h)")
    ax.set_ylabel("Seabed energy per result (J)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False, ncol=3,
              handlelength=2.0, columnspacing=0.9, borderaxespad=0.2)
    ax.text(0.02, 0.03, "Filled marker: at least 95% of results on time; hollow: fewer",
            transform=ax.transAxes, fontsize=6.5, color=INK2, va="bottom")
    _save(fig, "fig4_deadline_energy")


def _shares(df):
    d = df[df.policy == "cacgr"]
    out = {}
    for v, g in d.groupby("value"):
        acc = {k: 0.0 for k, _ in TIERS}
        for s in g.b_where:
            for k, x in json.loads(s).items():
                if k in acc:
                    acc[k] += x / len(g)
        out[v] = acc
    return out


def fig_placement():
    sw = _shares(_load("model"))
    sd = _shares(_load("deadline"))
    kfmt = lambda v: f"{v / 1000:g}k" if v >= 1000 else f"{int(v)}"
    labels = [kfmt(v) for v in sw] + [f"{v / 3600:g}" for v in sd]
    data = list(sw.values()) + list(sd.values())
    gap = 1.0
    x = list(range(len(sw))) + [len(sw) + gap + i for i in range(len(sd))]
    fig, ax = plt.subplots(figsize=(W1, 2.35))
    bottom = np.zeros(len(data))
    for (k, name), c in zip(TIERS, TIER_COLOR):
        vals = np.array([d[k] for d in data]) * 100
        ax.bar(x, vals, bottom=bottom, width=0.7, color=c, label=name,
               edgecolor="white", linewidth=0.8)
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=6.3)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("Share of stage-B executions (%)")
    ax.grid(axis="x", visible=False)
    mid1 = (x[0] + x[len(sw) - 1]) / 2
    mid2 = (x[len(sw)] + x[-1]) / 2
    ax.text(mid1, -13, "Workload (GOP)", ha="center", va="top",
            fontsize=7.5, color=INK, transform=ax.transData, clip_on=False)
    ax.text(mid2, -13, "Deadline (h)", ha="center", va="top",
            fontsize=7.5, color=INK, transform=ax.transData, clip_on=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.17), ncol=5, frameon=False,
              handlelength=1.0, columnspacing=0.8, fontsize=6.8)
    _save(fig, "fig5_placement")


def main():
    fig_workload()
    fig_deadline()
    fig_placement()
    print("figures written to", FIG)


if __name__ == "__main__":
    main()
