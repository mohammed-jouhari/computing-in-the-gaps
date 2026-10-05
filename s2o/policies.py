"""Placement policies compared in the paper and the per-task simulation loop."""
from __future__ import annotations

import itertools
import math

import numpy as np

from .config import HOUR, MIN, Params
from .engine import INF, Outcome, execute, plan
from .scenario import build

POLICIES = ("cloud", "sensor", "surface", "statopt", "cacgr")
ABLATIONS = ("cacgr_nc",)  # CA-CGR without the risk (on-time probability) check
LABELS = {
    "cloud": "Cloud-centric",
    "sensor": "On-sensor",
    "surface": "Surface MEC",
    "statopt": "Statistical placement",
    "cacgr": "CA-CGR (proposed)",
    "cacgr_nc": "CA-CGR without risk check",
}


# --------------------------------------------------------------------------
# Statistical (contact-agnostic) placement: expected delay from long-run means
# --------------------------------------------------------------------------
def statopt_placement(p: Params, info: dict, src_idx: int):
    """Pick (tier_A, tier_B) that minimises the expected delay computed from mean
    waiting times and nominal rates, ignoring the actual contact plan and queues."""
    sizes = (p.raw_bits, p.raw_bits * p.feat_ratio, p.result_bits)
    gops = dict(seabed=p.gops_seabed, auv=p.gops_auv, buoy=p.gops_buoy,
                uav=p.gops_uav, sat=p.gops_sat, cloud=p.gops_cloud)
    r_ac = info["rates_bps"][src_idx]
    d_ac = info["slant_km"][src_idx] / 1.5
    tour = info["tour_s"]

    def hop(a, b, bits):
        if (a, b) == ("seabed", "buoy"):
            return d_ac + (bits / r_ac if r_ac > 0 else INF)
        if (a, b) == ("seabed", "auv"):
            return p.auv_period_h * HOUR / 2 + bits / p.optical_bps
        if (a, b) == ("auv", "buoy"):
            via_ac = d_ac + (bits / r_ac if r_ac > 0 else INF)
            via_dock = tour / 2 + bits / p.dock_bps
            return min(via_ac, via_dock)
        if (a, b) == ("buoy", "sat"):
            return p.leo_gap_min * MIN / 2 + bits / p.leo_up_bps
        if (a, b) == ("sat", "cloud"):
            return np.mean(p.leo_dl_delay_min) * MIN + bits / p.leo_down_bps
        if (a, b) == ("buoy", "uav"):
            return p.uav_period_h * HOUR / 2 + bits / p.uav_bps
        if (a, b) == ("uav", "cloud"):
            return p.uav_return_min * MIN + bits / p.shore_bps
        raise KeyError((a, b))

    chains = [["seabed", lo, "buoy", up, "cloud"] for lo in ("auv", None) for up in ("sat", "uav")]
    chains = [[x for x in c if x is not None] for c in chains]
    best = (INF, None)
    for ch in chains:
        for ia, ib in itertools.combinations_with_replacement(range(len(ch)), 2):
            d = p.ops_a_gop / gops[ch[ia]] + p.ops_b_gop / gops[ch[ib]]
            for h in range(len(ch) - 1):
                stage = 0 if h < ia else (1 if h < ib else 2)
                d += hop(ch[h], ch[h + 1], sizes[stage])
            if d < best[0]:
                best = (d, (ch[ia], ch[ib]))
    return best[1]


# --------------------------------------------------------------------------
# Action-set restrictions
# --------------------------------------------------------------------------
def _cpu_rule(policy, src, placement=None):
    if policy == "cloud":
        return lambda node, s: node.tier == "cloud"
    if policy == "sensor":
        return lambda node, s: node.name == src
    if policy == "surface":
        return lambda node, s: (s == 0 and node.name == src) or (s == 1 and node.tier == "buoy")
    if policy == "statopt":
        ta, tb = placement

        def ok(node, s):
            tier = ta if s == 0 else tb
            return node.name == src if tier == "seabed" else node.tier == tier
        return ok
    if policy in ("cacgr", "cacgr_nc"):
        return lambda node, s: node.tier != "seabed" or node.name == src
    raise ValueError(policy)


def _transport_ok(link, s):
    """Global transport rule: raw clips (stage 0) never use the acoustic medium."""
    return not (s == 0 and link.kind in ("ac", "auv_ac"))


_any_link = _transport_ok


def _seabed_profiles(src, stage):
    """CA-CGR seabed profiles: how far to process locally and which first link to use."""
    for s_dep in range(stage, 3):
        for first in ("ac", "opt"):
            if s_dep == 0 and first == "ac":
                continue
            cpu = (lambda s_dep: lambda node, s: (node.name == src and s < s_dep)
                   or node.tier != "seabed")(s_dep)
            lnk = (lambda s_dep, first: lambda link, s: _transport_ok(link, s) and (
                link.src != src or (s == s_dep and link.kind == first)))(s_dep, first)
            yield s_dep, first, cpu, lnk


def _profile_energy(p, net, src, stage, s_dep, first):
    e = sum(net.ops[s] / p.gops_seabed for s in range(stage, s_dep)) * p.p_cpu_seabed
    node = net.nodes[src]
    link = next(l for l in node.out if l.kind == first)
    rate = link.windows[0].rate if first == "ac" else p.optical_bps
    p_tx = p.p_tx_acoustic if first == "ac" else p.p_tx_optical
    return e + (net.sizes[s_dep] / rate * p_tx if rate > 0 else INF)


# Contacts that can be lost independently: AUV visits, UAV sorties, LEO passes and
# ground-station windows. The AUV acoustic relay shares the fate of its visit, and
# the dock and shore contacts follow the vehicle that is already on its way.
INTERMITTENT = ("opt", "uav", "up", "down")


def _on_time_prob(net, task, acts, deadline_abs, cpu_rule, p_loss):
    """First-order estimate of the probability that a plan delivers on time when
    each intermittent contact is lost independently with probability p_loss.

    The plan succeeds if no contact is lost, or if exactly one is lost and a
    re-plan started when the loss is noticed still meets the deadline (N-1 check).
    """
    hops = [a for a in acts if a[0] == "tx" and a[1].kind in INTERMITTENT]
    k = len(hops)
    prob = (1 - p_loss) ** k
    for kind, link, s, w, st in hops:
        saved, w.failed = w.failed, True
        arr, _ = plan(net, link.src, s, st, st, cpu_rule, _any_link)
        w.failed = saved
        if arr <= deadline_abs:
            prob += p_loss * (1 - p_loss) ** (k - 1)
    return prob


def _select(net, task, cands, target, deadline_abs, cpu_rule, p, risk):
    """CA-CGR seabed decision.

    Among plans that arrive before margin * deadline, keep the one with the lowest
    seabed energy; with contact loss, only plans whose estimated on-time
    probability reaches the target qualify (otherwise the most reliable one).
    If no plan meets the margin, take the earliest one.
    """
    feas = [c for c in cands if c[0] <= target]
    if not feas:
        arr, _, acts = min(cands, key=lambda c: (c[0], c[1]))
        return arr, acts
    if not risk:
        arr, _, acts = min(feas, key=lambda c: (c[1], c[0]))
        return arr, acts
    scored = [(c, _on_time_prob(net, task, c[2], deadline_abs, cpu_rule, p.p_miss)) for c in feas]
    good = [c for c, pr in scored if pr >= p.reliability]
    if good:
        arr, _, acts = min(good, key=lambda c: (c[1], c[0]))
    else:
        arr, _, acts = min(scored, key=lambda x: (-round(x[1], 6), x[0][1], x[0][0]))[0]
    return arr, acts


def run(p: Params, seed: int, policy: str):
    net, tasks, info = build(p, seed)
    power = {"cpu": p.p_cpu_seabed, "ac": p.p_tx_acoustic, "opt": p.p_tx_optical}
    end = (p.gen_hours + p.tail_hours) * HOUR
    src_idx = {f"sb{i}": i for i in range(p.n_nodes)}
    place = {}
    if policy == "statopt":
        place = {nm: statopt_placement(p, info, i) for nm, i in src_idx.items()}
    rows = []
    for task in tasks:
        out = Outcome()
        node, stage, t = task.src, 0, task.t_gen
        cpu_rule = _cpu_rule(policy, task.src, place.get(task.src))
        deadline_abs = task.t_gen + p.deadline_s
        guard = 0
        while t < end and guard < 50:
            guard += 1
            now = t
            if policy in ("cacgr", "cacgr_nc") and node == task.src:
                cands = []
                for s_dep, first, cpu, lnk in _seabed_profiles(task.src, stage):
                    arr, acts = plan(net, node, stage, t, now, cpu, lnk)
                    if acts is None:
                        continue
                    cands.append((arr, _profile_energy(p, net, task.src, stage, s_dep, first), acts))
                if not cands:
                    break
                target = task.t_gen + p.margin * p.deadline_s
                arr, acts = _select(net, task, cands, target, deadline_abs, cpu_rule, p,
                                    risk=(policy == "cacgr" and p.p_miss > 0))
            else:
                arr, acts = plan(net, node, stage, t, now, cpu_rule, _any_link)
                if acts is None:
                    break
            status, node, stage, t = execute(net, acts, node, stage, t, now, out, power)
            if status == "done":
                out.t_done = t
                break
            out.replans += 1
        lat = out.t_done - task.t_gen
        rows.append(dict(policy=policy, seed=seed, tid=task.tid, src=task.src,
                         t_gen=task.t_gen, latency_s=lat,
                         on_time=bool(lat <= p.deadline_s),
                         delivered=bool(math.isfinite(lat)),
                         energy_j=out.energy_seabed, where_a=out.where_a,
                         where_b=out.where_b, replans=out.replans,
                         acoustic_s=out.acoustic_s))
    return rows
