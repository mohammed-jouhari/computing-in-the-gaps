"""Builds one random realisation of the seabed-to-orbit scenario (geometry, contact plan, failures, tasks)."""
from __future__ import annotations

import math

import numpy as np

from .channel import acoustic_delay_s, acoustic_goodput_bps
from .config import HOUR, MIN, Params
from .engine import INF, Bookings, Link, Network, Task, Window


def _nn_tour(points):
    """Nearest-neighbour visiting order starting and ending at the origin (buoy)."""
    left = list(range(len(points)))
    order, cur = [], np.zeros(2)
    while left:
        j = min(left, key=lambda k: np.linalg.norm(points[k] - cur))
        order.append(j)
        cur = points[j]
        left.remove(j)
    return order


def build(p: Params, seed: int):
    g_geo, g_auv, g_uav, g_leo, g_task = [np.random.default_rng(s)
                                          for s in np.random.SeedSequence(seed).spawn(5)]
    horizon = (p.gen_hours + p.tail_hours) * HOUR
    sizes = (p.raw_bits, p.raw_bits * p.feat_ratio, p.result_bits)
    ops = (p.ops_a_gop, p.ops_b_gop)
    net = Network(sizes, ops)
    net.leo_lookahead = p.leo_lookahead
    miss = lambda g: bool(g.random() < p.p_miss)

    # ---------------- nodes ----------------
    half = p.site_km / 2
    xy = g_geo.uniform(-half, half, size=(p.n_nodes, 2))
    slant = np.sqrt(np.sum(xy ** 2, axis=1) + p.depth_km ** 2)
    names = [f"sb{i}" for i in range(p.n_nodes)]
    for nm in names:
        net.add_node(nm, "seabed", p.gops_seabed)
    net.add_node("auv", "auv", p.gops_auv)
    net.add_node("buoy", "buoy", p.gops_buoy)
    net.add_node("uav", "uav", p.gops_uav)
    net.add_node("cloud", "cloud", p.gops_cloud, shared=False)

    acoustic = Bookings()  # one shared acoustic medium for the whole site
    rates = [acoustic_goodput_bps(d) for d in slant]
    for i, nm in enumerate(names):
        net.add_link(Link(nm, "buoy", "ac",
                          [Window(0.0, INF, rates[i], acoustic_delay_s(slant[i]))], acoustic))

    # ---------------- AUV tours ----------------
    order = _nn_tour(xy)
    v, vz = p.auv_speed_mps / 1000.0, p.auv_vspeed_mps / 1000.0  # km/s
    descent = p.depth_km / vz
    hover = p.auv_hover_min * MIN
    opt_w = {nm: [] for nm in names}
    auv_ac_w = []
    dock_w = []
    t = g_auv.uniform(0, p.auv_period_h * HOUR)
    dock_w.append(Window(0.0, t, p.dock_bps))
    tour_s = None
    while t < horizon:
        start = t
        cur, tt = np.zeros(2), t + descent
        for j in order:
            tt += np.linalg.norm(xy[j] - cur) / v
            w_fail = miss(g_auv)
            opt_w[names[j]].append(Window(tt, tt + hover, p.optical_bps, 0.0, w_fail))
            # while hovering the AUV can also reach the buoy acoustically (same medium)
            auv_ac_w.append(Window(tt, tt + hover, rates[j], acoustic_delay_s(slant[j]), w_fail))
            tt += hover
            cur = xy[j]
        tt += np.linalg.norm(cur) / v + descent
        if tour_s is None:
            tour_s = tt - start
        nxt = max(start + p.auv_period_h * HOUR, tt + p.auv_min_dock_min * MIN)
        dock_w.append(Window(tt, nxt, p.dock_bps))
        t = nxt
    for nm in names:
        net.add_link(Link(nm, "auv", "opt", opt_w[nm], Bookings()))
    net.add_link(Link("auv", "buoy", "auv_ac", auv_ac_w, acoustic))
    net.add_link(Link("auv", "buoy", "dock", dock_w, Bookings()))

    # ---------------- UAV sorties ----------------
    uav_w, shore_w = [], []
    t = g_uav.uniform(0, p.uav_period_h * HOUR)
    while t < horizon:
        f = miss(g_uav)
        uav_w.append(Window(t, t + p.uav_hover_min * MIN, p.uav_bps, 0.0, f))
        land = t + (p.uav_hover_min + p.uav_return_min) * MIN
        shore_w.append(Window(land, land + p.uav_shore_min * MIN, p.shore_bps, 0.0, f))
        t += p.uav_period_h * HOUR
    net.add_link(Link("buoy", "uav", "uav", uav_w, Bookings()))
    net.add_link(Link("uav", "cloud", "shore", shore_w, None))

    # ---------------- LEO passes ----------------
    terminal = Bookings()  # the buoy has one satellite terminal
    t = g_leo.uniform(0, p.leo_gap_min * MIN)
    k = 0
    while t < horizon:
        dur = g_leo.uniform(*p.leo_pass_min) * MIN
        sat = f"leo{k}"
        net.add_node(sat, "sat", p.gops_sat)
        net.add_link(Link("buoy", sat, "up", [Window(t, t + dur, p.leo_up_bps, 0.0, miss(g_leo))], terminal))
        dl, t_dl = [], t + dur + g_leo.uniform(*p.leo_dl_delay_min) * MIN
        for _ in range(p.leo_dl_count):
            dl.append(Window(t_dl, t_dl + p.leo_dl_window_min * MIN, p.leo_down_bps, 0.0, miss(g_leo)))
            t_dl += p.leo_dl_repeat_min * MIN
        net.add_link(Link(sat, "cloud", "down", dl, Bookings()))
        t += dur + g_leo.uniform(0.5, 1.5) * p.leo_gap_min * MIN
        k += 1
    net.finalize()

    # ---------------- tasks ----------------
    tasks = []
    lam = p.rate_per_node_h / HOUR
    for nm in names:
        t = g_task.exponential(1 / lam)
        while t < p.gen_hours * HOUR:
            tasks.append((t, nm))
            t += g_task.exponential(1 / lam)
    tasks.sort()
    tasks = [Task(i, nm, t) for i, (t, nm) in enumerate(tasks)]

    info = dict(slant_km=slant, rates_bps=rates, tour=order, tour_s=tour_s,
                n_passes=k)
    return net, tasks, info
