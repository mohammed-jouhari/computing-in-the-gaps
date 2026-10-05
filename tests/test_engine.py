"""Sanity tests for the planner/executor. Run: python -m pytest -q tests (or python tests/test_engine.py)."""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from s2o.config import DEFAULT, MIN  # noqa: E402
from s2o.engine import INF, Bookings, Link, Network, Window, execute, plan, Outcome  # noqa: E402
from s2o.policies import POLICIES, ABLATIONS, run  # noqa: E402
from s2o.scenario import build  # noqa: E402


def _no_overlap(b: Bookings):
    for i in range(1, len(b.starts)):
        assert b.starts[i] >= b.ends[i - 1] - 1e-9, "overlapping bookings"
    for s, e in zip(b.starts, b.ends):
        assert e >= s


def test_bookings_first_gap():
    b = Bookings()
    b.add(10, 20)
    b.add(30, 40)
    assert b.first_gap(0, 5, INF) == 0
    assert b.first_gap(8, 5, INF) == 20
    assert b.first_gap(21, 10, INF) == 40
    assert b.first_gap(21, 9, INF) == 21
    assert b.first_gap(21, 10, 45) is None


def test_two_hop_window_plan():
    net = Network(sizes=(100.0, 10.0, 1.0), ops=(1.0, 1.0))
    net.add_node("a", "seabed", 1.0)
    net.add_node("buoy", "buoy", 1.0)
    net.add_node("cloud", "cloud", 1e6, shared=False)
    net.add_link(Link("a", "buoy", "opt", [Window(50, 60, 100.0)], Bookings()))
    net.add_link(Link("buoy", "cloud", "uav", [Window(100, 200, 10.0)], Bookings()))
    net.finalize()
    allow = lambda n, s: True
    t, acts = plan(net, "a", 0, 0.0, 0.0, allow, lambda l, s: True)
    # compute both stages at the source (2 s), wait for the window at 50 s,
    # send 1 bit, wait for the second window at 100 s, send 1 bit -> 100.1 s
    assert abs(t - 100.1) < 1e-9, t
    out = Outcome()
    st, node, stage, tt = execute(net, acts, "a", 0, 0.0, 0.0, out, {"cpu": 1, "opt": 1})
    assert st == "done" and node == "cloud" and stage == 2 and abs(tt - t) < 1e-9


def test_failed_window_is_skipped_after_notice():
    net = Network(sizes=(1.0, 1.0, 1.0), ops=(0.0, 0.0))
    net.add_node("buoy", "buoy", 1.0)
    net.add_node("cloud", "cloud", 1e6, shared=False)
    net.add_link(Link("buoy", "cloud", "uav",
                      [Window(10, 20, 1.0, 0.0, True), Window(30, 40, 1.0)], Bookings()))
    net.finalize()
    allow = lambda n, s: True
    t_before, _ = plan(net, "buoy", 2, 0.0, 0.0, allow, lambda l, s: True)
    t_after, _ = plan(net, "buoy", 2, 10.0, 10.0, allow, lambda l, s: True)
    assert abs(t_before - 11.0) < 1e-9 and abs(t_after - 31.0) < 1e-9


def test_full_runs_are_consistent():
    p = DEFAULT.with_(gen_hours=12.0, p_miss=0.2, deadline_s=360 * MIN)
    for pol in POLICIES + ABLATIONS:
        rows = run(p, 3, pol)
        assert rows, pol
        for r in rows:
            assert r["latency_s"] > 0
            assert r["energy_j"] >= 0
            if r["delivered"]:
                assert math.isfinite(r["latency_s"])
    net, tasks, info = build(p, 3)
    assert len(tasks) > 0 and info["tour_s"] > 0


def test_bookings_never_overlap_after_run():
    import s2o.policies as pol
    p = DEFAULT.with_(gen_hours=12.0, rate_per_node_h=6.0)
    captured = {}
    orig = pol.build

    def spy(pp, seed):
        net, tasks, info = orig(pp, seed)
        captured["net"] = net
        return net, tasks, info

    pol.build = spy
    try:
        run(p, 5, "cacgr")
    finally:
        pol.build = orig
    net = captured["net"]
    seen = set()
    for node in net.nodes.values():
        for link in [node.proc] + node.out + (net.sat_uplinks if node.name == "buoy" else []):
            if link.book is not None and id(link.book) not in seen:
                seen.add(id(link.book))
                _no_overlap(link.book)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
