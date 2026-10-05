"""Time-dependent planner and executor.

The network is a set of nodes connected by *links* whose availability is a list of
contact windows (start, end, rate, delay). Links that share a physical medium (the
acoustic channel of the site) share one booking table, so a transfer must find a
free slot that fits inside one window. Processors are modelled the same way: a
processor is a link from (node, stage s) to (node, stage s+1) whose "rate" is the
node speed in GOP/s.

Planning is an earliest-arrival search over states (node, stage). Because the
earliest feasible finish time never decreases when the ready time increases
(first-fit booking), the label-setting search returns the earliest result
delivery for the given set of allowed actions.
"""
from __future__ import annotations

import bisect
import heapq
import math
from dataclasses import dataclass, field

INF = math.inf


class Bookings:
    """Sorted, non-overlapping busy intervals of one resource."""

    __slots__ = ("starts", "ends")

    def __init__(self):
        self.starts: list[float] = []
        self.ends: list[float] = []

    def first_gap(self, t: float, dur: float, limit: float):
        """Earliest s >= t with [s, s+dur] free and s+dur <= limit, else None."""
        i = bisect.bisect_right(self.ends, t)
        s = t
        n = len(self.starts)
        while i < n and self.starts[i] < s + dur:
            if self.ends[i] > s:
                s = self.ends[i]
            i += 1
            if s + dur > limit:
                return None
        return s if s + dur <= limit else None

    def add(self, s: float, e: float):
        i = bisect.bisect_left(self.starts, s)
        self.starts.insert(i, s)
        self.ends.insert(i, e)


@dataclass
class Window:
    start: float
    end: float
    rate: float            # bit/s for links, GOP/s for processors
    delay: float = 0.0     # propagation delay added after the transfer
    failed: bool = False   # ground truth: the contact does not happen


@dataclass
class Link:
    src: str
    dst: str
    kind: str                       # 'ac', 'opt', 'dock', 'auv_ac', 'uav', 'shore', 'up', 'down'
    windows: list
    book: Bookings | None           # None: unlimited parallel capacity
    ends: list = field(default_factory=list)

    def finalize(self):
        self.windows.sort(key=lambda w: w.start)
        self.ends = [w.end for w in self.windows]

    def earliest(self, t: float, amount: float, now: float):
        """Earliest (start, finish, window) for a transfer of `amount` ready at t.

        Windows that failed and started at or before `now` are known to be lost
        and are skipped; later windows are assumed to follow the plan.
        """
        i = bisect.bisect_right(self.ends, t)
        for w in self.windows[i:]:
            if w.failed and w.start <= now:
                continue
            if w.rate <= 0:
                continue
            dur = amount / w.rate
            s0 = max(t, w.start)
            if s0 + dur > w.end:
                continue
            if self.book is None:
                return s0, s0 + dur, w
            s = self.book.first_gap(s0, dur, w.end)
            if s is not None:
                return s, s + dur, w
        return None


@dataclass
class Node:
    name: str
    tier: str            # 'seabed', 'auv', 'buoy', 'uav', 'sat', 'cloud'
    proc: Link           # processor modelled as a self-link
    out: list = field(default_factory=list)


@dataclass
class Task:
    tid: int
    src: str
    t_gen: float


class Network:
    def __init__(self, sizes, ops):
        self.nodes: dict[str, Node] = {}
        self.sizes = sizes          # bits at stage 0, 1, 2
        self.ops = ops              # GOP for transitions 0->1 and 1->2
        self.sat_uplinks: list = []  # buoy -> satellite links sorted by window start
        self.sat_upl_ends: list = []
        self.leo_lookahead = 12

    def add_node(self, name, tier, gops, shared=True):
        proc = Link(name, name, "cpu", [Window(0.0, INF, gops)],
                    Bookings() if shared else None)
        proc.finalize()
        self.nodes[name] = Node(name, tier, proc)

    def add_link(self, link: Link):
        link.finalize()
        self.nodes[link.src].out.append(link)
        if link.kind == "up":
            self.sat_uplinks.append(link)

    def finalize(self):
        self.sat_uplinks.sort(key=lambda l: l.windows[0].start)
        self.sat_upl_ends = [l.windows[0].end for l in self.sat_uplinks]
        buoy = self.nodes.get("buoy")
        if buoy is not None:
            buoy.out = [l for l in buoy.out if l.kind != "up"]

    def out_links(self, node: Node, t: float):
        if node.name == "buoy":
            i = bisect.bisect_right(self.sat_upl_ends, t)
            return node.out + self.sat_uplinks[i:i + self.leo_lookahead]
        return node.out


def plan(net: Network, src: str, stage: int, t0: float, now: float,
         allow_cpu, allow_link, dest="cloud"):
    """Earliest-arrival search over (node, stage). Returns (t_arrival, actions) or (INF, None).

    allow_cpu(node, stage) and allow_link(link, stage) restrict the action set
    (this is how fixed-placement baselines and seabed profiles are expressed).
    """
    best = {(src, stage): t0}
    prev = {}
    heap = [(t0, src, stage)]
    goal = (dest, 2)
    while heap:
        t, n, s = heapq.heappop(heap)
        if best.get((n, s), INF) < t:
            continue
        if (n, s) == goal:
            acts = []
            k = goal
            while k in prev:
                k, act = prev[k]
                acts.append(act)
            acts.reverse()
            return t, acts
        node = net.nodes[n]
        if s < 2 and allow_cpu(node, s):
            r = node.proc.earliest(t, net.ops[s], now)
            if r is not None:
                st, fin, w = r
                k = (n, s + 1)
                if fin < best.get(k, INF):
                    best[k] = fin
                    prev[k] = ((n, s), ("cpu", node.proc, s, w, st))
                    heapq.heappush(heap, (fin, n, s + 1))
        for link in net.out_links(node, t):
            if not allow_link(link, s):
                continue
            r = link.earliest(t, net.sizes[s], now)
            if r is None:
                continue
            st, fin, w = r
            arr = fin + w.delay
            k = (link.dst, s)
            if arr < best.get(k, INF):
                best[k] = arr
                prev[k] = ((n, s), ("tx", link, s, w, st))
                heapq.heappush(heap, (arr, link.dst, s))
    return INF, None


@dataclass
class Outcome:
    t_done: float = INF
    energy_seabed: float = 0.0
    where_a: str = "-"
    where_b: str = "-"
    replans: int = 0
    acoustic_s: float = 0.0


def execute(net: Network, actions, node: str, stage: int, t: float, plan_now: float,
            out: Outcome, power):
    """Run a plan step by step, booking resources. Returns (status, node, stage, t).

    status is 'done' or 'fail' (a contact used by the plan did not happen; t is
    then the time at which the node notices the missing contact).
    """
    for kind, link, s, *_ in actions:
        amount = net.ops[s] if kind == "cpu" else net.sizes[s]
        r = link.earliest(t, amount, plan_now)
        if r is None:  # should not happen: same state as at planning time
            return "fail", node, stage, t
        st, fin, w = r
        if w.failed:
            return "fail", node, stage, max(t, w.start)
        if link.book is not None:
            link.book.add(st, fin)
        tier = net.nodes[link.src].tier
        if kind == "cpu":
            if tier == "seabed":
                out.energy_seabed += (fin - st) * power["cpu"]
            if s == 0:
                out.where_a = tier
            else:
                out.where_b = tier
            stage = s + 1
            t = fin
        else:
            if tier == "seabed":
                out.energy_seabed += (fin - st) * power[link.kind]
            if link.kind in ("ac", "auv_ac"):
                out.acoustic_s += fin - st
            node = link.dst
            t = fin + w.delay
    return "done", node, stage, t
