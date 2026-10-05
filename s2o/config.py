"""Default parameters of the seabed-to-orbit case study (see Table 1 of the paper)."""
from __future__ import annotations

from dataclasses import dataclass, field, replace

MIN = 60.0
HOUR = 3600.0


@dataclass(frozen=True)
class Params:
    # ---- deployment -------------------------------------------------------
    n_nodes: int = 8                 # seabed sensing nodes
    site_km: float = 7.0             # square site edge; buoy at the centre
    depth_km: float = 0.8            # seabed depth
    # ---- workload (passive acoustic monitoring clip) -----------------------
    rate_per_node_h: float = 2.0     # Poisson task rate per node (tasks/h)
    gen_hours: float = 48.0          # tasks are generated during this period
    raw_bits: float = 4 * 8e6        # 4 MB raw clip
    feat_ratio: float = 0.04         # stage A output / raw size
    result_bits: float = 2 * 8e3     # 2 kB semantic result (stage B output)
    ops_a_gop: float = 4.0           # stage A: feature extraction + learned compression
    ops_b_gop: float = 400.0         # stage B: detector/classifier on features
    deadline_s: float = 90 * MIN     # application deadline for the result
    # ---- compute capabilities (GOPS) ----------------------------------------
    gops_seabed: float = 1.0
    gops_auv: float = 200.0
    gops_buoy: float = 10.0
    gops_uav: float = 50.0
    gops_sat: float = 100.0
    gops_cloud: float = 5000.0
    # ---- seabed power draw (W), used for the seabed energy metric -----------
    p_cpu_seabed: float = 0.5
    p_tx_acoustic: float = 15.0
    p_tx_optical: float = 3.0
    # ---- links ---------------------------------------------------------------
    optical_bps: float = 10e6        # seabed -> hovering AUV
    dock_bps: float = 20e6           # docked AUV -> buoy
    uav_bps: float = 20e6            # buoy -> hovering UAV
    shore_bps: float = 100e6         # landed UAV -> cloud
    leo_up_bps: float = 1e6          # buoy -> LEO satellite
    leo_down_bps: float = 100e6      # LEO satellite -> ground station / cloud
    # ---- contact plan --------------------------------------------------------
    auv_period_h: float = 8.0        # one survey tour every 8 h
    auv_speed_mps: float = 1.5
    auv_vspeed_mps: float = 1.0
    auv_hover_min: float = 8.0       # optical window at each node
    auv_min_dock_min: float = 30.0
    uav_period_h: float = 8.0
    uav_hover_min: float = 15.0
    uav_return_min: float = 60.0
    uav_shore_min: float = 30.0
    leo_gap_min: float = 30.0        # mean revisit gap between usable passes
    leo_pass_min: tuple = (4.0, 10.0)
    leo_dl_delay_min: tuple = (5.0, 45.0)  # pass end -> first ground-station window
    leo_dl_window_min: float = 8.0
    leo_dl_repeat_min: float = 95.0
    leo_dl_count: int = 6
    tail_hours: float = 96.0         # contacts generated beyond the task period
    # ---- uncertainty ---------------------------------------------------------
    p_miss: float = 0.0              # probability that a scheduled contact does not happen
    # ---- planner ---------------------------------------------------------------
    margin: float = 0.8              # energy-frugal plan must arrive by margin * deadline
    reliability: float = 0.95        # min estimated on-time probability of a frugal plan
    leo_lookahead: int = 12          # passes examined per expansion at the buoy

    def with_(self, **kw) -> "Params":
        return replace(self, **kw)


DEFAULT = Params()
