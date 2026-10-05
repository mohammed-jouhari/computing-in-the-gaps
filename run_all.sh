#!/usr/bin/env bash
# Reproduces all results of the paper: tests, sweeps, figures and quoted numbers.
set -euo pipefail
cd "$(dirname "$0")"
SEEDS="${SEEDS:-30}"
python3 tests/test_engine.py
python3 -W ignore -m s2o.experiments --seeds "$SEEDS"
python3 -m s2o.plots
python3 -m s2o.summary
# keep the paper's copies of the figures in sync when the paper folder is next to this one
if [ -d ../paper/figures ]; then cp figures/fig3_workload.pdf figures/fig4_deadline_energy.pdf figures/fig5_placement.pdf ../paper/figures/; fi
