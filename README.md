# Computing in the Gaps: seabed-to-orbit simulator

This repository contains the simulator used in the article:

M. Jouhari and H. Benaddi, "Computing in the Gaps: Contact-Aware Edge Processing from Seabed to Orbit," submitted to *IEEE Communications Magazine*, Special Issue on Edge and In-Network Computing for Underwater Communication.

The code produces Figs. 3 to 5 of the article and all the numbers quoted in its case study. The files in `results/` are the results used in the article, and a clean rerun gives the same values.

## What the simulator models

The simulator works at the level of contacts and bookings. It considers the following elements.

| Element | Model |
|---|---|
| Network | Eight seabed nodes on a 7 km x 7 km site at 800 m depth, a surface buoy used as gateway, one AUV with optical pick-up, acoustic relay and dock links, UAV sorties, LEO satellite passes with store-and-forward downlinks, and the cloud |
| Links | Each link is a list of contact windows (start, end, rate, delay). Links that share a medium share one booking table: one acoustic channel per site and one satellite terminal on the buoy. The acoustic goodput comes from the capacity-distance model of Stojanovic, with Thorp absorption and ambient noise, mapped to four modem modes |
| Workload | Passive acoustic monitoring clips of 4 MB, processed in two stages: stage A (features, 4 GOP, 160 kB) and stage B (detector and classifier, 400 GOP by default, 2 kB report) |
| Planner | Compute-augmented contact graph, where vertices are (node, stage) pairs and edges are contacts or processing steps. An earliest-arrival label-setting search with first-fit bookings returns the plan |
| Policies | CA-CGR with the deadline-aware energy rule and the on-time probability check, CA-CGR without the check, and four baselines: cloud-centric, on-sensor, surface MEC, and statistical placement |

## Installation

Python 3.9 or newer is needed, with NumPy, pandas, and Matplotlib.

```bash
git clone https://github.com/YOUR-USERNAME/computing-in-the-gaps.git
cd computing-in-the-gaps
python3 -m venv .venv
source .venv/bin/activate        # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Reproducing the results

On Linux or macOS, one script runs the tests, the parameter sweeps, the figures, and the summary of the numbers:

```bash
./run_all.sh
```

It takes about eight minutes on a two-core computer.

The same steps can be run one by one. This is also the way to run the code on Windows, where `run_all.sh` needs Git Bash or WSL:

```bash
python tests/test_engine.py              # sanity tests
python -m s2o.experiments --seeds 30     # writes results/sweep_*.csv
python -m s2o.plots                      # writes figures/fig3 to fig5 (.pdf and .png)
python -m s2o.summary                    # prints the numbers quoted in the article
```

The option `--sweeps model deadline` runs only some of the sweeps, and `--procs N` sets the number of worker processes. All runs are seeded with `numpy.random.SeedSequence`, so the same seeds always give the same results. Contact losses use common random numbers, so a higher loss probability removes a superset of the contacts lost at a lower one.

For a quick test of the installation, `SEEDS=2 ./run_all.sh` uses two seeds instead of 30. This overwrites the files in `results/` and `figures/` with less precise values. To get back the results of the article, run `git checkout results figures`.

## Repository content

| File or folder | Content |
|---|---|
| `s2o/config.py` | all parameters (Table 1 of the article) |
| `s2o/channel.py` | acoustic path loss, noise, SNR, and modem goodput |
| `s2o/scenario.py` | geometry, contact plan (AUV tours, UAV sorties, LEO passes), contact losses, and tasks |
| `s2o/engine.py` | bookings, contact windows, planner (search), and executor |
| `s2o/policies.py` | CA-CGR, CA-CGR without the check, and the baselines; per-task simulation loop |
| `s2o/experiments.py` | parameter sweeps (30 seeds per point by default) |
| `s2o/plots.py` | Figs. 3 to 5 |
| `s2o/summary.py` | numbers quoted in the text |
| `tests/test_engine.py` | sanity tests of the planner and the executor |
| `results/` | CSV outputs of the sweeps used in the article, and `summary.txt` |
| `figures/` | figures used in the article |
| `run_all.sh` | script that reproduces all the results |
| `requirements.txt` | Python packages |

## Limits of the model

* The simulator does not model packets. MAC collisions and bit errors inside a contact are not simulated. The acoustic modem mode and a MAC efficiency factor of 0.7 represent these effects.
* Tasks are planned in the order of their creation, and each plan sees the bookings of the earlier tasks immediately. The article discusses this assumption.
* A lost contact is detected at the time when its window should start, and the node then plans again.

## Citation

If you use this code, please cite the article:

```bibtex
@misc{jouhari2026gaps,
  author = {Jouhari, Mohammed and Benaddi, Hafsa},
  title  = {Computing in the Gaps: Contact-Aware Edge Processing from Seabed to Orbit},
  note   = {Submitted to IEEE Communications Magazine},
  year   = {2026}
}
```

## Use of AI tools

The simulation code was written with Claude (Anthropic) and checked by the authors, as stated in the Acknowledgment of the article.

## License

This code is released under the MIT License (see `LICENSE`).

## Contact

Mohammed Jouhari, Ibn Tofail University, Kenitra, Morocco (mohammed.jouhari1@uit.ac.ma)

Hafsa Benaddi, Hassan II University of Casablanca, Morocco (hafsa.benaddi@univh2c.ma)
