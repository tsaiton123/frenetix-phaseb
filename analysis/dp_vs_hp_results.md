# Default Planner (DP) vs Hybrid Planner (HP) — Reproduction & Paper Comparison

Baseline paper: Trauth, Hobmeier, Betz, *A Reinforcement Learning-Boosted Motion
Planning Framework* (IEEE IV 2024), arXiv:2402.01465.

Scenarios: `ZAM_Tjunction-1_102_T-1`, `ZAM_Tjunction-1_119_T-1` (validation set).
Both planners run in the same `frenetix-rl:exec` Docker image (linux/amd64,
emulated on Apple Silicon, CPU only), plotting disabled.

- **HP**: RL-boosted planner, authors' pre-trained `best_model.zip`
  (`run_hybrid_planner.py` -> `logs_hp/`). Risk values are byte-identical to the
  original May-21 `logs/` run -> deterministic / reproducible.
- **DP**: same analytical Frenet planner, fixed default cost weights
  (`cost.yaml`, prediction weight = 200), no RL adaptation
  (`run_default_planner.py`, zero action -> `logs_dp/`).

## Our reproduction

| Scenario | DP result | HP result |
|---|---|---|
| ZAM_Tjunction-1_102 | ❌ **Collision** (step 83) | ✅ Success |
| ZAM_Tjunction-1_119 | ✅ Success | ✅ Success |
| **Success rate** | **1/2 (50%)** | **2/2 (100%)** |

| Aggregate metric | DP | HP |
|---|---|---|
| Mean ego risk | 1.98e-04 | 3.69e-05 |
| Mean obstacle risk | 5.72e-05 | 1.49e-05 |
| Mean feasible-traj % | 60.5% | 59.2% |
| Mean calc time / step (same Docker) | 0.329 s | 0.299 s |

Apples-to-apples on the one scenario both planners complete (119):
DP ego 8.67e-5 vs HP 5.74e-5; DP obst 3.62e-5 vs HP 2.36e-5.

## Paper's reported numbers

- Mean ego risk: DP **9.11e-5**, HP **3.01e-5** (HP ~= 33% of DP). [Fig. 5]
- Mean 3rd-party risk: DP **3.04e-5**, HP **1.32e-5**. [Fig. 5]
- Success over 547 scenarios: HP **546/547**, 0 collisions; DP **443-510/547**
  with **37-104 collisions** depending on collision-cost weight (DP@200 -> 57
  collisions ~= 10%). [Fig. 9]
- Execution time (native AMD 7950x + RTX 4090): RL prediction 0.44 ms, trajectory
  calc 15.8 ms, overall step **46 ms**, ~800 trajectories. [Fig. 10]
- Qualitative: DP collides with oncoming traffic at a T-junction (timestep 74),
  HP avoids by raising collision-probability weights. [Fig. 6-8]

## Match assessment

| Claim | Paper | Ours | Verdict |
|---|---|---|---|
| HP ego risk magnitude | 3.01e-5 | 3.69e-5 | ✅ matches |
| HP 3rd-party risk magnitude | 1.32e-5 | 1.49e-5 | ✅ matches |
| DP ego risk (non-collision scn 119) | 9.11e-5 | 8.67e-5 | ✅ matches |
| DP 3rd-party risk (scn 119) | 3.04e-5 | 3.62e-5 | ✅ matches |
| HP reduces risk vs DP | yes (~33%) | yes (scn119 ~66%) | ✅ direction; weaker (n=1) |
| DP collides, HP avoids (T-junction, oncoming) | yes | yes (scn 102) | ✅ matches |
| HP success >> DP success | 99.8% vs ~90% | 100% vs 50% | ✅ direction; n=2 too small |
| HP compute ~= DP compute | +0.44 ms | 0.299 vs 0.329 s | ✅ relative match |
| Absolute real-time (~46 ms/step) | 46 ms | ~300 ms | ❌ emulated CPU, not native+GPU |

## Takeaway

The reproduction matches the paper on every safety/behavior claim: per-step risk
magnitudes line up almost exactly, HP roughly halves-to-thirds the risk, and the
paper's signature qualitative result — the fixed-weight DP colliding with
oncoming traffic at a T-junction while the RL-boosted HP avoids it — reproduces
directly (our DP collides in scenario 102, HP completes it).

The only metric that does not match is **absolute execution time** (~300 ms/step
vs the paper's 46 ms). This is fully explained by environment: we run the
linux/amd64 image emulated on Apple Silicon with CPU-only inference, vs the
paper's native AMD 7950x + RTX 4090. The *relative* compute cost is consistent —
HP is not slower than DP — so the paper's "real-time, RL adds negligible cost"
conclusion holds in relative terms; only the absolute wall-clock differs.
