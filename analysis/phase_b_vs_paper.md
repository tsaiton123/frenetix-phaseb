# Phase B (Dynamic Sampling) vs HP-100k (Fair Budget) vs Paper HP

## Setup

Both ours-trained agents share **everything except the action space**:
- Same training budget: **100k steps** (paper used 7M)
- Same dataset split: 410 train / 82 val / **55 test** (seed=42)
- Same env, same prediction network, same reward terms
- Same hardware/runtime for evaluation (emulated linux/amd64 Docker on Mac)

| Agent | Action space | Sampling grid | Trained on |
|---|---|---|---|
| **HP-100k** (ours) | 5-dim (prediction + 4 cost weights) | **fixed** (paper's default) | 100k steps |
| **Phase B-100k** (ours) | 7-dim (HP + horizon + lateral half-span) | **RL-controlled** | 100k steps |
| Paper HP (reference) | 5-dim | fixed | 7M steps (70× more) |
| Paper DP (reference) | n/a (no RL) | fixed | n/a |

## 1. Success / collision (paper Fig 9 metric)

| Method | Set | Success | Collision | Other | Success % | Collision % |
|---|---|---|---|---|---|---|
| Paper HP (7M) | 547 | 546 | 1 | 0 | 99.82% | 0.18% |
| Paper DP (pred=200) | 547 | 490 | 57 | 0 | 89.58% | 10.42% |
| **HP-100k (ours)** | **55** | **54** | **0** | **1 (out-of-time)** | **98.18%** (100% goal-reached) | **0%** |
| **Phase B-100k (ours)** | **55** | **43** | **12** | **0** | **78.18%** | **21.82%** |

Headline: **at the same 100k-step budget**, dropping fixed-grid for RL-controlled
grid costs ~20 percentage points of success and adds 12 collisions.

## 2. Risk (paper Fig 5 metric — R = max p(T) · H(T))

| Method | Mean ego risk | Mean 3rd-party risk |
|---|---|---|
| Paper HP (7M) | 3.01e-5 | 1.32e-5 |
| Paper DP | 9.11e-5 | 3.04e-5 |
| **HP-100k (ours)** | **2.30e-5** | **9.20e-6** |
| **Phase B-100k (ours)** | **3.07e-4** | **7.99e-5** |

Phase B-100k has **~13× higher ego risk** and **~9× higher 3rd-party risk** than
HP-100k on the same test set. The lower-than-paper HP-100k numbers are likely
sample-size noise (n=55 vs n=64), or modestly easier test scenarios than the
paper's set.

## 3. Trajectory feasibility & calc time

| Method | Feasible-traj % | Calc time / step | Note |
|---|---|---|---|
| Paper HP (native + GPU) | n/a | ~46 ms (RL pred + traj calc) | not comparable across hardware |
| HP-100k (emulated Docker) | 57.5% | **0.262 s** | single env, no GPU |
| Phase B-100k (emulated Docker) | 65.7% | **0.364 s** | same hardware |

Phase B is actually **40% slower per step** than HP-100k despite using 30% fewer
candidates. Likely causes: per-step planning is harder when the agent shrinks
the grid in difficult states (more cost-checking failures), and collision-prone
states inflate average step cost.

## 4. Phase B's unique contribution — the dynamic-sampling mechanism

These rows have no paper counterpart because the paper uses a **fixed** sampling
grid. They are the contribution of this work:

| Knob | Fixed (HP, paper, HP-100k) | Phase B-100k | Reduction |
|---|---|---|---|
| Longitudinal horizon | 3.0 s | **2.28 s** | 24% shorter |
| Lateral half-span | 3.5 m | **1.90 m** | 46% narrower |
| Candidate trajectories per step | ~800 | **560** | **30% fewer** |

## 5. Honest interpretation

- **The dynamic-sampling mechanism is real and learnable.** The Phase B agent
  consistently runs ~30% under the fixed-grid candidate count, with both knobs
  in active use (not collapsed to extremes).
- **Same-budget head-to-head: HP-100k beats Phase B-100k decisively on safety**
  (0 vs 12 collisions). The 7-dim action space is a harder learning problem at
  fixed budget — the policy network has to learn both *what cost weights* and
  *what grid shape* the situation needs.
- **Surprising finding: Phase B is also slower per step**, despite 30% fewer
  candidates. So at 100k steps, dynamic sampling buys you neither safety nor
  wall-clock — the only thing it buys is grid-size reduction.
- **Probable remedies the report can suggest as future work:**
  (a) more training (this aligns with paper's HP requiring 7M steps),
  (b) lower the efficiency reward weight (currently -0.0002),
  (c) a curriculum: train weights-only first, then unfreeze the sampling head.

## 6. What to claim in the report

- ✅ Phase B mechanism implemented, validated, and trained end-to-end with
  same compute budget as a fair-comparison HP baseline (HP-100k).
- ✅ Quantitative evidence of learned grid shaping: **30% candidate count
  reduction**, 24% horizon reduction, 46% lateral-span reduction (vs fixed
  paper grid). Both knobs actively used.
- ⚠️ At 100k training steps, the dynamic-sampling formulation **trades safety
  for grid efficiency**: 0 → 12 collisions, ego risk 13× higher, 3rd-party
  risk 9× higher vs HP at the same budget.
- ⚠️ Compute efficiency did NOT translate to wall-clock at this scale — see
  §3. Phase B is 40% slower per step.
- 🔜 **Future work**: (i) longer training to recover safety, (ii) sweep
  `reward_sampling_efficiency` (currently -0.0002 may be too aggressive),
  (iii) curriculum learning to bootstrap from the HP baseline, (iv) expose
  `change_max_sampling_level()` as a third RL action for direct density
  control.

## 7. One-line summary for the abstract

> Adding an RL-controlled sampling grid on top of the Trauth et al. RL-boosted
> Frenet planner reduces candidate trajectories by 30% per step but
> significantly degrades safety at matched training budget; recovering the
> baseline's safety while keeping the efficiency gain remains the key open
> problem.
