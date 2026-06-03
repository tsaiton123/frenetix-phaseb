# Phase B — Dynamic Sampling Optimization (implementation)

Proposal contribution: let the RL agent shape the Frenet **sampling grid**
(longitudinal horizon + lateral span), not just the cost weights.

## What changed

| File | Change |
|---|---|
| `frenetix_rl/gym_environment/configs.yaml` | `action_type: weights` (baseline) ↔ `weights_and_sampling` (Phase B). Added `sample_t_low: 1.5`, `sample_d_low: 0.5`, `sample_d_high: 3.5`, and `reward_sampling_efficiency: 0.`. |
| `.../environment/agent_env.py` | Action space grows by 2 dims when sampling enabled. New `_apply_sampling_action()` calls `planner.set_sampling_parameters(t_min, horizon, -dd, +dd)` each step. Absolute rescaling for horizon ∈ [1.5, 3.0] s and lateral half-span dd ∈ [0.5, 3.5] m. |
| `.../reward/hybrid_reward.py` | New `_sampling_efficiency_reward()` = `reward_sampling_efficiency × len(agent.all_trajectories)` — rewards a smaller grid. |

Action layout (Phase B, 7-dim): `[prediction, lat_jerk, lon_jerk, dist_ref, vel_offset, horizon, lateral_half_span]`.
Backward compatible: `action_type: weights` keeps the 5-dim baseline and `best_model.zip`.

## Integration point (confirmed via in-container introspection)

`frenetix_motion_planner.planner.Planner.set_sampling_parameters(t_min, horizon, delta_d_min, delta_d_max)`
→ `SamplingHandler.update_static_params(...)`. Density level is separately
controllable via `SamplingHandler.change_max_sampling_level(lvl)` (not used yet).

## Validation

`analysis/smoke_test_sampling.py` (run in Docker with `PYTHONPATH=/app`):
```
ACTION_SHAPE: (7,)
WIDE   step: d in [-3.50, 3.50], n_traj=300   (short & wide)
NARROW step: d in [-0.50, 0.50], n_traj=800   (long & narrow)
RESULT: PASS - sampling grid responds to action
```

## Training

Requires a fresh agent (action space changed → `best_model.zip` incompatible).
Use `colab/train_dynamic_sampling.ipynb` (Python 3.10 via condacolab, GPU torch,
Drive checkpointing, reduced `TOTAL_TIMESTEPS`). Note: workload is CPU-bound
(C++ planner), so GPU helps little; target a proof-of-concept run.

## Evaluation for the report

Compare the trained dynamic-sampling agent vs the fixed-grid baseline on the
same scenarios. Headline metric for the efficiency claim: **mean candidate-
trajectory count / calc-time per step**, alongside success/collision and risk.
