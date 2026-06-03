# Frenetix-RL — Phase B (Dynamic-Sampling) Training

A trimmed, self-contained copy of our ECE228 project for **training the Phase B
dynamic-sampling RL agent for more timesteps** on a capable machine.

Phase B = the RL agent shapes the Frenet **sampling grid** (planning horizon +
lateral span), not just the planner's cost weights. The action space is 7-dim
(`weights_and_sampling`). This is an extension of Trauth et al., *RL-Boosted
Motion Planning* (arXiv:2402.01465).

This repo is **code + scenarios only — no trained checkpoints.** You train your
own; `train.py` always starts a fresh agent.

---

## TL;DR — train Phase B for more timesteps

The training budget lives in [frenetix_rl/gym_environment/configs.yaml](frenetix_rl/gym_environment/configs.yaml):

```yaml
env_configs:
  training_configs:
    num_envs: 2               # -> set to ~ your physical CPU core count (e.g. 16)
    total_timesteps: 100000   # -> raise this (e.g. 2000000; 7000000 ~ the paper)
  action_configs:
    action_type: weights_and_sampling   # Phase B (already set — leave it)
```

Then train (pick one path below). Output lands in `logs/` (`best_model/`,
`intermediate_model/` checkpoints, `logs_tensorboard/`).

> **Heads-up:** this is **CPU-bound** — the C++ Frenet planner steps the env; the
> LSTM policy is tiny. A GPU barely helps. What matters is **CPU cores**, exposed
> via `num_envs`. Pinned to **Python 3.10** (the `frenetix` wheel is cp310).

---

## Path A — Docker (most reproducible)

```bash
docker build -t frenetix-rl:exec .          # ~15–30 min first build (compiles C++ core)
docker run --rm -v "$PWD":/app frenetix-rl:exec python train.py
```
Mounting `$PWD` makes your edited configs/code take effect and persists `logs/`
on the host.

## Path B — Native install (Linux x86_64 only)

Requires an **x86_64 Linux** host (workstation, cloud VM, or WSL2) — the
`frenetix` / `commonroad` wheels are not built for Apple Silicon.

```bash
sudo apt-get update && sudo apt-get install -y \
    python3.10 python3.10-venv python3.10-dev \
    build-essential cmake libeigen3-dev libboost-all-dev libomp-dev libgl1 libglib2.0-0
python3.10 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or a CUDA wheel
pip install .
pip install "matplotlib<3.8" "scipy<1.14"
python train.py
```

## Path C — Google Colab (free GPU, no local setup)

Open [colab/train_dynamic_sampling.ipynb](colab/train_dynamic_sampling.ipynb)
and run the cells top to bottom. Bump `total_timesteps` in the config cell first.

---

## Monitor & evaluate

```bash
tensorboard --logdir logs_tensorboard --port 6006     # watch rollout/ep_rew_mean
```
After training, evaluate the agent on the held-out 55-scenario test set:
```bash
python run_phase_b_eval.py                            # (prefix with the docker run … on Path A)
python analysis/extract_metrics.py --logs-dir logs_phase_b --label "Phase B"
```

Rough timing on 16 cores: ~80 env-steps/s ⇒ 2M steps ≈ 7 h, 7M ≈ 24 h. Fewer
cores scale roughly linearly slower.

---

## Layout

| Path | What |
|---|---|
| [train.py](train.py) | Phase B training entry point |
| [frenetix_rl/](frenetix_rl/) | the RL package — env, configs, hyperparams |
| [frenetix_rl/gym_environment/configs.yaml](frenetix_rl/gym_environment/configs.yaml) | **the file to edit** (timesteps, num_envs) |
| [scenarios/](scenarios/) · [scenarios_validation/](scenarios_validation/) · [scenarios_test/](scenarios_test/) | 410 / 82 / 55 T-junction scenarios (75/15/10 split, `split_manifest.json`) |
| [run_phase_b_eval.py](run_phase_b_eval.py) | evaluate a trained agent on the test set |
| [analysis/](analysis/) · [scripts/](scripts/) | metrics, comparison, scenario download/split |
| [colab/](colab/) | Colab training notebook (Path C) |
| [REPRODUCE.md](REPRODUCE.md) · [TRAINING_LINUX.md](TRAINING_LINUX.md) | full reproduction & Linux guides |
| [README_UPSTREAM.md](README_UPSTREAM.md) · [LICENSE](LICENSE) | original authors' README (TUM-AVS) & license (LGPL-3.0) |

Based on [TUM-AVS/Frenetix-RL](https://github.com/TUM-AVS/Frenetix-RL) (LGPL-3.0).
