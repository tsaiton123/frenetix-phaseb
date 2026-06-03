# Training the Dynamic-Sampling Agent on a Linux Machine

Phase B = the RL agent shapes the Frenet sampling grid (horizon + lateral span).
Training needs a **fresh** agent (the action space changed, so the authors'
`best_model.zip` is incompatible).

This workload is **CPU-bound** (the C++ Frenet planner steps the env; the LSTM
policy is tiny). A GPU helps only marginally — what matters is **CPU cores**, via
`num_envs` parallel environments. Stack is pinned to **Python 3.10**.

---

## 0. Configure for Phase B (do this first, either path)

Edit `frenetix_rl/gym_environment/configs.yaml`:

```yaml
env_configs:
  training_configs:
    num_envs: 16            # set to ~ number of physical CPU cores (e.g. 16)
    total_timesteps: 2000000  # 2M for a solid run; 7000000 to match the paper
  action_configs:
    action_type: weights_and_sampling   # was: weights
  reward_configs:
    dense_reward:
      reward_sampling_efficiency: -0.0002  # was: 0.
```

(Optional, to match the paper's Table III exactly) in
`frenetix_rl/hyperparams/ppo2.yml` swap the two transposed values:

```yaml
  gamma: 0.99        # was 0.97
  gae_lambda: 0.97   # was 0.99
```

Sanity-check the sampling mechanism before a long run (see §3).

---

## 1. Path A — Docker (recommended, reproducible)

Requires Docker. The provided `Dockerfile` builds the whole stack (Python 3.10 +
frenetix C++ + deps). It installs **CPU** PyTorch, which is fine here.

```bash
cd /path/to/Frenetix-RL

# Build once (~15-30 min first time; compiles the frenetix C++ core)
docker build -t frenetix-rl .

# Train. Mount the repo so edited code + configs are used and logs persist on host.
docker run --rm \
  -v "$PWD":/app \
  frenetix-rl python train.py
```

- Logs/checkpoints land in `./logs` on the host (`best_model/`, `intermediate_model/`).
- TensorBoard data in `./logs_tensorboard`.
- `train.py` runs from `/app` so the mounted source is used (no PYTHONPATH needed).

**GPU (optional, minor benefit):** install the NVIDIA Container Toolkit, change the
torch line in `Dockerfile` to a CUDA wheel
(`pip install torch --index-url https://download.pytorch.org/whl/cu121`),
rebuild, and add `--gpus all` to `docker run`.

**Long runs:** add `-d` to detach, drop `--rm`, and name it
(`--name frl_train`); follow with `docker logs -f frl_train`.

---

## 2. Path B — Native virtualenv (no Docker)

Requires system Python 3.10 and build tools.

```bash
# System deps (Ubuntu/Debian)
sudo apt-get update && sudo apt-get install -y \
    python3.10 python3.10-venv python3.10-dev \
    build-essential cmake libeigen3-dev libboost-all-dev libomp-dev libgl1 libglib2.0-0

cd /path/to/Frenetix-RL
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel

# CPU torch (or a CUDA wheel if you have a GPU)
pip install torch --index-url https://download.pytorch.org/whl/cpu

pip install .
pip install "matplotlib<3.8" "scipy<1.14"

# Train
python train.py
```

---

## 3. Smoke test (verify the sampling action works)

```bash
# Docker:
docker run --rm -v "$PWD":/app -e PYTHONPATH=/app -w /app \
  frenetix-rl python analysis/smoke_test_sampling.py
# Native (from repo root, venv active):
PYTHONPATH="$PWD" python analysis/smoke_test_sampling.py
```

Expect `RESULT: PASS - sampling grid responds to action`.

---

## 4. Monitor

```bash
# from repo root (host)
tensorboard --logdir logs_tensorboard --port 6006
# open http://localhost:6006
```

Watch `rollout/ep_rew_mean` (rising) and the eval success rate. The periodic
`logs/intermediate_model/` checkpoints let you resume or evaluate mid-run.

---

## 5. Evaluate the trained agent

`execute.py` loads `logs/best_model/best_model.zip` and runs the validation
scenarios, then extract metrics:

```bash
# Docker:
docker run --rm -v "$PWD":/app frenetix-rl python execute.py
docker run --rm -v "$PWD":/app frenetix-rl \
  python analysis/extract_metrics.py --logs-dir logs --label "Dynamic-Sampling HP"
# Native:
python execute.py
python analysis/extract_metrics.py --logs-dir logs --label "Dynamic-Sampling HP"
```

Compare against the fixed-grid baseline (`analysis/dp_vs_hp_results.md`). For the
efficiency claim, the headline metric is **mean candidate-trajectory count /
calc-time per step** (lower for the dynamic-sampling agent when it narrows the
grid), alongside success/collision and ego/3rd-party risk.

---

## Rough timing

~80 env-steps/s aggregate on a 16-core machine (paper's setup). So:

| total_timesteps | 16 cores (~paper) |
|---|---|
| 2,000,000 | ~7 h |
| 7,000,000 | ~24 h |

Fewer cores scale roughly linearly slower. GPU does not materially change this.
