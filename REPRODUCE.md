# Reproducing All Results Locally

End-to-end guide to regenerate every result in this project from scratch on a
local machine. Companion to [TRAINING_LINUX.md](TRAINING_LINUX.md) (which focuses
only on Phase B training) and the Colab notebook
[colab/train_dynamic_sampling.ipynb](colab/train_dynamic_sampling.ipynb) (the
GPU-cloud training path).

> **Why Docker.** The planner stack (`frenetix` C++ core, `cr_scenario_handler`,
> `commonroad-*`) is pinned to **Python 3.10** and is **not** installed natively
> on the Mac. Everything below runs inside the `frenetix-rl:exec` image. The
> workload is **CPU-bound** (the C++ planner steps the env; the LSTM policy is
> tiny), so a GPU barely helps — what matters is CPU cores via `num_envs`.
>
> **No Docker?** See [§10 — Running without Docker](#10-running-without-docker).
> Short version: on this Apple-Silicon (arm64) Mac the planner *cannot* run
> natively (no arm64 wheels for `frenetix` / `commonroad-drivability-checker`),
> so the heavy steps go to **Colab** or a **Linux x86_64** box. The pure-Python
> analysis scripts do run on the host with no install.

---

## 0. Build the image (once)

```bash
cd /Users/tsaiyuntong/Documents/homeworks/ece228/final/Frenetix-RL
docker build -t frenetix-rl:exec .          # ~15–30 min first time (compiles C++ core)
```

### The run wrapper

Every command below uses this pattern. Mounting `$PWD` over `/app` means your
**edited code + configs + logs persist on the host**, and overrides the stale
copy baked into the image.

```bash
# define once per shell
frx() { docker run --rm -v "$PWD":/app -w /app -e PYTHONPATH=/app frenetix-rl:exec "$@"; }
```

> **Gotcha — `PYTHONPATH=/app -w /app` is mandatory** for any script under
> `analysis/` or `scripts/`. Without it, Python imports the stale `frenetix_rl`
> baked into the image's dist-packages instead of your mounted edits.

---

## 1. Data — download & split the scenarios

The repo already ships the split (`scenarios/`, `scenarios_validation/`,
`scenarios_test/`) and `split_manifest.json`, so **skip this section unless you
want to regenerate the split.** Counts: **410 train / 82 val / 55 test** (75/15/10
of 547 `ZAM_Tjunction` scenarios, seed 42).

```bash
# 1) download the 547 T-junction scenarios into scenarios_pool/
frx python scripts/download_scenarios.py            # add --workers 16 to parallelize

# 2) deterministic 75/15/10 split -> scenarios/, scenarios_validation/, scenarios_test/
frx python scripts/split_scenarios.py --clean       # writes split_manifest.json
```

---

## 2. Models inventory

| File / dir | Action dims | `action_type` | How produced |
|---|---|---|---|
| `logs/best_model/best_model_authors.zip` | 5 (weights) | `weights` | shipped — authors' pre-trained HP |
| `best_model_hp/` → `best_model_hp_100k.zip` | 5 (weights) | `weights` | `train.py`, 100k steps, fair-budget HP baseline |
| `best_model-700k/` → `best_model_700k.zip` | 5 (weights) | `weights` | `train.py`, ~700k steps, extended-budget HP |
| `best_model_b/`, `best_model_phase_b.zip`, `best_model.zip` | 7 (weights+sampling) | `weights_and_sampling` | `train.py` Phase B, 100k steps |

The raw SB3 directories (`best_model_hp/`, `best_model-700k/`, `best_model_b/`)
are unzipped checkpoints. The eval scripts re-pack them into
`logs/best_model/*.zip` on the fly if the zip is missing (important on Colab,
where `logs/` is a Drive symlink). `logs/` is gitignored — the `best_model*/`
dirs are the source of truth.

---

## 3. The config toggle (read before training/eval)

The committed [frenetix_rl/gym_environment/configs.yaml](frenetix_rl/gym_environment/configs.yaml)
is set for **Phase B** (`action_type: weights_and_sampling`, 7-dim,
`reward_sampling_efficiency: -0.0002`, `total_timesteps: 100000`, `num_envs: 2`).

- **Phase B** runs use the committed file as-is.
- **HP / weights-only** *training* needs `action_type: weights` in `configs.yaml`.
- **HP / weights-only** *eval* scripts (`run_hp_100k_eval.py`, `run_700k_eval.py`)
  override `action_type="weights"` **in memory**, so they work regardless of the
  committed value — no edit needed.

Bump `num_envs` to ~your physical core count to train faster.

---

## 4. Training

Outputs land in `logs/best_model/best_model.zip` (+ `logs/intermediate_model/`
checkpoints, `logs_tensorboard/`). **`train.py` always overwrites
`best_model.zip`**, so rename/move it after each run before training the next
variant.

### 4a. Phase B (dynamic-sampling agent) — committed config
```bash
frx python train.py
cp logs/best_model/best_model.zip logs/best_model/best_model_phase_b.zip
```

### 4b. HP-100k baseline (fixed grid, weights-only)
Set `action_type: weights` in `configs.yaml` (keep `total_timesteps: 100000`), then:
```bash
frx python train.py
# the raw checkpoint dir best_model_hp/ is this run's best_model/ contents
```

### 4c. HP-700k baseline (extended budget)
Set `action_type: weights` and `total_timesteps: 700000`, then `frx python train.py`.
(Best done on Colab GPU/long-CPU — see the notebook. The result is `best_model-700k/`.)

> Rough timing on 16 cores (~paper): 2M steps ≈ 7 h, 7M ≈ 24 h. Restore the
> committed `configs.yaml` (`weights_and_sampling`) when done.

### Monitor
```bash
tensorboard --logdir logs_tensorboard --port 6006   # on the host; open localhost:6006
```
Watch `rollout/ep_rew_mean` rising and the eval success rate.

---

## 5. Evaluation on the held-out test set (55 scenarios)

Each writes per-scenario outcomes (`eval_summary.csv`), per-step timing
(`timing_steps.csv`), and planner logs into its own `logs_*/` folder.

```bash
# Phase B agent (7-dim) -> logs_phase_b/         [loads logs/best_model/best_model.zip]
frx python run_phase_b_eval.py

# HP-100k baseline (5-dim) -> logs_hp_100k/       [loads/packs best_model_hp]
frx python run_hp_100k_eval.py

# HP-700k baseline (5-dim) -> logs_700k/          [loads/packs best_model-700k]
frx python run_700k_eval.py
```

---

## 6. Planner baselines: Default vs Hybrid (validation set)

The original DP-vs-HP reproduction (see [analysis/dp_vs_hp_results.md](analysis/dp_vs_hp_results.md)):
DP collides in the T-junction, the RL-boosted HP avoids it. These load the
authors' 5-dim model, so **set `action_type: weights` in `configs.yaml` first.**

```bash
frx python run_default_planner.py    # zero-action no-op -> logs_dp/
frx python run_hybrid_planner.py     # authors' best_model.zip adapts weights -> logs_hp/
```

---

## 7. Smoke tests & sanity checks

```bash
# Phase B sampling action actually reshapes the grid (expect "RESULT: PASS")
frx python analysis/smoke_test_sampling.py

# candidate-count sweep over horizon / lateral half-span
frx python analysis/sweep_grid_count.py

# verify a model zip matches its env's action shape
frx python analysis/_check_hp_100k_model.py     # 5-dim HP
frx python analysis/_check_phase_b_model.py     # 7-dim Phase B
```

---

## 8. Metrics & comparison (post-hoc, no replanning)

`extract_metrics.py` summarizes one run; `compare_efficiency.py` puts several
side-by-side, including the paper's Fig.10 timing decomposition. Both are pure
stdlib over the logged CSVs.

```bash
# single-run summaries
frx python analysis/extract_metrics.py --logs-dir logs_phase_b --label "Phase B (dynamic sampling)"
frx python analysis/extract_metrics.py --logs-dir logs_hp_100k --label "HP-100k (fixed grid)"
frx python analysis/extract_metrics.py --logs-dir logs_700k    --label "HP-700k (fixed grid)"
frx python analysis/extract_metrics.py --logs-dir logs_dp      --label "DP"
frx python analysis/extract_metrics.py --logs-dir logs_hp      --label "HP"

# headline efficiency + safety comparison (the key Phase B claim)
frx python analysis/compare_efficiency.py \
    --run "HP-100k:logs_hp_100k" \
    --run "HP-700k:logs_700k" \
    --run "PhaseB:logs_phase_b"
```

If you ran eval on Colab and downloaded the Drive folders locally, point
`--run` / `--logs-dir` at those instead (e.g.
`frl_hp_100k_logs_test`, `frl_phaseB_logs_test`, `frl_700k_logs_test`).

---

## 9. Output map

| Folder | Produced by | Holds |
|---|---|---|
| `logs/` | `train.py` | `best_model/`, `intermediate_model/`, eval, `logs_tensorboard/` |
| `logs_phase_b/` | `run_phase_b_eval.py` | Phase B test-set logs, `eval_summary.csv`, `timing_steps.csv` |
| `logs_hp_100k/` | `run_hp_100k_eval.py` | HP-100k test-set logs |
| `logs_700k/` | `run_700k_eval.py` | HP-700k test-set logs |
| `logs_dp/` | `run_default_planner.py` | Default Planner validation logs |
| `logs_hp/` | `run_hybrid_planner.py` | Hybrid Planner validation logs |

Result write-ups already in the repo:
[analysis/dp_vs_hp_results.md](analysis/dp_vs_hp_results.md),
[analysis/phase_b_implementation.md](analysis/phase_b_implementation.md),
[analysis/phase_b_vs_paper.md](analysis/phase_b_vs_paper.md).

---

## 10. Running without Docker

There is **no native planner path on this arm64 Mac.** `frenetix==0.1.3` publishes
wheels only for macOS **x86_64** (Intel) and Linux; `commonroad-drivability-checker`
only for manylinux x86_64. No Apple-Silicon wheels exist, and the sdist needs the
full C++ toolchain to build. That is why the project is Docker-only locally.

Your two real no-Docker options for the heavy steps (train / eval / planners /
smoke tests):

### Option A — Google Colab (recommended, no Docker, free GPU)
Use [colab/train_dynamic_sampling.ipynb](colab/train_dynamic_sampling.ipynb). It
installs Python 3.10 via condacolab on Colab's Linux x86_64 (where the wheels
exist), then runs train + the three test-set evals + the efficiency comparison.
Cell map: 1–8 setup & train, 9 Phase B eval, 9b HP-100k, 9d HP-700k, 9c compare.
Results land in your Drive (`frl_phaseB_logs_test/`, `frl_hp_100k_logs_test/`,
`frl_700k_logs_test/`).

### Option B — Native install on a Linux x86_64 machine (full training)

Works because the manylinux wheels resolve. This is the path to run **everything**
(train / eval / planners / analysis) natively, no Docker. Apple Silicon won't work
— you need an x86_64 Linux host (a workstation, a cloud VM like AWS/GCP, a lab
box, or WSL2 on Windows). A multi-core CPU matters far more than a GPU here
(training is CPU-bound on the C++ planner).

**1. System dependencies** (Ubuntu/Debian; adjust for other distros):
```bash
sudo apt-get update && sudo apt-get install -y \
    python3.10 python3.10-venv python3.10-dev \
    build-essential cmake libeigen3-dev libboost-all-dev libomp-dev libgl1 libglib2.0-0
```
> Python **3.10 specifically** — the `frenetix` wheel is cp310. On a distro
> without a 3.10 package (e.g. Ubuntu 24.04), get it via `deadsnakes` PPA,
> `pyenv install 3.10`, or `conda create -n frl python=3.10`.

**2. Virtualenv + install:**
```bash
cd Frenetix-RL
python3.10 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU build (fine; CPU-bound)
# GPU box: instead use a CUDA wheel, e.g.
# pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install .
pip install "matplotlib<3.8" "scipy<1.14"
```

**3. Verify the stack imports** (fast sanity check before a long run):
```bash
python -c "import frenetix, cr_scenario_handler, sb3_contrib, torch; print('stack OK')"
python analysis/smoke_test_sampling.py        # expect: RESULT: PASS
```

**4. Run the whole pipeline.** Every `frx python …` command in §1–§8 works
verbatim if you just define a passthrough shim (or drop the `frx ` prefix):
```bash
frx() { python "$@"; }          # native passthrough — now §1–§8 commands run as-is
```
So, end to end on the Linux box:
```bash
# --- data (already shipped; regenerate only if needed) ---
# frx python scripts/download_scenarios.py
# frx python scripts/split_scenarios.py --clean

# --- tune for the machine: edit frenetix_rl/gym_environment/configs.yaml ---
#   num_envs: <~physical core count>     (e.g. 16)
#   action_type / total_timesteps per §3–§4 for each variant

# --- train (rename best_model.zip between variants; train.py overwrites it) ---
frx python train.py                                   # Phase B (committed config)
cp logs/best_model/best_model.zip logs/best_model/best_model_phase_b.zip
# HP-100k: set action_type: weights, total_timesteps: 100000 -> frx python train.py
# HP-700k: set action_type: weights, total_timesteps: 700000 -> frx python train.py

# --- evaluate on the held-out test set ---
frx python run_phase_b_eval.py        # -> logs_phase_b/
frx python run_hp_100k_eval.py        # -> logs_hp_100k/
frx python run_700k_eval.py           # -> logs_700k/

# --- planner baselines (set action_type: weights first; §6) ---
frx python run_default_planner.py     # -> logs_dp/
frx python run_hybrid_planner.py      # -> logs_hp/

# --- metrics + comparison ---
frx python analysis/compare_efficiency.py \
    --run "HP-100k:logs_hp_100k" --run "HP-700k:logs_700k" --run "PhaseB:logs_phase_b"
```
> **No `PYTHONPATH`/`-w` needed** natively (those only fixed the Docker image's
> stale baked-in copy) — just stay in the repo root so the local `frenetix_rl`
> is on `sys.path`. Logs/checkpoints land in the same `logs*/` folders as §9.

**Long / detached runs:** `nohup frx python train.py > train.out 2>&1 &` and watch
with `tail -f train.out`; or use `tmux`/`screen`. Monitor live with
`tensorboard --logdir logs_tensorboard --port 6006`. Rough timing: ~80 env-steps/s
on 16 cores ⇒ 2M steps ≈ 7 h, 7M ≈ 24 h; fewer cores scale ~linearly slower.

### What DOES run natively on the Mac (no Docker, no install)
These four scripts are pure stdlib — run them with the host `python3.10`
straight from the repo root. Use them for the post-hoc analysis on logs you bring
back from Colab/Linux:
```bash
python3.10 scripts/download_scenarios.py            # fetch scenarios (urllib)
python3.10 scripts/split_scenarios.py --clean       # 75/15/10 split
python3.10 analysis/extract_metrics.py --logs-dir frl_phaseB_logs_test --label "Phase B"
python3.10 analysis/compare_efficiency.py \
    --run "HP-100k:frl_hp_100k_logs_test" \
    --run "HP-700k:frl_700k_logs_test" \
    --run "PhaseB:frl_phaseB_logs_test"
```
(`extract_metrics.py` / `compare_efficiency.py` only read the logged CSVs; the
`scripts/` ones only use urllib/shutil. None import the planner stack.)

---

## TL;DR — full local run, fresh

```bash
docker build -t frenetix-rl:exec .
frx() { docker run --rm -v "$PWD":/app -w /app -e PYTHONPATH=/app frenetix-rl:exec "$@"; }

# (data already shipped; regenerate only if needed)
# frx python scripts/download_scenarios.py && frx python scripts/split_scenarios.py --clean

# train (edit configs.yaml between variants per §3/§4), then:
frx python train.py                                  # Phase B -> best_model.zip

# evaluate on the test set
frx python run_phase_b_eval.py
frx python run_hp_100k_eval.py
frx python run_700k_eval.py

# compare
frx python analysis/compare_efficiency.py \
    --run "HP-100k:logs_hp_100k" --run "HP-700k:logs_700k" --run "PhaseB:logs_phase_b"
```
