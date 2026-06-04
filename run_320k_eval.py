"""Evaluate the Phase B 320k dynamic-sampling agent (best_model_b_320k) on the
held-out 55-scenario test set, for comparison against the previously-generated
HP-700k baseline result.

best_model_b_320k is a 7-dim (weights_and_sampling) Phase B policy -- the same
action type as run_phase_b_eval.py, so it uses the committed configs.yaml as-is
(no action_type override). It differs only in WHICH checkpoint is loaded and
WHERE outputs go (logs_320k/).

Because this repo ships code-only (no checkpoints), the model zip is resolved
from, in order:
  1. $FRL_320K_ZIP                                  (explicit override)
  2. /content/drive/MyDrive/best_model_b_320k.zip   (Colab: your Drive upload)
  3. <repo>/logs/best_model/best_model_b_320k.zip   (local pre-packed zip)
  4. re-pack <repo>/best_model_b_320k/ on the fly   (local raw SB3 dir)
"""
import csv
import os
import time
import zipfile
from os import listdir
from os.path import isfile, join

from sb3_contrib import RecurrentPPO

from frenetix_rl.gym_environment.paths import PATH_PARAMS
from frenetix_rl.gym_environment.environment.agent_env import AgentEnv
from frenetix_rl.utils.helper_functions import load_environment_configs


def _resolve_320k_zip(mod_path):
    """Return a path to the 320k model zip, packing it from a raw dir if needed."""
    candidates = [
        os.environ.get("FRL_320K_ZIP"),
        "/content/drive/MyDrive/best_model_b_320k.zip",
        os.path.join(mod_path, "logs", "best_model", "best_model_b_320k.zip"),
    ]
    for c in candidates:
        if c and os.path.exists(c):
            return c
    # last resort: re-pack from a raw SB3 directory shipped alongside the repo
    src_dir = os.path.join(mod_path, "best_model_b_320k")
    if os.path.isdir(src_dir):
        zip_path = os.path.join(mod_path, "logs", "best_model", "best_model_b_320k.zip")
        os.makedirs(os.path.dirname(zip_path), exist_ok=True)
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as z:
            for fn in sorted(os.listdir(src_dir)):
                p = os.path.join(src_dir, fn)
                if isfile(p):
                    z.write(p, fn)
        print(f"Packed 320k model from {src_dir} -> {zip_path}")
        return zip_path
    raise FileNotFoundError(
        "Could not find best_model_b_320k.zip. Set $FRL_320K_ZIP, upload it to "
        "/content/drive/MyDrive/best_model_b_320k.zip, or place best_model_b_320k/ "
        "in the repo root."
    )


def main():
    mod_path = os.path.dirname(os.path.abspath(__file__))

    # committed config is already Phase B (weights_and_sampling, 7-dim) -- no override
    env_configs = load_environment_configs(PATH_PARAMS["configs"])

    model_path = _resolve_320k_zip(mod_path)
    model = RecurrentPPO.load(model_path)
    print(f"Loaded model: {model_path} action_space={model.action_space.shape}")

    scen_dir = os.path.join(mod_path, "scenarios_test")
    scen_files = sorted(join(scen_dir, f) for f in listdir(scen_dir)
                        if isfile(join(scen_dir, f)) and f.endswith(".xml"))
    print(f"Test scenarios: {len(scen_files)}")

    test_env = AgentEnv(scenario_paths=scen_files, env_configs=env_configs,
                        test_env=True, plot_agents=False, pick_random_scenario=False)
    test_env.output_path = os.path.join(mod_path, "logs_320k")
    os.makedirs(test_env.output_path, exist_ok=True)
    assert test_env.action_space.shape == model.action_space.shape, \
        f"env {test_env.action_space.shape} vs model {model.action_space.shape}"

    summary_path = os.path.join(test_env.output_path, "eval_summary.csv")
    summary_fh = open(summary_path, "w", buffering=1)
    summary_w = csv.writer(summary_fh, delimiter=";")
    summary_w.writerow(["name", "termination_reason", "steps", "wall_seconds"])

    # Per-timestep timing for the paper's Fig.10 decomposition: RL prediction
    # (predict_s) and the env step (step_s). Overall step = predict_s + step_s;
    # the trajectory-bundle component is logs.csv's calculation_time_s.
    timing_path = os.path.join(test_env.output_path, "timing_steps.csv")
    timing_fh = open(timing_path, "w", buffering=1)
    timing_w = csv.writer(timing_fh, delimiter=";")
    timing_w.writerow(["name", "timestep", "predict_s", "step_s"])

    results = []
    t_run0 = time.perf_counter()
    for idx, scen in enumerate(scen_files, 1):
        name = os.path.basename(scen).replace(".xml", "")
        print(f"\n[{idx}/{len(scen_files)}] {name}")
        t0 = time.perf_counter()
        obs, _ = test_env.reset()
        done = False
        steps = 0
        while not done:
            tp = time.perf_counter()
            action, _ = model.predict(obs)
            predict_s = time.perf_counter() - tp
            ts = time.perf_counter()
            obs, reward, done, truncated, info = test_env.step(action)
            step_s = time.perf_counter() - ts
            timing_w.writerow([name, steps, f"{predict_s:.6f}", f"{step_s:.6f}"])
            steps += 1
            if done:
                reason = info.get("termination_reason", test_env.termination_reason)
                dt = time.perf_counter() - t0
                results.append((name, reason, steps, dt))
                summary_w.writerow([name, reason or "", steps, f"{dt:.2f}"])
                print(f"  -> {reason}  ({steps} steps, {dt:.1f}s)")
    summary_fh.close()
    timing_fh.close()

    total_t = time.perf_counter() - t_run0
    succ = sum(1 for _, r, _, _ in results if r and r.lower().startswith("is_goal_reached_success"))
    coll = sum(1 for _, r, _, _ in results if r and "collision" in (r or "").lower())
    print("\n=== Phase B-320k test-set summary ===")
    print(f"  scenarios: {len(results)}   success: {succ}   collisions: {coll}")
    print(f"  wall:      {total_t/60:.1f} min")


if __name__ == "__main__":
    main()
