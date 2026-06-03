"""Evaluate the 700k-step weights-only ("HP") agent on the held-out 55-scenario
test set, for an extended-budget comparison against the 100k HP baseline and the
Phase B agent.

best_model-700k/ is a 5-dim (weights-only) policy -- action_space=[5], same as
the HP baseline, just trained ~7x longer. So this mirrors run_hp_100k_eval.py:
- Overrides action_type="weights" in-memory (committed configs.yaml stays as
  weights_and_sampling for Phase B); this gives a 5-dim action space.
- Loads logs/best_model/best_model_700k.zip, re-packing it from the raw SB3
  directory best_model-700k/ on the fly if the zip is missing (e.g. on Colab,
  where logs/ is symlinked to Drive and the local zip is invisible).
- Writes outputs to logs_700k/ (isolated from the other runs).
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


def _ensure_700k_zip(mod_path):
    """Return a path to the 700k zip; create it from best_model-700k/ if needed."""
    zip_path = os.path.join(mod_path, "logs", "best_model", "best_model_700k.zip")
    if os.path.exists(zip_path):
        return zip_path
    src_dir = os.path.join(mod_path, "best_model-700k")
    if not os.path.isdir(src_dir):
        raise FileNotFoundError(
            f"Could not find 700k model: neither {zip_path} nor {src_dir} exists."
        )
    os.makedirs(os.path.dirname(zip_path), exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as z:
        for fn in sorted(os.listdir(src_dir)):
            p = os.path.join(src_dir, fn)
            if isfile(p):
                z.write(p, fn)
    print(f"Packed 700k model from {src_dir} -> {zip_path}")
    return zip_path


def main():
    mod_path = os.path.dirname(os.path.abspath(__file__))

    env_configs = load_environment_configs(PATH_PARAMS["configs"])
    env_configs["action_configs"]["action_type"] = "weights"   # 5-dim weights-only

    model_path = _ensure_700k_zip(mod_path)
    model = RecurrentPPO.load(model_path)
    print(f"Loaded model: {model_path} action_space={model.action_space.shape}")

    scen_dir = os.path.join(mod_path, "scenarios_test")
    scen_files = sorted(join(scen_dir, f) for f in listdir(scen_dir)
                        if isfile(join(scen_dir, f)) and f.endswith(".xml"))
    print(f"Test scenarios: {len(scen_files)}")

    test_env = AgentEnv(scenario_paths=scen_files, env_configs=env_configs,
                        test_env=True, plot_agents=False, pick_random_scenario=False)
    test_env.output_path = os.path.join(mod_path, "logs_700k")
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
    print("\n=== 700k test-set summary ===")
    print(f"  scenarios: {len(results)}   success: {succ}   collisions: {coll}")
    print(f"  wall:      {total_t/60:.1f} min")


if __name__ == "__main__":
    main()
