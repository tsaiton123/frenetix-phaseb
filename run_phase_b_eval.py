"""Evaluate the trained Phase B (dynamic-sampling) agent on the held-out test
set (55 ZAM_Tjunction scenarios per split_manifest.json).

Mirrors execute.py but: (a) reads from scenarios_test/, (b) writes to logs_phase_b/
to keep results isolated, (c) plotting disabled for speed (we only need logs.csv
metrics, not per-step PNGs).
"""
import csv
import os
import time
from os import listdir
from os.path import isfile, join

from sb3_contrib import RecurrentPPO

from frenetix_rl.gym_environment.paths import PATH_PARAMS
from frenetix_rl.gym_environment.environment.agent_env import AgentEnv
from frenetix_rl.utils.helper_functions import load_environment_configs


def main():
    mod_path = os.path.dirname(os.path.abspath(__file__))

    env_configs = load_environment_configs(PATH_PARAMS["configs"])

    model_path = os.path.join(mod_path, "logs", "best_model", "best_model.zip")
    model = RecurrentPPO.load(model_path)
    print(f"Loaded model: {model_path} action_space={model.action_space.shape}")

    scen_dir = os.path.join(mod_path, "scenarios_test")
    scen_files = sorted(join(scen_dir, f) for f in listdir(scen_dir)
                        if isfile(join(scen_dir, f)) and f.endswith(".xml"))
    print(f"Test scenarios: {len(scen_files)}")

    test_env = AgentEnv(scenario_paths=scen_files, env_configs=env_configs,
                        test_env=True, plot_agents=False, pick_random_scenario=False)
    test_env.output_path = os.path.join(mod_path, "logs_phase_b")
    os.makedirs(test_env.output_path, exist_ok=True)

    # Authoritative per-scenario outcome CSV -- written incrementally so a
    # disconnect/crash mid-run still leaves valid partial data the extractor
    # can read.
    summary_path = os.path.join(test_env.output_path, "eval_summary.csv")
    summary_fh = open(summary_path, "w", buffering=1)  # line-buffered
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
    n_succ = sum(1 for *_, r, _, _ in [(None,) + r for r in results]
                 if r is not None and r.lower().startswith("is_goal_reached_success"))
    # simpler counts
    succ = sum(1 for _, r, _, _ in results if r and r.lower().startswith("is_goal_reached_success"))
    coll = sum(1 for _, r, _, _ in results if r and "collision" in (r or "").lower())
    print("\n=== Phase B test-set summary ===")
    print(f"  scenarios     : {len(results)}")
    print(f"  success       : {succ}")
    print(f"  collisions    : {coll}")
    print(f"  other failures: {len(results) - succ - coll}")
    print(f"  total wall    : {total_t/60:.1f} min")
    for name, reason, steps, dt in results:
        print(f"    {name}: {reason} ({steps} steps, {dt:.1f}s)")


if __name__ == "__main__":
    main()
