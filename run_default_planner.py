"""Run the Default Planner (DP) baseline: the analytical Frenet planner with
fixed cost weights and NO reinforcement-learning adaptation.

Mechanism: the RL action is a *weight update* whose rescaling has zero bias
(see AgentEnv.rescale_action / _set_rescale_*_factor_and_bias). Feeding an
all-zero action every step therefore applies a zero update, leaving the
planner pinned at its default cost weights from
configurations/frenetix_motion_planner/cost.yaml. No trained model is loaded.

Runs on the same validation scenarios as execute.py (the HP run) so the two
are directly comparable. Output goes to ./logs_dp so HP logs are untouched.
"""
import os
from os import listdir
from os.path import isfile, join

import numpy as np

from frenetix_rl.gym_environment.paths import PATH_PARAMS
from frenetix_rl.gym_environment.environment.agent_env import AgentEnv
from frenetix_rl.utils.helper_functions import load_environment_configs
from frenetix_rl.evaluation.agent_run_visualization import visualize_agent_run


def run_default_planner():
    mod_path = os.path.dirname(os.path.abspath(__file__))

    env_configs = load_environment_configs(PATH_PARAMS["configs"])

    path_to_scenarios = os.path.join(mod_path, "scenarios_validation")
    scenario_files = [join(path_to_scenarios, f) for f in listdir(path_to_scenarios)
                      if isfile(join(path_to_scenarios, f))]

    # plot_agents=False: we only need logs.csv for metrics; skipping per-step
    # PNG rendering makes the run much faster (does not affect logged calc time).
    test_env = AgentEnv(scenario_paths=scenario_files, env_configs=env_configs,
                        test_env=True, plot_agents=False, pick_random_scenario=False)

    # Redirect output so the Default Planner run does not overwrite the HP logs.
    test_env.output_path = os.path.join(mod_path, "logs_dp")
    os.makedirs(test_env.output_path, exist_ok=True)

    zero_action = np.zeros(test_env.action_space.shape, dtype=np.float64)

    results = []
    for scenario in scenario_files:
        name = os.path.basename(scenario).replace(".xml", "")
        print(f"\n=== Default Planner | Scenario {name} ===")
        obs, _ = test_env.reset()
        done = False
        steps = 0
        while not done:
            obs, reward, done, truncated, info = test_env.step(zero_action)
            steps += 1
            if done:
                reason = info.get("termination_reason", test_env.termination_reason)
                results.append((name, reason, steps))
                print(f"Scenario {name} finished after {steps} steps -> {reason}")
                try:
                    visualize_agent_run(test_env.simulation.log_path, mod_path)
                except Exception as e:
                    print(f"  (skipped run visualization: {e})")

    print("\n=== Default Planner summary ===")
    for name, reason, steps in results:
        print(f"  {name}: {reason} ({steps} steps)")


if __name__ == "__main__":
    run_default_planner()
