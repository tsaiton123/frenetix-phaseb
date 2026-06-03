"""Re-run the Hybrid Planner (HP) under the SAME conditions as
run_default_planner.py so calc-time is directly comparable: same Docker image,
plotting disabled, output to ./logs_hp (originals in ./logs untouched).

Loads the authors' pre-trained RecurrentPPO model (best_model.zip) and lets it
adjust the planner cost weights each step (the RL-boosted hybrid planner).
"""
import os
from os import listdir
from os.path import isfile, join

from sb3_contrib import RecurrentPPO

from frenetix_rl.gym_environment.paths import PATH_PARAMS
from frenetix_rl.gym_environment.environment.agent_env import AgentEnv
from frenetix_rl.utils.helper_functions import load_environment_configs
from frenetix_rl.evaluation.agent_run_visualization import visualize_agent_run


def run_hybrid_planner():
    mod_path = os.path.dirname(os.path.abspath(__file__))

    env_configs = load_environment_configs(PATH_PARAMS["configs"])

    path_to_model = os.path.join(mod_path, "logs", "best_model", "best_model.zip")
    model = RecurrentPPO.load(path_to_model)

    path_to_scenarios = os.path.join(mod_path, "scenarios_validation")
    scenario_files = [join(path_to_scenarios, f) for f in listdir(path_to_scenarios)
                      if isfile(join(path_to_scenarios, f))]

    # plot_agents=False to match the DP run conditions (fair timing).
    test_env = AgentEnv(scenario_paths=scenario_files, env_configs=env_configs,
                        test_env=True, plot_agents=False, pick_random_scenario=False)

    # Separate output dir so the original May-21 HP logs in ./logs are preserved.
    test_env.output_path = os.path.join(mod_path, "logs_hp")
    os.makedirs(test_env.output_path, exist_ok=True)

    results = []
    for scenario in scenario_files:
        name = os.path.basename(scenario).replace(".xml", "")
        print(f"\n=== Hybrid Planner | Scenario {name} ===")
        obs, _ = test_env.reset()
        done = False
        steps = 0
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, truncated, info = test_env.step(action)
            steps += 1
            if done:
                reason = info.get("termination_reason", test_env.termination_reason)
                results.append((name, reason, steps))
                print(f"Scenario {name} finished after {steps} steps -> {reason}")
                try:
                    visualize_agent_run(test_env.simulation.log_path, mod_path)
                except Exception as e:
                    print(f"  (skipped run visualization: {e})")

    print("\n=== Hybrid Planner summary ===")
    for name, reason, steps in results:
        print(f"  {name}: {reason} ({steps} steps)")


if __name__ == "__main__":
    run_hybrid_planner()
