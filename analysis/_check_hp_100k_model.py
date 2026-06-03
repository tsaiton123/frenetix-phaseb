"""Verify the HP-100k model loads + matches a 5-dim (weights) env."""
import os
from sb3_contrib import RecurrentPPO

from frenetix_rl.gym_environment.paths import PATH_PARAMS
from frenetix_rl.gym_environment.environment.agent_env import AgentEnv
from frenetix_rl.utils.helper_functions import load_environment_configs

mod = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cfg = load_environment_configs(PATH_PARAMS["configs"])
cfg["action_configs"]["action_type"] = "weights"

scen_dir = os.path.join(mod, "scenarios_test")
scen = [os.path.join(scen_dir, f) for f in sorted(os.listdir(scen_dir)) if f.endswith(".xml")][:1]
env = AgentEnv(scenario_paths=scen, env_configs=cfg, test_env=True,
               plot_agents=False, pick_random_scenario=False)
print("ENV action_space:", env.action_space.shape)
model = RecurrentPPO.load(os.path.join(mod, "logs", "best_model", "best_model_hp_100k.zip"))
print("MODEL action_space:", model.action_space.shape)
print("RESULT:", "PASS" if env.action_space.shape == model.action_space.shape else "FAIL")
