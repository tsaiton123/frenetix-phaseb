"""Verify the Phase B trained model loads + action space matches env (7-dim)."""
import os
from sb3_contrib import RecurrentPPO

from frenetix_rl.gym_environment.paths import PATH_PARAMS
from frenetix_rl.gym_environment.environment.agent_env import AgentEnv
from frenetix_rl.utils.helper_functions import load_environment_configs

mod = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cfg = load_environment_configs(PATH_PARAMS["configs"])

# Build env with same config as training/eval will use
scen_dir = os.path.join(mod, "scenarios_test")
scen = [os.path.join(scen_dir, f) for f in sorted(os.listdir(scen_dir)) if f.endswith(".xml")][:1]
env = AgentEnv(scenario_paths=scen, env_configs=cfg, test_env=True,
               plot_agents=False, pick_random_scenario=False)
print(f"ENV action_space: {env.action_space.shape}")

model = RecurrentPPO.load(os.path.join(mod, "logs", "best_model", "best_model.zip"))
print(f"MODEL action_space: {model.action_space.shape}")

ok = env.action_space.shape == model.action_space.shape
print("RESULT:", "PASS - shapes match" if ok else "FAIL - shape mismatch (env vs model)")
