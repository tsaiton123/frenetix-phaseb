"""Sweep horizon and lateral half-span independently; log candidate count per step."""
import os
import numpy as np

from frenetix_rl.gym_environment.paths import PATH_PARAMS
from frenetix_rl.gym_environment.environment.agent_env import AgentEnv
from frenetix_rl.utils.helper_functions import load_environment_configs

mod = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cfg = load_environment_configs(PATH_PARAMS["configs"])
cfg["action_configs"]["action_type"] = "weights_and_sampling"

scen_dir = os.path.join(mod, "scenarios_validation")
scen = [os.path.join(scen_dir, f) for f in os.listdir(scen_dir) if f.endswith(".xml")][:1]

env = AgentEnv(scenario_paths=scen, env_configs=cfg, test_env=True,
               plot_agents=False, pick_random_scenario=False)


def norm(value, lo, hi):
    """Inverse rescale: real value -> normalized [-1,1] for the action vector."""
    return (value - (hi + lo) / 2.0) / ((hi - lo) / 2.0)


# from configs: sample_t in [1.5, 3.0], sample_d in [0.5, 3.5]
def make_action(horizon, dd):
    h = norm(horizon, env.sample_t_low, env.sample_t_high)
    d = norm(dd, env.sample_d_low, env.sample_d_high)
    return np.array([0, 0, 0, 0, 0, h, d])


def step_and_count(horizon, dd):
    env.step(make_action(horizon, dd))
    return len(env.agent.all_trajectories)


env.reset()
print(f"{'horizon (s)':<12} {'half-span (m)':<14} {'n_traj':<8} note")
print("-" * 60)

# A) vary horizon, hold lateral half-span at max (3.5 = original)
for h in (1.5, 2.0, 2.5, 3.0):
    n = step_and_count(h, 3.5)
    print(f"{h:<12} {3.5:<14} {n:<8} vary horizon (dd fixed = original 3.5 m)")

# B) vary lateral half-span, hold horizon at max (3.0 = original)
for d in (0.5, 1.5, 2.5, 3.5):
    n = step_and_count(3.0, d)
    print(f"{3.0:<12} {d:<14} {n:<8} vary half-span (horizon fixed = original 3.0 s)")
