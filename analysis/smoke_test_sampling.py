"""Phase B smoke test: verify the dynamic-sampling action reshapes the grid.

Builds AgentEnv with action_type="weights_and_sampling", steps with a "wide"
then a "narrow" lateral action, and checks the lateral spread of the sampled
trajectories responds. Run inside the frenetix-rl Docker image.
"""
import os
import numpy as np

from frenetix_rl.gym_environment.paths import PATH_PARAMS
from frenetix_rl.gym_environment.environment.agent_env import AgentEnv
from frenetix_rl.utils.helper_functions import load_environment_configs

mod = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cfg = load_environment_configs(PATH_PARAMS["configs"])
cfg["action_configs"]["action_type"] = "weights_and_sampling"   # Phase B mode

scen_dir = os.path.join(mod, "scenarios_validation")
scen = [os.path.join(scen_dir, f) for f in os.listdir(scen_dir) if f.endswith(".xml")][:1]

env = AgentEnv(scenario_paths=scen, env_configs=cfg, test_env=True,
               plot_agents=False, pick_random_scenario=False)

print("ACTION_SHAPE:", env.action_space.shape, "(expected (7,))")
assert env.action_space.shape == (7,), "action space not extended to 7"

env.reset()


def d_spread():
    trajs = env.agent.all_trajectories
    ds = [t.sampling_parameters[10] for t in trajs]   # lateral target d per traj
    return (min(ds), max(ds), len(ds))


# action layout: [pred, c1..c4, horizon, dd] normalized in [-1, 1]
wide = np.array([0, 0, 0, 0, 0, -1.0, +1.0])   # short horizon, dd=3.5 (wide)
narrow = np.array([0, 0, 0, 0, 0, +1.0, -1.0])  # long horizon, dd=0.5 (narrow)

env.step(wide)
w = d_spread()
print(f"WIDE   step: d in [{w[0]:.2f}, {w[1]:.2f}], n_traj={w[2]}")

env.step(narrow)
n = d_spread()
print(f"NARROW step: d in [{n[0]:.2f}, {n[1]:.2f}], n_traj={n[2]}")

ok = abs(w[1]) > abs(n[1]) + 0.5   # wide should reach larger |d| than narrow
print("RESULT:", "PASS - sampling grid responds to action" if ok
      else "FAIL - lateral spread did not change")
