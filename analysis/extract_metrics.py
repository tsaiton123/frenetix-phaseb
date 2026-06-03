"""Extract per-scenario planner metrics from cr_scenario_handler run logs.

Reads each scenario's logs.csv (semicolon-separated, one row per planning
timestep) plus the run-level score_overview.csv, and prints a comparison-ready
metrics summary. Works for both the Default Planner (DP) and Hybrid Planner
(HP) runs -- point --logs-dir at the corresponding logs folder.

No third-party deps (stdlib csv only).
"""
import argparse
import csv
import os
from statistics import mean


def _floats(rows, col):
    out = []
    for r in rows:
        v = r.get(col)
        if v in (None, "", "nan"):
            continue
        try:
            out.append(float(v))
        except ValueError:
            pass
    return out


def read_scores(logs_dir):
    """Map scenario_name -> final status string from score_overview.csv."""
    path = os.path.join(logs_dir, "score_overview.csv")
    status = {}
    if not os.path.exists(path):
        return status
    with open(path) as fh:
        for row in csv.reader(fh, delimiter=";"):
            if len(row) >= 6:
                # name;?;steps;AgentStatus.X;reason;Success/Failure
                status[row[0]] = (row[3], row[5])
    return status


def read_eval_summary(logs_dir):
    """Authoritative per-scenario outcomes if run_phase_b_eval.py wrote them.

    eval_summary.csv format: name;termination_reason;steps;wall_seconds.
    Returns {name: (reason, "Success" / "Failure")}.
    """
    path = os.path.join(logs_dir, "eval_summary.csv")
    out = {}
    if not os.path.exists(path):
        return out
    with open(path) as fh:
        reader = csv.reader(fh, delimiter=";")
        next(reader, None)  # header
        for row in reader:
            if len(row) >= 2:
                name, reason = row[0], row[1]
                success = "Success" if (reason or "").lower().startswith("is_goal_reached_success") else "Failure"
                out[name] = (reason, success)
    return out


# Full T-junction episode runs ~146 steps. Anything notably shorter means the
# scenario terminated early (collision / infeasibility / max_s_position).
# This is the last-resort fallback when no authoritative outcome file is present.
_FULL_EPISODE_STEPS = 140


def status_from_logs_length(timesteps):
    if timesteps is None:
        return ("?", "?")
    if timesteps >= _FULL_EPISODE_STEPS:
        return ("logs>=140", "Success")
    return (f"logs={timesteps}", "Failure")


def scenario_metrics(scenario_dir):
    logs = os.path.join(scenario_dir, "logs.csv")
    if not os.path.exists(logs):
        return None
    with open(logs) as fh:
        rows = list(csv.DictReader(fh, delimiter=";"))
    if not rows:
        return None
    calc = _floats(rows, "calculation_time_s")
    m = {
        "timesteps": len(rows),
        "calc_time_mean_s": mean(calc) if calc else float("nan"),
        "calc_time_max_s": max(calc) if calc else float("nan"),
        "ego_risk_mean": mean(_floats(rows, "ego_risk") or [float("nan")]),
        "ego_risk_max": max(_floats(rows, "ego_risk") or [float("nan")]),
        "obst_risk_mean": mean(_floats(rows, "obst_risk") or [float("nan")]),
        "obst_risk_max": max(_floats(rows, "obst_risk") or [float("nan")]),
        "feasible_pct_mean": mean(_floats(rows, "percentage_feasible_traj") or [float("nan")]),
        "desired_v_mean": mean(_floats(rows, "desired_velocity_mps") or [float("nan")]),
    }
    m.update(_phase_b_action_stats(scenario_dir))
    return m


# Empirical horizon -> candidate-count mapping from analysis/sweep_grid_count.py
# (slope ~333 candidates per second of horizon, intercept ~-200 at planner t_min).
# n_traj ~= 333 * horizon - 200, valid for horizon in [1.5, 3.0] s; otherwise None.
def _estimate_n_traj(horizon: float) -> float:
    if horizon is None:
        return float("nan")
    return max(0.0, 333.0 * horizon - 200.0)


def _phase_b_action_stats(scenario_dir):
    """Parse the per-step Phase B sampling actions from agent_logs.csv.

    The AgentLogger writes ALL action values to each row but its header only
    labels the 5 baseline action columns, so the Phase B horizon (action[5])
    and lateral half-span (action[6]) appear as UNLABELED columns 6 and 7 of
    each row. We parse them positionally; if the row layout is the baseline
    5-dim shape we just report n/a.
    """
    path = os.path.join(scenario_dir, "agent_logs.csv")
    out = {"horizon_mean": float("nan"), "dd_mean": float("nan"),
           "est_n_traj_mean": float("nan"), "is_phase_b": False}
    if not os.path.exists(path):
        return out
    horizons, dds = [], []
    with open(path) as fh:
        next(fh, None)  # header line (no trailing newline in writer)
        for line in fh:
            cols = line.strip().split(";")
            # Phase B row: timestep + 7 actions + ~10-11 rewards >= 18 columns.
            # Baseline row has 5 actions instead and col 6 is a reward.
            if len(cols) < 18:
                return out
            try:
                horizons.append(float(cols[6]))
                dds.append(float(cols[7]))
            except (ValueError, IndexError):
                continue
    if not horizons:
        return out
    out["is_phase_b"] = True
    out["horizon_mean"] = mean(horizons)
    out["dd_mean"] = mean(dds)
    out["est_n_traj_mean"] = _estimate_n_traj(out["horizon_mean"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs-dir", default="logs",
                    help="Run logs dir containing per-scenario subfolders + score_overview.csv")
    ap.add_argument("--label", default="planner", help="Planner label (e.g. DP or HP)")
    args = ap.parse_args()

    # Status sources in priority order:
    #   1. score_overview.csv (cr_scenario_handler authoritative, often missing
    #      when run output_path was overridden post-construction)
    #   2. eval_summary.csv (written by run_phase_b_eval.py — authoritative
    #      per-scenario termination_reason)
    #   3. logs.csv length heuristic (last resort: < ~140 steps = early
    #      termination = failure)
    scores = read_scores(args.logs_dir)
    eval_summary = read_eval_summary(args.logs_dir)
    per = {}
    for name in sorted(os.listdir(args.logs_dir)):
        sdir = os.path.join(args.logs_dir, name)
        if not os.path.isdir(sdir):
            continue
        m = scenario_metrics(sdir)
        if m is None:
            continue
        m["status"] = (scores.get(name)
                       or eval_summary.get(name)
                       or status_from_logs_length(m["timesteps"]))
        per[name] = m

    any_phase_b = any(m["is_phase_b"] for m in per.values())
    print(f"\n=== {args.label} metrics ({len(per)} scenario(s){', Phase B detected' if any_phase_b else ''}) ===")
    for name, m in per.items():
        st = m["status"]
        print(f"\n{name}: {st[1]} ({st[0]})")
        print(f"  timesteps           : {m['timesteps']}")
        print(f"  calc_time mean/max  : {m['calc_time_mean_s']:.4f} / {m['calc_time_max_s']:.4f} s")
        print(f"  ego_risk  mean/max  : {m['ego_risk_mean']:.3e} / {m['ego_risk_max']:.3e}")
        print(f"  obst_risk mean/max  : {m['obst_risk_mean']:.3e} / {m['obst_risk_max']:.3e}")
        print(f"  feasible% mean      : {m['feasible_pct_mean']:.2f}")
        print(f"  desired_v mean      : {m['desired_v_mean']:.3f} m/s")
        if m["is_phase_b"]:
            print(f"  horizon mean        : {m['horizon_mean']:.3f} s")
            print(f"  lat half-span mean  : {m['dd_mean']:.3f} m")
            print(f"  est n_traj mean     : {m['est_n_traj_mean']:.0f}  (back-computed from horizon)")

    if per:
        succ = sum(1 for m in per.values() if m["status"][1].lower().startswith("success"))
        agg = lambda k: mean([m[k] for m in per.values()])
        print(f"\n--- {args.label} aggregate ---")
        print(f"  success rate        : {succ}/{len(per)} = {100*succ/len(per):.1f}%")
        print(f"  mean calc_time      : {agg('calc_time_mean_s'):.4f} s")
        print(f"  mean ego_risk       : {agg('ego_risk_mean'):.3e}")
        print(f"  mean obst_risk      : {agg('obst_risk_mean'):.3e}")
        print(f"  mean feasible%      : {agg('feasible_pct_mean'):.2f}")
        if any_phase_b:
            phase_b_per = [m for m in per.values() if m["is_phase_b"]]
            agg_pb = lambda k: mean([m[k] for m in phase_b_per])
            print(f"  mean horizon        : {agg_pb('horizon_mean'):.3f} s  (full grid = 3.0)")
            print(f"  mean lat half-span  : {agg_pb('dd_mean'):.3f} m  (full grid = 3.5)")
            est = agg_pb("est_n_traj_mean")
            print(f"  mean est n_traj     : {est:.0f}  (full grid = 800; savings vs full = {100*(800-est)/800:.1f}%)")


if __name__ == "__main__":
    main()
