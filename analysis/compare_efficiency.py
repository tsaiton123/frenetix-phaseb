"""Side-by-side computational-efficiency + safety comparison of two (or more)
eval runs, e.g. the fixed-grid HP-100k baseline vs the dynamic-sampling Phase B
agent, on the same held-out test set.

Reports every metric BOTH ways so the aggregation choice is explicit:
  * micro-average: pool every per-timestep row across all scenarios, weight
    each timestep equally (long scenarios contribute more).
  * macro-average: reduce each scenario to one value, then average the
    scenarios, weight each scenario equally (matches the paper's Fig. 5 style).

Metrics, and where each comes from (see run_phase_b_eval.py / cr_scenario_handler logs):
  - success rate        : eval_summary.csv  (termination_reason startswith is_goal_reached_success)
  - calc_time_s         : logs.csv col `calculation_time_s`     (planner compute / timestep)
  - n_traj (sampled)    : reconstructed = infeasible_sum / (1 - %feasible/100)  [both runs]
  - n_traj (horizon est): agent_logs.csv action[5] -> 333*h-200  [Phase B only; cross-check]
  - ego_risk / obst_risk: logs.csv cols `ego_risk`,`obst_risk`  (paper Eq. 3 proxy)

No third-party deps (stdlib csv only).

Usage:
  python analysis/compare_efficiency.py \
      --run "HP-100k:logs_hp_100k" --run "PhaseB:logs_phase_b"
  # locally, against the downloaded Drive folders:
  python analysis/compare_efficiency.py \
      --run "HP-100k:frl_hp_100k_logs_test" --run "PhaseB:frl_phaseB_logs_test"
"""
import argparse
import csv
import glob
import os
from statistics import mean


def _floats(rows, col):
    out = []
    for r in rows:
        v = r.get(col)
        if v in (None, "", "nan", "NaN"):
            continue
        try:
            f = float(v)
        except ValueError:
            continue
        if f == f:  # drop NaN
            out.append(f)
    return out


def _safe_mean(xs):
    return mean(xs) if xs else float("nan")


# horizon -> candidate-count fit from analysis/sweep_grid_count.py (Phase B only).
def _est_n_traj(horizon):
    if horizon is None or horizon != horizon:
        return float("nan")
    return max(0.0, 333.0 * horizon - 200.0)


def _phase_b_horizon_mean(scenario_dir):
    """Mean per-step planning horizon (action[5]) from agent_logs.csv, or nan.

    AgentLogger labels only the 5 baseline action columns, so the Phase B
    horizon lands in unlabeled positional column 6 (0-indexed). Baseline
    (weights-only) rows have <18 columns -> return nan.
    """
    path = os.path.join(scenario_dir, "agent_logs.csv")
    if not os.path.exists(path):
        return float("nan")
    horizons = []
    with open(path) as fh:
        next(fh, None)  # header (no trailing newline from the writer)
        for line in fh:
            cols = line.strip().split(";")
            if len(cols) < 18:
                return float("nan")
            try:
                horizons.append(float(cols[6]))
            except (ValueError, IndexError):
                continue
    return _safe_mean(horizons)


def read_eval_summary(run_dir):
    """name -> (reason, is_success bool) from eval_summary.csv."""
    path = os.path.join(run_dir, "eval_summary.csv")
    out = {}
    if not os.path.exists(path):
        return out
    with open(path) as fh:
        reader = csv.reader(fh, delimiter=";")
        next(reader, None)
        for row in reader:
            if len(row) >= 2:
                reason = row[1] or ""
                out[row[0]] = (reason,
                               reason.lower().startswith("is_goal_reached_success"))
    return out


def scenario_rows(scenario_dir):
    path = os.path.join(scenario_dir, "logs.csv")
    if not os.path.exists(path):
        return []
    with open(path) as fh:
        return list(csv.DictReader(fh, delimiter=";"))


def read_timing_steps(run_dir):
    """Per-timestep (predict_s, step_s) written by the instrumented eval loops.

    Returns ([predict_s...], [step_s...]) pooled over all timesteps, or empty
    lists if the run predates the timing instrumentation. These feed the paper's
    Fig.10 decomposition: RL prediction = predict_s, overall step = predict_s +
    step_s (the bundle component is logs.csv's calculation_time_s).
    """
    path = os.path.join(run_dir, "timing_steps.csv")
    predict, step = [], []
    if not os.path.exists(path):
        return predict, step
    with open(path) as fh:
        reader = csv.DictReader(fh, delimiter=";")
        for r in reader:
            try:
                predict.append(float(r["predict_s"]))
                step.append(float(r["step_s"]))
            except (KeyError, ValueError, TypeError):
                continue
    return predict, step


def _bundle_sizes(rows):
    """Per-timestep reconstructed sampled-trajectory count.

    total = infeasible / (1 - feasible_fraction). Rows with 100% feasible
    (no infeasible to anchor on) can't be inverted and are skipped -- this
    slightly under-counts the easiest timesteps, so treat n_traj as an
    estimate, not an exact census.
    """
    out = []
    for r in rows:
        try:
            pf = float(r["percentage_feasible_traj"]) / 100.0
            inf = float(r["infeasible_sum"])
        except (KeyError, ValueError, TypeError):
            continue
        if pf < 1.0:
            out.append(inf / (1.0 - pf))
    return out


def collect(run_dir):
    """Return (per_scenario list of dicts, micro pools dict)."""
    summary = read_eval_summary(run_dir)
    per = []
    micro = {"calc": [], "n_traj": [], "ego": [], "obst": []}
    micro["predict"], micro["step"] = read_timing_steps(run_dir)
    for sdir in sorted(glob.glob(os.path.join(run_dir, "ZAM*"))):
        name = os.path.basename(sdir)
        rows = scenario_rows(sdir)
        if not rows:
            continue
        calc = _floats(rows, "calculation_time_s")
        ntj = _bundle_sizes(rows)
        ego = _floats(rows, "ego_risk")
        obst = _floats(rows, "obst_risk")
        micro["calc"] += calc
        micro["n_traj"] += ntj
        micro["ego"] += ego
        micro["obst"] += obst
        reason, succ = summary.get(name, ("", None))
        per.append({
            "name": name,
            "success": succ,                      # None if no eval_summary
            "calc_s": _safe_mean(calc),
            "n_traj": _safe_mean(ntj),
            "ego_mean": _safe_mean(ego),
            "ego_max": max(ego) if ego else float("nan"),
            "obst_mean": _safe_mean(obst),
            "obst_max": max(obst) if obst else float("nan"),
            "horizon": _phase_b_horizon_mean(sdir),
        })
    return per, micro


def summarize(label, per, micro):
    n = len(per)
    succ = sum(1 for p in per if p["success"])
    have_succ = any(p["success"] is not None for p in per)
    horizons = [p["horizon"] for p in per if p["horizon"] == p["horizon"]]

    def macro(key):
        xs = [p[key] for p in per if p[key] == p[key]]
        return _safe_mean(xs)

    out = {
        "label": label, "n": n,
        "success": (succ, n) if have_succ else None,
        # micro = pool all timesteps; macro = mean of per-scenario means
        "calc_micro_ms": _safe_mean(micro["calc"]) * 1000,
        "calc_macro_ms": macro("calc_s") * 1000,
        "ntraj_micro": _safe_mean(micro["n_traj"]),
        "ntraj_macro": macro("n_traj"),
        "ego_micro": _safe_mean(micro["ego"]),
        "ego_macro_mean": macro("ego_mean"),
        "ego_macro_max": macro("ego_max"),    # per-scenario max then mean (Fig.5 style)
        "obst_micro": _safe_mean(micro["obst"]),
        "obst_macro_mean": macro("obst_mean"),
        "obst_macro_max": macro("obst_max"),
        "horizon_est_ntraj": _est_n_traj(_safe_mean(horizons)) if horizons else float("nan"),
        # Fig.10 decomposition (micro, ms). overall = predict + step; bundle is
        # the calculation_time_s already captured above. nan if not instrumented.
        "fig10_predict_ms": _safe_mean(micro["predict"]) * 1000 if micro["predict"] else float("nan"),
        "fig10_step_ms": _safe_mean(micro["step"]) * 1000 if micro["step"] else float("nan"),
        "fig10_bundle_ms": _safe_mean(micro["calc"]) * 1000,
        "fig10_overall_ms": (_safe_mean(micro["predict"]) + _safe_mean(micro["step"])) * 1000
                            if micro["predict"] and micro["step"] else float("nan"),
    }
    return out


def _pct(a, b):
    if b == 0 or b != b or a != a:
        return "n/a"
    return f"{100 * (a - b) / b:+.1f}%"


def print_run(s):
    print(f"\n=== {s['label']}  ({s['n']} scenarios) ===")
    if s["success"]:
        sc, n = s["success"]
        print(f"  success rate           : {sc}/{n} = {100*sc/n:.1f}%")
    print(f"  calc_time / iter  micro : {s['calc_micro_ms']:.1f} ms   macro: {s['calc_macro_ms']:.1f} ms")
    print(f"  n_traj sampled    micro : {s['ntraj_micro']:.0f}        macro: {s['ntraj_macro']:.0f}"
          + (f"   (horizon-est: {s['horizon_est_ntraj']:.0f})" if s['horizon_est_ntraj'] == s['horizon_est_ntraj'] else ""))
    print(f"  ego_risk   micro : {s['ego_micro']:.3e}   macro(mean): {s['ego_macro_mean']:.3e}   macro(max): {s['ego_macro_max']:.3e}")
    print(f"  obst_risk  micro : {s['obst_micro']:.3e}   macro(mean): {s['obst_macro_mean']:.3e}   macro(max): {s['obst_macro_max']:.3e}")
    if s["fig10_overall_ms"] == s["fig10_overall_ms"]:  # instrumented run
        print(f"  -- Fig.10 decomposition (ms/iter, micro) --")
        print(f"     RL prediction       : {s['fig10_predict_ms']:.3f}")
        print(f"     trajectory bundle   : {s['fig10_bundle_ms']:.1f}   (= calculation_time_s)")
        print(f"     env step (w/ bundle): {s['fig10_step_ms']:.1f}")
        print(f"     overall model step  : {s['fig10_overall_ms']:.1f}   (= predict + step)")
    else:
        print(f"  -- Fig.10: only trajectory bundle available ({s['fig10_bundle_ms']:.1f} ms);"
              f" re-run eval with timing_steps.csv for RL-prediction + overall --")


def print_delta(a, b):
    """Delta of b relative to a (a = baseline/HP, b = PhaseB)."""
    print(f"\n=== {b['label']} vs {a['label']} (relative to {a['label']}) ===")
    if a["success"] and b["success"]:
        ra = 100 * a["success"][0] / a["success"][1]
        rb = 100 * b["success"][0] / b["success"][1]
        print(f"  success rate      : {ra:.1f}% -> {rb:.1f}%  ({rb-ra:+.1f} pp)")
    rows = [
        ("calc_time micro", "calc_micro_ms"), ("calc_time macro", "calc_macro_ms"),
        ("n_traj micro", "ntraj_micro"), ("n_traj macro", "ntraj_macro"),
        ("ego_risk micro", "ego_micro"), ("ego_risk macro(mean)", "ego_macro_mean"),
        ("obst_risk micro", "obst_micro"), ("obst_risk macro(mean)", "obst_macro_mean"),
    ]
    for lbl, k in rows:
        print(f"  {lbl:22}: {a[k]:.3g} -> {b[k]:.3g}  ({_pct(b[k], a[k])})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, metavar="LABEL:DIR",
                    help="Repeatable. e.g. --run HP-100k:logs_hp_100k --run PhaseB:logs_phase_b")
    args = ap.parse_args()

    runs = []
    for spec in args.run:
        if ":" not in spec:
            ap.error(f"--run expects LABEL:DIR, got {spec!r}")
        label, d = spec.split(":", 1)
        per, micro = collect(d)
        if not per:
            ap.error(f"no ZAM* scenario logs found under {d!r}")
        runs.append(summarize(label, per, micro))

    for s in runs:
        print_run(s)
    if len(runs) == 2:
        # second run treated as the candidate, first as the baseline
        print_delta(runs[0], runs[1])

    print("\nnote: micro = every timestep weighted equally; macro = every scenario "
          "weighted equally.\n      n_traj is reconstructed from feasibility and is an "
          "estimate; horizon-est is\n      the independent Phase-B cross-check.")


if __name__ == "__main__":
    main()
