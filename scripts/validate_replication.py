"""
validate_replication.py
-----------------------
Checks Phase 1a success criteria against a completed run's metrics.npz.
Prints PASS or FAIL for each criterion, then an overall result.

Usage:
    python scripts/validate_replication.py --results results/baseline_faithful/seed_0/
    python scripts/validate_replication.py --results results/baseline_faithful/ --all-seeds

Exits with code 0 if all criteria pass, 1 if any fail.
"""

import argparse
import os
import sys
import numpy as np


# ─── Success criteria from docs/technical-spec-kd-replication.md ─────────────

# These thresholds are from K&D Section 4 and Figure 7.
# Do not change them without updating the spec document.

CRITERIA = [
    {
        "name": "Prey w_pred < 0 (fear evolved)",
        "description": "Mean prey w_pred is negative at end of run",
        "check": lambda m: float(m["prey_mean_w_pred"][-1]) < 0,
        "threshold_note": "< 0 at step 10M",
    },
    {
        "name": "Prey w_prey > 0 (social affiliation evolved)",
        "description": "Mean prey w_prey is positive at end of run",
        "check": lambda m: float(m["prey_mean_w_prey"][-1]) > 0,
        "threshold_note": "> 0 at step 10M",
    },
    {
        "name": "Prey w_eat > 0 (food reward positive)",
        "description": "Mean prey w_eat is positive at end of run",
        "check": lambda m: float(m["prey_mean_w_eat"][-1]) > 0,
        "threshold_note": "> 0 at step 10M",
    },
    {
        "name": "Predator w_prey > 0 (prey attraction evolved)",
        "description": "Mean predator w_prey is positive at end of run",
        "check": lambda m: float(m["pred_mean_w_prey"][-1]) > 0,
        "threshold_note": "> 0 at step 10M",
    },
    {
        "name": "Prey population not extinct",
        "description": "Prey population > 0 throughout entire run",
        "check": lambda m: np.all(np.array(m["prey_population"]) > 0),
        "threshold_note": "prey_population > 0 at all logged steps",
    },
    {
        "name": "Predator population not extinct",
        "description": "Predator population > 0 throughout entire run",
        "check": lambda m: np.all(np.array(m["predator_population"]) > 0),
        "threshold_note": "predator_population > 0 at all logged steps",
    },
    {
        "name": "Population oscillations present",
        "description": (
            "Prey population shows variation consistent with Lotka-Volterra dynamics. "
            "Std dev of prey population > 10% of mean prey population."
        ),
        "check": lambda m: (
            np.std(m["prey_population"]) / (np.mean(m["prey_population"]) + 1e-8) > 0.10
        ),
        "threshold_note": "std(prey_pop) / mean(prey_pop) > 0.10",
    },
    {
        "name": "Reward weights drifted from initialization",
        "description": (
            "Prey w_pred at end of run differs from initialization mean (0.0) "
            "by more than 2 standard deviations of the init distribution (0.1). "
            "Confirms evolution is running."
        ),
        "check": lambda m: abs(float(m["prey_mean_w_pred"][-1])) > 0.2,
        "threshold_note": "|mean(w_pred)| > 0.2 at end of run",
    },
]


def load_metrics(path: str) -> dict:
    """Load metrics.npz from a seed directory."""
    metrics_path = os.path.join(path, "metrics.npz")
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(
            f"No metrics.npz found at {metrics_path}. "
            "Has the run completed? Check that metrics are saved correctly."
        )
    data = np.load(metrics_path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def check_single_seed(seed_dir: str) -> tuple[bool, list[dict]]:
    """
    Run all criteria against one seed's metrics.
    Returns (all_passed: bool, results: list of criterion result dicts).
    """
    print(f"\n{'─'*60}")
    print(f"Checking: {seed_dir}")
    print(f"{'─'*60}")

    try:
        metrics = load_metrics(seed_dir)
    except FileNotFoundError as e:
        print(f"  ERROR: {e}")
        return False, []

    # Print run summary
    n_steps = len(metrics.get("steps", []))
    if n_steps > 0:
        final_step = int(metrics["steps"][-1])
        prey_pop = int(metrics["prey_population"][-1])
        pred_pop = int(metrics["predator_population"][-1])
        prey_w_pred = float(metrics["prey_mean_w_pred"][-1])
        prey_w_prey = float(metrics["prey_mean_w_prey"][-1])
        print(f"  Final step: {final_step:,}")
        print(f"  Population: {prey_pop} prey, {pred_pop} predators")
        print(f"  Prey w_pred (fear):    {prey_w_pred:+.3f}  (want < 0)")
        print(f"  Prey w_prey (social):  {prey_w_prey:+.3f}  (want > 0)")
        print()

    results = []
    all_passed = True

    for criterion in CRITERIA:
        try:
            passed = criterion["check"](metrics)
        except (KeyError, IndexError, ZeroDivisionError) as e:
            passed = False
            criterion = {**criterion, "_error": str(e)}

        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}  {criterion['name']}")
        if not passed:
            print(f"           Threshold: {criterion['threshold_note']}")
            if "_error" in criterion:
                print(f"           Error: {criterion['_error']}")
            all_passed = False

        results.append({"criterion": criterion["name"], "passed": passed})

    return all_passed, results


def main():
    parser = argparse.ArgumentParser(
        description="Validate Phase 1a replication success criteria."
    )
    parser.add_argument(
        "--results",
        required=True,
        help=(
            "Path to a single seed directory (e.g., results/baseline_faithful/seed_0/) "
            "or to the experiment directory to check all seeds "
            "(e.g., results/baseline_faithful/)."
        ),
    )
    parser.add_argument(
        "--all-seeds",
        action="store_true",
        help="Check all seed_* subdirectories under the results path.",
    )
    parser.add_argument(
        "--min-seeds-passing",
        type=int,
        default=1,
        help=(
            "When checking multiple seeds, how many must pass for overall PASS. "
            "K&D success criteria require 3 of 5 seeds for the key reward results. "
            "Default: 1 (for early Phase 1a checking)."
        ),
    )
    args = parser.parse_args()

    results_path = args.results.rstrip("/")

    # Determine which directories to check
    if args.all_seeds or not os.path.exists(
        os.path.join(results_path, "metrics.npz")
    ):
        # Look for seed_* subdirectories
        seed_dirs = sorted(
            [
                os.path.join(results_path, d)
                for d in os.listdir(results_path)
                if d.startswith("seed_") and os.path.isdir(os.path.join(results_path, d))
            ]
        )
        if not seed_dirs:
            print(f"ERROR: No seed_* directories found under {results_path}")
            print("       Either provide a path to a specific seed directory,")
            print("       or ensure the experiment has completed.")
            sys.exit(1)
    else:
        seed_dirs = [results_path]

    print(f"\n{'='*60}")
    print("Phase 1a Replication Validation")
    print(f"Checking {len(seed_dirs)} seed(s)")
    print(f"{'='*60}")

    seed_results = {}
    for seed_dir in seed_dirs:
        passed, criteria_results = check_single_seed(seed_dir)
        seed_dirs_name = os.path.basename(seed_dir)
        seed_results[seed_dirs_name] = passed

    # Summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    n_passed = sum(seed_results.values())
    n_total = len(seed_results)
    for seed_name, passed in seed_results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}  {seed_name}")

    print(f"\n  {n_passed}/{n_total} seeds passing all criteria")

    # K&D paper requires fear and social affiliation in ≥3/5 seeds.
    # For early validation (1-2 seeds), default min is 1.
    if n_passed >= args.min_seeds_passing:
        print(f"\n✅ OVERALL PASS ({n_passed} >= {args.min_seeds_passing} required)")
        print("\nNext step: proceed to Phase 1b (shared policy comparison).")
        sys.exit(0)
    else:
        print(f"\n❌ OVERALL FAIL ({n_passed} < {args.min_seeds_passing} required)")
        print("\nDo not proceed to Phase 1b. Debug following AGENTS.md instructions.")
        sys.exit(1)


if __name__ == "__main__":
    main()
