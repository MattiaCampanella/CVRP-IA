"""
Batch experiment runner for all CVRP instances.
Runs 5 runs per instance with 350,000 FE each.
Saves results to results/ directory.
"""
import json
import os
import sys
import time

from src.experiment import (
    run_experiment, run_all_experiments, BKS, extract_optimal_from_comment,
)
from src.stats import (
    generate_summary_table, generate_latex_table, generate_all_plots,
)


def ensure_dirs():
    os.makedirs("results", exist_ok=True)


def main():
    ensure_dirs()

    # Check which instances are available
    available = []
    for name in BKS:
        path = os.path.join("instances", f"{name}.vrp")
        if os.path.exists(path):
            available.append(name)
        else:
            print(f"WARNING: {name}.vrp not found, skipping")

    if not available:
        print("No instances found in instances/ directory!")
        return 1

    print(f"Running experiments on {len(available)} instances:")
    for n in available:
        print(f"  - {n}")
    print(f"\nSettings: 5 runs × 350,000 FE per instance")
    print(f"Estimated time: varies by instance size\n")

    overall_start = time.time()

    results = run_all_experiments(
        instances_dir="instances",
        results_dir="results",
        n_runs=5,
        max_fe=350_000,
        verbose=True,
    )

    overall_elapsed = time.time() - overall_start

    print(f"\n{'='*60}")
    print(f"ALL EXPERIMENTS COMPLETED in {overall_elapsed:.0f}s "
          f"({overall_elapsed/3600:.1f}h)")
    print(f"{'='*60}")

    # Generate summary
    generate_summary_table(results)
    generate_latex_table(results)
    generate_all_plots(results)

    # Print final summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"{'Instance':<15} {'BKS':>8} {'Best':>8} {'Mean':>8} "
          f"{'Std':>8} {'Gap%':>8} {'Satisf'}")
    print("-" * 75)
    for r in results:
        gap = ((r.best - r.best_known) / r.best_known * 100
               if r.best_known else float('nan'))
        print(f"{r.instance_name:<15} {r.best_known or '?':>8} "
              f"{r.best:>8.0f} {r.mean:>8.0f} {r.std:>8.0f} "
              f"{gap:>7.1f}% {r.satisfability}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
