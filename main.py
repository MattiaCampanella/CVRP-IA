"""
Entry point for running CVRP Immunological Algorithm experiments.

Usage:
    python main.py [--quick] [--instance INSTANCE_NAME]

Options:
    --quick      Run a quick test on a single instance with reduced FE budget
    --instance   Specify a single instance to test (e.g. A-n45-k7)
"""
import argparse
import os
import sys
import time

from src.experiment import run_all_experiments, run_experiment, BKS
from src.stats import (
    generate_summary_table,
    generate_latex_table,
    generate_all_plots,
)


def main():
    parser = argparse.ArgumentParser(
        description="CVRP - Algoritmo Immunologico (CLONALG)"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a quick test with reduced budget",
    )
    parser.add_argument(
        "--instance",
        type=str,
        default=None,
        help="Run a single instance (e.g. A-n45-k7)",
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=5,
        help="Number of runs per instance (default: 5)",
    )
    parser.add_argument(
        "--max-fe",
        type=int,
        default=350_000,
        help="Max fitness evaluations (default: 350000)",
    )
    args = parser.parse_args()

    instances_dir = "instances"

    if args.quick:
        print("\n" + "=" * 60)
        print("QUICK TEST MODE")
        print("=" * 60)
        instance_name = args.instance or "A-n45-k7"
        path = os.path.join(instances_dir, f"{instance_name}.vrp")

        if not os.path.exists(path):
            print(f"ERROR: Instance file not found: {path}")
            print("Please download instances from CVRPLIB first.")
            sys.exit(1)

        result = run_experiment(
            instance_path=path,
            n_runs=2,
            max_fe=10_000,
            pop_size=50,
            verbose=True,
            best_known=BKS.get(instance_name),
        )
        print(f"\nResults for {result.instance_name}:")
        print(f"  Best: {result.best:.2f}")
        print(f"  Mean: {result.mean:.2f}")
        print(f"  Std:  {result.std:.2f}")
        print(f"  Satisfability: {result.satisfability}")
        return

    if args.instance:
        instance_name = args.instance
        path = os.path.join(instances_dir, f"{instance_name}.vrp")
        if not os.path.exists(path):
            print(f"ERROR: Instance file not found: {path}")
            sys.exit(1)

        result = run_experiment(
            instance_path=path,
            n_runs=args.n_runs,
            max_fe=args.max_fe,
            verbose=True,
            best_known=BKS.get(instance_name),
        )
        print(f"\nResults for {result.instance_name}:")
        print(f"  Best:      {result.best:.2f}")
        print(f"  Mean:      {result.mean:.2f}")
        print(f"  Std Dev:   {result.std:.2f}")
        print(f"  Satisf.:   {result.satisfability}")
        print(f"  Avg Iter:  {result.avg_iterations_to_best:.0f}")
        if result.best_known:
            gap = (result.best - result.best_known) / result.best_known * 100
            print(f"  BKS:       {result.best_known}")
            print(f"  Gap:       {gap:.2f}%")
        return

    # Run all experiments
    start_time = time.time()
    results = run_all_experiments(
        instances_dir=instances_dir,
        results_dir="results",
        n_runs=args.n_runs,
        max_fe=args.max_fe,
        verbose=True,
    )

    elapsed = time.time() - start_time

    # Generate output files
    print("\n" + "=" * 60)
    print("Generating summary tables and plots...")
    print("=" * 60)

    generate_summary_table(results)
    generate_latex_table(results)
    generate_all_plots(results)

    print(f"\nAll experiments completed in {elapsed:.0f}s "
          f"({elapsed/3600:.1f}h)")
    print("Results saved to results/")


if __name__ == "__main__":
    main()
