"""
Statistics and visualization utilities for CVRP experimental results.
"""
import json
import os
from typing import List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from src.experiment import InstanceResult, BKS

# Use non-interactive backend
matplotlib.use("Agg")


def generate_summary_table(
    results: List[InstanceResult], output_path: str = "results/summary.csv"
) -> None:
    """Generate a CSV summary table of all results."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    rows = [
        "Instance,BKS,Best,Mean,StdDev,Satisfability,AvgIter,Time(s),Gap%"
    ]
    for r in results:
        gap = ""
        if r.best_known:
            gap = f"{(r.best - r.best_known) / r.best_known * 100:.2f}"
        rows.append(
            f"{r.instance_name},{r.best_known},{r.best:.2f},"
            f"{r.mean:.2f},{r.std:.2f},"
            f"{r.satisfability},{r.avg_iterations_to_best:.0f},"
            f"{r.total_time:.1f},{gap}"
        )

    with open(output_path, "w") as f:
        f.write("\n".join(rows))

    print(f"Summary table saved to {output_path}")


def generate_latex_table(
    results: List[InstanceResult], output_path: str = "results/table.tex"
) -> None:
    """Generate a LaTeX table for the report."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Risultati sperimentali dell'Algoritmo Immunologico per CVRP.}",
        r"\label{tab:results}",
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r"Istanza & BKS & Best & Mean & Std Dev & Satisf. & Iter Medie & Gap \% \\",
        r"\midrule",
    ]

    for r in results:
        gap_str = (
            f"{(r.best - r.best_known) / r.best_known * 100:.1f}\\%"
            if r.best_known
            else "--"
        )
        lines.append(
            f"{r.instance_name} & {r.best_known or '--'} & "
            f"{r.best:.0f} & {r.mean:.1f} & {r.std:.1f} & "
            f"{r.satisfability} & {r.avg_iterations_to_best:.0f} & "
            f"{gap_str} \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    print(f"LaTeX table saved to {output_path}")


def plot_convergence(
    instance_name: str,
    results_dir: str = "results",
    output_dir: str = "results",
) -> None:
    """
    Generate a convergence plot for a single instance.

    History format: list of (fe, best_cost_so_far) tuples, sampled every 100 FE.
    """
    conv_path = os.path.join(results_dir, f"{instance_name}_convergence.json")
    if not os.path.exists(conv_path):
        print(f"No convergence data for {instance_name}")
        return

    with open(conv_path, "r") as f:
        data = json.load(f)

    plt.figure(figsize=(10, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, len(data["runs"])))

    for idx, run_data in enumerate(data["runs"]):
        history = run_data["history"]
        # history is [[fe, best_cost_so_far], ...]
        fe_vals = [h[0] for h in history]
        cost_vals = [h[1] for h in history]
        plt.plot(
            fe_vals,
            cost_vals,
            color=colors[idx],
            alpha=0.6,
            linewidth=1.0,
            label=f"Run {run_data['run_id']}",
        )

    # BKS line if available
    bks = BKS.get(instance_name)
    if bks:
        plt.axhline(
            y=bks, color="red", linestyle="--",
            linewidth=1.5, alpha=0.7, label=f"BKS ({bks})"
        )

    plt.xlabel("Fitness Evaluations (FE)")
    plt.ylabel("Best Cost Found")
    plt.title(f"Convergenza - {instance_name}")
    plt.legend(loc="upper right", fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{instance_name}_convergence.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Convergence plot saved to {out_path}")


def generate_all_plots(
    results: List[InstanceResult],
    results_dir: str = "results",
    n_plots: int = 3,
) -> None:
    """
    Generate convergence plots for representative instances.
    """
    # Pick representative instances: small, medium, large
    selected = []
    for instance_name in [
        "A-n45-k7",   # small, set A
        "B-n78-k10",  # medium, set B
        "P-n101-k4",  # large, set P
    ]:
        if os.path.exists(
            os.path.join(results_dir, f"{instance_name}_convergence.json")
        ):
            selected.append(instance_name)

    for inst in selected:
        plot_convergence(inst, results_dir, results_dir)
