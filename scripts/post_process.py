"""
Post-processing script for CVRP-IA experimental results.

Produces additional artifacts useful for the report:
1. A 2x5 CONVERGENCE GRID with all 10 instances on a single PNG
   (reads from existing results/{instance}_convergence.json files).
2. STATIC ROUTES PNGs for each instance (best run snapshot).
   For each instance, this script:
   - Identifies which of the 5 published runs (seed 1..5) produced the
     published best cost.
   - Re-runs the IA with that exact seed to retrieve the SOLUTION object
     (which carries the actual routes).
   - Saves routes to results/best_routes/{instance}_best_routes.json.
   - Draws the routes on top of the instance map and saves a PNG.

Run with:

    python scripts/post_process.py

Prerequisites: results/{instance}_convergence.json must exist (produced by
`run_experiments.py`).
"""
from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Ensure project root is on sys.path when running as a standalone script.
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.experiment import BKS  # noqa: E402
from src.immune_algorithm import ImmuneAlgorithm  # noqa: E402
from src.instance import Instance  # noqa: E402


# =====================================================================
# 1) Convergence grid (reads existing JSONs — no rerun)
# =====================================================================


def _final_cost_of_run(run_data: Dict[str, Any]) -> float:
    """Return the last (best-so-far) cost of a run from its history."""
    history = run_data.get("history") or []
    if not history:
        return float("inf")
    return float(history[-1][1])


def _best_run_id_for(instance_name: str, results_dir: str) -> int:
    """Return the run_id (1..5) whose published cost is lowest.

    Falls back to run_id=1 if convergence data is missing.
    """
    path = os.path.join(results_dir, f"{instance_name}_convergence.json")
    if not os.path.exists(path):
        return 1
    with open(path, "r") as f:
        data = json.load(f)
    runs = data.get("runs") or []
    if not runs:
        return 1
    best = min(runs, key=_final_cost_of_run)
    return int(best.get("run_id") or 1)


def plot_convergence_grid(
    instance_names: List[str],
    results_dir: str,
    output_path: str,
) -> None:
    """Plot a 2x5 convergence grid for all instances."""
    n = len(instance_names)
    cols = 5
    rows = (n + cols - 1) // cols  # 2 rows for 10 instances

    fig, axes = plt.subplots(rows, cols, figsize=(22, 8), sharex=False)
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    legend_added = False
    for i, name in enumerate(instance_names):
        ax = axes_flat[i]
        conv_path = os.path.join(results_dir, f"{name}_convergence.json")
        if not os.path.exists(conv_path):
            ax.text(
                0.5, 0.5, f"No data:\n{name}", ha="center", va="center",
                transform=ax.transAxes, fontsize=10,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        with open(conv_path) as f:
            data = json.load(f)

        n_runs = len(data.get("runs") or [])
        cmap = plt.get_cmap("tab10")
        for idx, run_data in enumerate(data.get("runs") or []):
            history = run_data.get("history") or []
            xs = [h[0] for h in history]
            ys = [h[1] for h in history]
            ax.plot(
                xs, ys, color=cmap(idx % 10), alpha=0.7, linewidth=1.2,
                label=f"Run {run_data.get('run_id', idx+1)}",
            )

        bks = BKS.get(name)
        if bks:
            show_label = not legend_added
            ax.axhline(
                y=bks, color="red", linestyle="--", linewidth=1.2,
                alpha=0.7,
                label=f"BKS ({bks})" if show_label else None,
            )
            legend_added = True

        n_runs_label = f"{n_runs} run" + ("s" if n_runs != 1 else "")
        ax.set_title(f"{name}\n({n_runs_label})", fontsize=10, fontweight="bold")
        ax.set_xlabel("Fitness Evaluations (FE)", fontsize=8)
        ax.set_ylabel("Miglior costo", fontsize=8)
        ax.grid(True, alpha=0.3)
        # Show legend only on first subplot to avoid clutter
        if i == 0:
            ax.legend(loc="upper right", fontsize=7)

    # Hide any unused subplots (shouldn't be any for 10 instances in 2x5)
    for j in range(len(instance_names), len(axes_flat)):
        axes_flat[j].axis("off")

    fig.suptitle(
        "Convergenza Algoritmo Immunologico — tutte le 10 istanze CVRP",
        fontsize=15, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Convergence grid saved to {output_path}")


# =====================================================================
# 2) Extract best routes (per-instance IA run with best run_id's seed)
# =====================================================================


def _run_for_routes(instance_name: str, seed: int) -> Dict[str, Any]:
    """Re-run IA with a specific seed to retrieve the Solution object.

    Returns a dictionary with cost and routes.
    """
    path = f"instances/{instance_name}.vrp"
    inst = Instance.from_file(path)
    start = time.time()
    algo = ImmuneAlgorithm(
        instance=inst,
        pop_size=100, clone_factor=10.0, beta=0.2, rho=0.5,
        replacement_rate=0.2, local_search_top_k=10,
        max_fe=350_000, seed=seed, verbose=False,
    )
    best_sol, _ = algo.run()
    elapsed = time.time() - start
    return {
        "instance": instance_name,
        "seed": seed,
        "elapsed_sec": elapsed,
        "cost": best_sol.cost,
        "feasible": best_sol.feasible,
        "n_routes": best_sol.num_routes,
        "customers_served": best_sol.num_customers_served,
        "n_customers": inst.dimension - 1,
        "capacity": inst.capacity,
        "best_known": BKS.get(instance_name),
        "depot": inst.depot,
        "routes": [list(r) for r in best_sol.routes],
    }


def plot_static_routes(
    routes_data: Dict[str, Any],
    instance: Instance,
    output_path: str,
) -> None:
    """Draw depot, customers, and the routes on a 2-D map; save as PNG."""
    coords = instance.coordinates
    depot = int(routes_data["depot"])
    instance_name = routes_data["instance"]
    n_routes = len(routes_data["routes"])
    cost = routes_data["cost"]
    n_customers = routes_data["n_customers"]
    served = routes_data["customers_served"]
    bks = routes_data.get("best_known")

    fig, ax = plt.subplots(figsize=(9, 7))

    # Customers (excluding depot)
    cx = [coords[i][0] for i in range(len(coords)) if i != depot]
    cy = [coords[i][1] for i in range(len(coords)) if i != depot]
    ax.scatter(
        cx, cy, c="#3b82f6", s=22, zorder=3, label="Clienti",
        edgecolors="white", linewidths=0.4,
    )

    # Depot
    ax.scatter(
        coords[depot][0], coords[depot][1], c="#dc2626", s=200,
        marker="s", zorder=5, label="Deposito",
        edgecolors="white", linewidths=0.8,
    )

    # Routes
    cmap = plt.get_cmap("tab10")
    for i, route in enumerate(routes_data["routes"]):
        color = cmap(i % 10)
        pts = (
            [coords[depot]] + [coords[c] for c in route] + [coords[depot]]
        )
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(
            xs, ys, color=color, linewidth=1.5, alpha=0.85,
            label=(
                f"Rotta {i+1} ({len(route)} clienti)"
                if i < 8 else None
            ),
        )

    info = (
        f"Costo: {cost:,.2f}  |  Veicoli: {n_routes}  |  "
        f"Clienti serviti: {served}/{n_customers}"
    )
    if bks:
        gap = (cost - bks) / bks * 100
        info += f"  |  BKS: {bks}  |  Gap: {gap:.2f}%"

    ax.set_title(
        f"Miglior soluzione — {instance_name}\n{info}",
        fontsize=11, fontweight="bold",
    )
    ax.set_xlabel("X", fontsize=9)
    ax.set_ylabel("Y", fontsize=9)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)
    if 0 < n_routes <= 8:
        ax.legend(loc="upper left", fontsize=8, ncol=2)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# =====================================================================
# Main
# =====================================================================


def main() -> None:
    os.makedirs("results", exist_ok=True)
    instance_names = list(BKS.keys())

    # ---- 1) Convergence grid (no rerun) ---------------------------
    print("\n=== Generating 2x5 convergence grid for all 10 instances ===")
    plot_convergence_grid(
        instance_names, "results", "results/all_convergence.png",
    )

    # ---- 2) Extract best routes per instance ----------------------
    print("\n=== Identifying best-seed per instance + extracting routes ===")
    best_routes_dir = "results/best_routes"
    os.makedirs(best_routes_dir, exist_ok=True)

    plan = []
    for name in instance_names:
        seed = _best_run_id_for(name, "results")
        plan.append((name, seed))
        print(f"  {name}: best-seed = {seed}")

    # Run in parallel: each instance gets its own process.
    print("\n=== Running IA once per instance (parallel, 350k FE) ===")
    print("    This reproduces the published best run's solution so that")
    print("    we can extract the actual routes for visualization.")

    routes_results: Dict[str, Dict[str, Any]] = {}
    n_parallel = min(len(plan), os.cpu_count() or 4)
    with ProcessPoolExecutor(max_workers=n_parallel) as pool:
        futures = {
            pool.submit(_run_for_routes, name, seed): name
            for name, seed in plan
        }
        for fut in as_completed(futures):
            name = futures[fut]
            try:
                routes_results[name] = fut.result()
                rd = routes_results[name]
                print(
                    f"  [OK] {name} (seed={rd['seed']}): "
                    f"cost={rd['cost']:.2f}, "
                    f"{rd['n_routes']} routes, {rd['elapsed_sec']:.1f}s"
                )
            except Exception as exc:  # noqa: BLE001
                print(f"  [ERR] {name}: {exc}")

    # Save JSON + PNG for each instance (in deterministic order)
    for name in instance_names:
        rd = routes_results.get(name)
        if rd is None:
            continue
        json_path = os.path.join(
            best_routes_dir, f"{name}_best_routes.json",
        )
        with open(json_path, "w") as f:
            json.dump(rd, f, indent=2)
        inst = Instance.from_file(f"instances/{name}.vrp")
        png_path = os.path.join(best_routes_dir, f"{name}_routes.png")
        plot_static_routes(rd, inst, png_path)
        print(f"  -> {json_path}")
        print(f"  -> {png_path}")

    print("\n=== Post-processing complete ===")


if __name__ == "__main__":
    main()
