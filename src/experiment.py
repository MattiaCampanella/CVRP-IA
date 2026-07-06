"""
Experimental protocol for the CVRP Immunological Algorithm project.

Runs the required experiments:
- 5 runs per instance
- Termination: 3.5 × 10^5 fitness evaluations
- Records: best, mean, std, satisfability, avg iterations to best
- Supports parallel execution of runs via multiprocessing
"""
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Callable, Dict, Any, List, Optional, Tuple

import numpy as np

from src.immune_algorithm import ImmuneAlgorithm
from src.instance import Instance


@dataclass
class RunResult:
    """Result of a single run."""
    run_id: int
    best_cost: float
    n_routes: int
    customers_served: int
    feasible: bool
    iterations_to_best: int  # generation when best was found
    history: List[Tuple] = field(default_factory=list)


@dataclass
class InstanceResult:
    """Aggregated results for one instance across all runs."""
    instance_name: str
    best_known: float | None
    best: float
    mean: float
    std: float
    satisfability: str
    avg_iterations_to_best: float
    run_results: List[RunResult] = field(default_factory=list)
    total_time: float = 0.0


def extract_optimal_from_comment(instance: Instance) -> float | None:
    """Try to extract the optimal value from the instance comment."""
    import re
    match = re.search(r'(?:Optimal|Best)\s+value:\s*(\d+\.?\d*)',
                      instance.comment, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return None


def _run_single(
    instance_path: str,
    run_id: int,
    max_fe: int,
    pop_size: int,
    clone_factor: float,
    beta: float,
    rho: float,
    replacement_rate: float,
    local_search_top_k: int,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    stop_check: Optional[Callable[[], bool]] = None,
) -> RunResult:
    """
    Execute a single run of the algorithm. Designed to work with
    ProcessPoolExecutor (all arguments must be picklable).

    progress_callback and stop_check are *optional* hooks used by the GUI:
    - progress_callback receives dict events emitted by the algorithm.
    - stop_check returns True to request early termination.

    In multiprocessing mode (ProcessPoolExecutor) these hooks are NOT used,
    because callbacks cannot cross process boundaries cheaply. They are
    only meaningful for in-process, single-threaded execution (GUI mode).
    """
    instance = Instance.from_file(instance_path)
    algo = ImmuneAlgorithm(
        instance=instance,
        pop_size=pop_size,
        clone_factor=clone_factor,
        beta=beta,
        rho=rho,
        replacement_rate=replacement_rate,
        local_search_top_k=local_search_top_k,
        max_fe=max_fe,
        seed=run_id,
        verbose=False,
        progress_callback=progress_callback,
        stop_check=stop_check,
    )
    algo.run_id_for_callback = run_id

    best_sol, history = algo.run()

    return RunResult(
        run_id=run_id,
        best_cost=best_sol.cost,
        n_routes=best_sol.num_routes,
        customers_served=best_sol.num_customers_served,
        feasible=best_sol.feasible,
        iterations_to_best=algo.best_found_at_generation,
        history=history,
    )


def run_experiment(
    instance_path: str,
    n_runs: int = 5,
    max_fe: int = 350_000,
    pop_size: int = 100,
    clone_factor: float = 10.0,
    beta: float = 0.2,
    rho: float = 0.5,
    replacement_rate: float = 0.2,
    local_search_top_k: int = 10,
    verbose: bool = False,
    best_known: float | None = None,
    parallel: bool = True,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    stop_check: Optional[Callable[[], bool]] = None,
) -> InstanceResult:
    """
    Run the full experimental protocol on a single CVRP instance.

    If parallel=True and n_runs > 1, runs are executed in parallel
    using ProcessPoolExecutor (callbacks cannot be used in this mode).
    Otherwise runs are executed sequentially; if ``progress_callback`` is
    provided it receives events emitted by the algorithm.

    If ``stop_check`` returns True mid-run, the currently active run is
    discarded and execution stops. Returns an InstanceResult with only the
    fully completed runs.
    """
    instance = Instance.from_file(instance_path)
    run_results: List[RunResult] = []

    if verbose:
        print(f"\n{'='*60}")
        print(f"Instance: {instance.name} ({instance.dimension} nodes, "
              f"capacity={instance.capacity})")
        print(f"{'='*60}")

    # Notify GUI about experiment-level events.
    if progress_callback is not None:
        progress_callback({
            "event": "experiment_start",
            "instance": instance.name,
            "n_runs": n_runs,
            "max_fe": max_fe,
            "pop_size": pop_size,
        })

    total_start = time.time()

    # The combination (parallel=True, callbacks enabled) is not supported
    # because the worker processes can't share live state with the GUI.
    if parallel and n_runs > 1:
        # Parallel execution — callbacks disabled.
        if verbose:
            print(f"  Running {n_runs} runs in parallel...")
        if progress_callback is not None:
            progress_callback({
                "event": "log",
                "level": "warning",
                "message": (
                    "Parallel execution does NOT emit live progress to "
                    "the GUI. Set --no-parallel (or parallel=False) for "
                    "live updates."
                ),
            })
        futures = {}
        with ProcessPoolExecutor(max_workers=min(n_runs, os.cpu_count() or 4)) as pool:
            for run_id in range(1, n_runs + 1):
                future = pool.submit(
                    _run_single,
                    instance_path, run_id, max_fe, pop_size,
                    clone_factor, beta, rho, replacement_rate,
                    local_search_top_k,
                )
                futures[future] = run_id

            for future in as_completed(futures):
                result = future.result()
                run_results.append(result)
                if verbose:
                    print(
                        f"  Run {result.run_id} done | "
                        f"Best: {result.best_cost:.2f} | "
                        f"Routes: {result.n_routes} | "
                        f"Gen to best: {result.iterations_to_best}"
                    )

        # Sort by run_id for deterministic ordering
        run_results.sort(key=lambda r: r.run_id)
    else:
        # Sequential execution — callbacks fully supported.
        for run_id in range(1, n_runs + 1):
            # Honor early stop requests before starting a new run.
            if stop_check is not None and stop_check():
                if progress_callback is not None:
                    progress_callback({
                        "event": "log",
                        "level": "warning",
                        "message": (
                            f"Stop requested before run {run_id}. "
                            "Discarded remaining runs."
                        ),
                    })
                break

            if progress_callback is not None:
                progress_callback({
                    "event": "run_start",
                    "run_id": run_id,
                    "n_runs": n_runs,
                    "max_fe": max_fe,
                })

            run_start = time.time()
            result = _run_single(
                instance_path, run_id, max_fe, pop_size,
                clone_factor, beta, rho, replacement_rate,
                local_search_top_k,
                progress_callback=progress_callback,
                stop_check=stop_check,
            )
            # If the run was stopped mid-way, best_cost ends at finite value
            # but the run is incomplete: skip it from the summary.
            if stop_check is not None and stop_check():
                if verbose:
                    print(f"  Run {run_id} interrupted by user — discarded.")
                if progress_callback is not None:
                    progress_callback({
                        "event": "log",
                        "level": "warning",
                        "message": (
                            f"Run {run_id} interrupted mid-execution. "
                            "Discarding its result from the summary."
                        ),
                    })
                continue

            run_time = time.time() - run_start
            run_results.append(result)

            if progress_callback is not None:
                progress_callback({
                    "event": "run_end",
                    "run_id": run_id,
                    "best_cost": result.best_cost,
                    "n_routes": result.n_routes,
                    "customers_served": result.customers_served,
                    "iterations_to_best": result.iterations_to_best,
                    "elapsed_sec": run_time,
                })

            if verbose:
                print(
                    f"  Best: {result.best_cost:.2f} | "
                    f"Routes: {result.n_routes} | "
                    f"Feasible: {result.feasible} | "
                    f"Gen to best: {result.iterations_to_best} | "
                    f"Time: {run_time:.1f}s"
                )

    total_time = time.time() - total_start

    # Aggregate statistics
    costs = [r.best_cost for r in run_results]
    best = min(costs)
    mean = np.mean(costs)
    std = np.std(costs, ddof=1) if len(costs) > 1 else 0.0
    avg_iter = np.mean([r.iterations_to_best for r in run_results])

    # Satisfability
    n_customers = instance.dimension - 1
    all_served = all(r.customers_served == n_customers
                     for r in run_results)
    servings = [r.customers_served for r in run_results]
    if all_served:
        satisfability_str = f"{n_customers}/{n_customers}"
    else:
        satisfability_str = (
            f"min={min(servings)}/{n_customers}, "
            f"avg={np.mean(servings):.1f}/{n_customers}"
        )

    return InstanceResult(
        instance_name=instance.name,
        best_known=best_known,
        best=best,
        mean=mean,
        std=std,
        satisfability=satisfability_str,
        avg_iterations_to_best=avg_iter,
        run_results=run_results,
        total_time=total_time,
    )


# BKS values extracted from instance file headers via extract_optimal_from_comment()
BKS = {
    "A-n45-k7": 1146,
    "A-n60-k9": 1354,
    "A-n80-k10": 1314,
    "B-n56-k7": 707,
    "B-n66-k9": 1316,
    "B-n78-k10": 1221,
    "E-n76-k8": 735,
    "E-n101-k14": 1071,
    "P-n50-k10": 696,
    "P-n101-k4": 681,
}


def run_all_experiments(
    instances_dir: str = "instances",
    results_dir: str = "results",
    n_runs: int = 5,
    max_fe: int = 350_000,
    verbose: bool = True,
    parallel: bool = True,
) -> List[InstanceResult]:
    """
    Run experiments on all required instances.
    """
    required_instances = list(BKS.keys())
    all_results: List[InstanceResult] = []

    # BKS overrides from instance file comments
    for inst_name in required_instances:
        path = os.path.join(instances_dir, f"{inst_name}.vrp")
        if not os.path.exists(path):
            print(f"WARNING: {path} not found, skipping...")
            continue

        # Check if we can extract BKS from the instance file
        try:
            inst = Instance.from_file(path)
            extracted = extract_optimal_from_comment(inst)
        except Exception:
            extracted = None

        # Use extracted value if available, otherwise fallback to BKS dict
        bks = extracted if extracted is not None else BKS.get(inst_name)

        result = run_experiment(
            instance_path=path,
            n_runs=n_runs,
            max_fe=max_fe,
            verbose=verbose,
            best_known=bks,
            parallel=parallel,
        )
        all_results.append(result)

        # Save individual result
        os.makedirs(results_dir, exist_ok=True)
        result_path = os.path.join(results_dir, f"{inst_name}.json")
        save_result_json(result, result_path)

        # Save convergence data for plotting
        # History is now (fe, best_cost_so_far) — much smaller
        conv_path = os.path.join(
            results_dir, f"{inst_name}_convergence.json"
        )
        convergence_data = {
            "instance": inst_name,
            "runs": [
                {
                    "run_id": r.run_id,
                    "history": r.history,
                    "best": r.best_cost,
                }
                for r in result.run_results
            ],
        }
        with open(conv_path, "w") as f:
            json.dump(convergence_data, f, default=str)

    return all_results


def save_result_json(result: InstanceResult, path: str) -> None:
    """Save InstanceResult to JSON."""
    data = {
        "instance": result.instance_name,
        "best_known": result.best_known,
        "best": result.best,
        "mean": result.mean,
        "std": result.std,
        "satisfability": result.satisfability,
        "avg_iterations_to_best": result.avg_iterations_to_best,
        "total_time": result.total_time,
        "gap_pct": (
            round(
                (result.best - result.best_known) / result.best_known
                * 100, 2
            )
            if result.best_known
            else None
        ),
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
