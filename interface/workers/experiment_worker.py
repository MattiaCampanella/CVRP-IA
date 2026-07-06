"""
Background ExperimentWorker — runs the immunological algorithm in a
separate thread so the Tk main loop stays responsive.

Communication contract
----------------------
The worker puts dicts of the form::

    {
        "event": <one of "run_start"|"step"|"new_best"|"run_end"|"run_stopped"
                    |"experiment_start"|"experiment_end"|"log"|"error">,
        ...
    }

into ``outgoing_queue``. The GUI polls this queue from the Tk main
thread (``root.after``) and routes events to the appropriate widgets.

The expected parameters in the constructor ``params`` dict mirror the
``run_experiment`` public signature, plus:
- ``instance_path``: absolute path to the .vrp file
- ``best_known``: optional reference value for gap calculation
"""
from __future__ import annotations

import queue
import threading
import time
import traceback
from typing import Any, Callable, Dict, Optional

from src.experiment import run_experiment


class ExperimentWorker(threading.Thread):
    """Thread wrapper around ``run_experiment`` with live progress events.

    Runs in non-parallel mode (so callbacks can be used) and pushes all
    progress events into a thread-safe ``queue.Queue``. The owning GUI
    drains this queue from its main thread.
    """

    def __init__(
        self,
        params: Dict[str, Any],
        outgoing_queue: "queue.Queue[Dict[str, Any]]",
    ) -> None:
        super().__init__(daemon=True, name="ExperimentWorker")
        self.params = params
        self.outgoing_queue = outgoing_queue
        self._stop_event = threading.Event()

    # -- public interface ---------------------------------------------------

    def request_stop(self) -> None:
        """Ask the worker to terminate gracefully."""
        self._stop_event.set()

    def stop_requested(self) -> bool:
        return self._stop_event.is_set()

    # -- internal helpers ---------------------------------------------------

    def _emit(self, payload: Dict[str, Any]) -> None:
        """Append a payload to the queue (fire-and-forget)."""
        self.outgoing_queue.put(payload)

    def _progress_callback(self, info: Dict[str, Any]) -> None:
        """Forward algorithm-emitted events verbatim."""
        self.outgoing_queue.put(info)

    def _stop_check(self) -> bool:
        return self._stop_event.is_set()

    # -- main loop ----------------------------------------------------------

    def run(self) -> None:  # noqa: D401 — overridden Thread.run
        try:
            self._execute()
        except Exception as exc:  # noqa: BLE001 — report any error to the GUI
            self._emit({
                "event": "error",
                "message": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            })

    def _execute(self) -> None:
        instance_path: str = self.params["instance_path"]
        n_runs: int = int(self.params["n_runs"])

        self._emit({
            "event": "log",
            "level": "info",
            "message": (
                f"Worker avviato su {instance_path} "
                f"(N={n_runs}, max_fe={int(self.params['max_fe']):,})"
            ),
        })

        wall_start = time.time()
        result = run_experiment(
            instance_path=instance_path,
            n_runs=n_runs,
            max_fe=int(self.params["max_fe"]),
            pop_size=int(self.params["pop_size"]),
            clone_factor=float(self.params["clone_factor"]),
            beta=float(self.params["beta"]),
            rho=float(self.params["rho"]),
            replacement_rate=float(self.params["replacement_rate"]),
            local_search_top_k=int(self.params["local_search_top_k"]),
            verbose=False,
            best_known=self.params.get("best_known"),
            parallel=False,
            progress_callback=self._progress_callback,
            stop_check=self._stop_check,
        )

        wall_elapsed = time.time() - wall_start
        cancelled = self._stop_event.is_set()

        self._emit({
            "event": "experiment_end",
            "instance_name": result.instance_name,
            "best_known": result.best_known,
            "best": result.best,
            "mean": result.mean,
            "std": result.std,
            "satisfability": result.satisfability,
            "avg_iterations_to_best": result.avg_iterations_to_best,
            "total_time": result.total_time,
            "wall_time": wall_elapsed,
            "n_runs_completed": len(result.run_results),
            "n_runs_requested": n_runs,
            "cancelled": cancelled,
            "run_results": [
                {
                    "run_id": r.run_id,
                    "best_cost": r.best_cost,
                    "n_routes": r.n_routes,
                    "customers_served": r.customers_served,
                    "iterations_to_best": r.iterations_to_best,
                    "history": r.history,
                }
                for r in result.run_results
            ],
        })
