"""
Immunological Algorithm (CLONALG) for the CVRP.

Inspired by the Clonal Selection Principle (De Castro & Von Zuben, 2002):
1. Generate initial population of antibodies (permutations)
2. Evaluate affinity (inverse of cost)
3. While not terminated:
   a. Select best antibodies
   b. Clone proportionally to affinity rank
   c. Hypermutate clones (mutation rate ∝ 1/affinity)
   d. Evaluate clones
   e. Replace worst antibodies with best clones
   f. Introduce new random antibodies for diversity
"""
import math
import random
import time
from typing import List, Callable, Tuple, Optional, Dict, Any

from src.instance import Instance
from src.operators import (
    hypermutation,
    two_opt_intra_route,
    PERM_OPERATORS,
)
from src.solution import Solution
from src.utils import set_seed


# History sampling: record one entry every HISTORY_SAMPLE_INTERVAL FE
HISTORY_SAMPLE_INTERVAL = 500

# Routes snapshot: emit routes to the GUI every N generations (periodic visual update)
ROUTES_SNAPSHOT_INTERVAL = 150


class ImmuneAlgorithm:
    """
    CLONALG implementation for the Capacitated Vehicle Routing Problem.

    Parameters
    ----------
    instance : Instance
        The CVRP instance to solve.
    pop_size : int
        Population size (number of antibodies).
    clone_factor : float
        Base number of clones per selected antibody.
    beta : float
        Fraction of population selected for cloning (0 < beta <= 1).
    rho : float
        Scaling factor for hypermutation rate.
    replacement_rate : float
        Fraction of population replaced with new random antibodies each gen.
    local_search_top_k : int
        Number of top solutions to apply 2-opt local search per generation.
    max_fe : int
        Maximum number of fitness evaluations (termination criterion).
    seed : int or None
        Random seed for reproducibility.
    verbose : bool
        Whether to print progress.
    progress_callback : Optional[Callable[[Dict[str, Any]], None]]
        If provided, invoked periodically with progress dicts. Used by
        the GUI to render live updates. Safe to leave ``None`` for the
        original CLI workflow.
    stop_check : Optional[Callable[[], bool]]
        If provided, called once per generation. Returning ``True``
        requests early termination. Used by the GUI's Stop button.
    """

    def __init__(
        self,
        instance: Instance,
        pop_size: int = 100,
        clone_factor: float = 10.0,
        beta: float = 0.2,
        rho: float = 0.5,
        replacement_rate: float = 0.2,
        local_search_top_k: int = 10,
        max_fe: int = 350_000,
        seed: int | None = None,
        verbose: bool = False,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        stop_check: Optional[Callable[[], bool]] = None,
    ):
        self.instance = instance
        self.pop_size = pop_size
        self.clone_factor = clone_factor
        self.beta = beta
        self.rho = rho
        self.replacement_rate = replacement_rate
        self.local_search_top_k = local_search_top_k
        self.max_fe = max_fe
        self.seed = seed
        self.verbose = verbose

        # GUI / progress hooks. progress_callback(info: dict) is invoked
        # periodically to report the algorithm state to a UI (keys: event,
        # run_id, generation, fe, max_fe, best_cost, current_best,
        # elapsed_sec, routes, history_point, ...). stop_check() returning
        # True requests termination. Both are Optional — when None (the
        # default) the original CLI behaviour applies.
        self.progress_callback = progress_callback
        self.stop_check = stop_check
        # Identity of the currently-running run (1-indexed). Set by
        # experiment.py / run_experiment().
        self.run_id_for_callback: int = 1
        # Wall-clock time when the run() method actually started.
        self.run_start_time: float = 0.0
        # Tracks the best cost last reported via callback so we can
        # emit a "new_best" event when the global best changes.
        self._last_reported_best: float = float("inf")

        # Precompute customer list (excludes depot) for efficiency
        self.customers = [
            i for i in range(instance.dimension)
            if i != instance.depot
        ]

        # State
        self.population: List[Solution] = []
        self.fe_counter = 0
        self.best_solution: Solution | None = None
        self.generation = 0
        self.best_found_at_generation = 0
        self.best_found_at_fe = 0
        self.history: List = []  # tuples of (fe, best_cost_so_far)
        self._last_history_fe = -HISTORY_SAMPLE_INTERVAL  # force first record

    def _random_permutation(self) -> List[int]:
        """Generate a random permutation of customer indices."""
        customers = list(self.customers)
        random.shuffle(customers)
        return customers

    def _random_solution(self) -> Solution:
        """Create a random solution and count the fitness evaluation."""
        perm = self._random_permutation()
        sol = Solution(perm, self.instance)
        self.fe_counter += 1
        self._maybe_record_history()
        return sol

    def _count_fe(self) -> None:
        """Increment FE counter and sample history if needed."""
        self.fe_counter += 1
        self._maybe_record_history()

    def _maybe_record_history(self) -> None:
        """Record history entry every HISTORY_SAMPLE_INTERVAL FEs.

        Also emits a 'step' (or 'new_best') event to the GUI callback.
        """
        if self.fe_counter - self._last_history_fe >= HISTORY_SAMPLE_INTERVAL:
            best_cost = (
                self.best_solution.cost
                if self.best_solution is not None
                else float("inf")
            )
            self.history.append((self.fe_counter, best_cost))
            self._last_history_fe = self.fe_counter
            self._emit_progress(event="step")

    def initialize(self) -> None:
        """Create the initial random population."""
        if self.seed is not None:
            set_seed(self.seed)

        self.fe_counter = 0
        self.generation = 0
        self.history = []
        self._last_history_fe = -HISTORY_SAMPLE_INTERVAL

        self.population = [
            self._random_solution() for _ in range(self.pop_size)
        ]
        self.population.sort(key=lambda s: s.cost)

        self.best_solution = self.population[0].copy()
        self.best_found_at_generation = 0
        self.best_found_at_fe = self.fe_counter

        # Record initial best
        self._maybe_record_history()

        if self.verbose:
            print(
                f"Init | FE: {self.fe_counter}/{self.max_fe} "
                f"| Best: {self.best_solution.cost:.2f}"
            )

    def step(self) -> None:
        """
        Perform one generation of the CLONALG algorithm.
        """
        self.generation += 1
        pop = self.population

        # 1. Select best antibodies for cloning
        n_select = max(1, int(self.pop_size * self.beta))
        selected = pop[:n_select]

        # Precompute affinity normalization for selected antibodies
        costs = [s.cost for s in selected]
        c_min, c_max = min(costs), max(costs)
        cost_range = c_max - c_min if c_max > c_min else 1.0

        # 2. Clone and hypermutate
        clones: List[Solution] = []
        for rank, antibody in enumerate(selected):
            # More clones for better antibodies
            n_clones = max(
                1, int(self.clone_factor * (1.0 - rank / n_select))
            )

            # Affinity-based mutation rate (CLONALG standard):
            # α = exp(-ρ * affinity_norm)
            # Good antibodies (low cost → high affinity) mutate LESS
            # Poor antibodies (high cost → low affinity) mutate MORE
            # norm_cost: 0 = best (lowest cost), 1 = worst (highest cost)
            norm_cost = (antibody.cost - c_min) / cost_range
            # affinity_norm: 1 = best, 0 = worst
            affinity_norm = 1.0 - norm_cost
            # mutation_rate: α = exp(-ρ * affinity_norm)
            mutation_rate = math.exp(-self.rho * affinity_norm)

            for _ in range(n_clones):
                if self.fe_counter >= self.max_fe:
                    break
                clone = hypermutation(
                    antibody,
                    mutation_rate,
                    PERM_OPERATORS,
                )
                # hypermutation calls _split() once internally → 1 FE
                self._count_fe()
                clones.append(clone)

        if self.fe_counter >= self.max_fe:
            self._update_best()
            return

        # 3. Apply local search (2-opt) to the best NEW clones.
        # Clones are fresh unrefined solutions that benefit most from 2-opt.
        # This is NOT counted as FE (it refines routes directly).
        if self.local_search_top_k > 0 and clones:
            clones.sort(key=lambda s: s.cost)
            for i in range(min(self.local_search_top_k, len(clones))):
                improved = two_opt_intra_route(clones[i])
                if improved.cost < clones[i].cost:
                    clones[i] = improved

        # 4. Selection: combine clones + original population
        combined = pop + clones
        combined.sort(key=lambda s: s.cost)

        # Remove duplicates (same permutation = same solution)
        unique: List[Solution] = []
        seen: set = set()
        for s in combined:
            key = tuple(s.permutation)
            if key not in seen:
                seen.add(key)
                unique.append(s)
            if len(unique) >= self.pop_size:
                break

        # 5. Build new population
        new_pop = unique[:self.pop_size]

        # If we don't have enough, fill with random
        while len(new_pop) < self.pop_size:
            if self.fe_counter < self.max_fe:
                new_pop.append(self._random_solution())
            else:
                break

        # 6. Diversity: replace worst with new random solutions
        n_replace = int(self.pop_size * self.replacement_rate)
        if n_replace > 0 and self.fe_counter < self.max_fe:
            new_pop = new_pop[: self.pop_size - n_replace]
            for _ in range(n_replace):
                if self.fe_counter >= self.max_fe:
                    break
                new_pop.append(self._random_solution())

        new_pop.sort(key=lambda s: s.cost)
        self.population = new_pop

        # Update best
        self._update_best()

        if self.verbose and self.generation % 50 == 0:
            print(
                f"Gen {self.generation:4d} | "
                f"FE: {self.fe_counter}/{self.max_fe} | "
                f"Best: {self.best_solution.cost:.2f} | "
                f"Pop best: {self.population[0].cost:.2f}"
            )

    def _update_best(self) -> None:
        """Update the global best solution if current population has a better one."""
        if self.population and self.population[0].cost < self.best_solution.cost:
            self.best_solution = self.population[0].copy()
            self.best_found_at_generation = self.generation
            self.best_found_at_fe = self.fe_counter
            # Emit a 'new_best' event so the GUI can refresh its routes view.
            # We do NOT include fast-changing fields (full history) here to keep
            # the queue payload small.
            self._emit_progress(event="new_best", include_routes=True)

    def _emit_progress(
        self,
        event: str = "step",
        include_routes: bool = False,
    ) -> None:
        """Send a progress event to the GUI callback (if any).

        The callback runs in the WORKER thread; the GUI is expected to push
        the payload into a thread-safe queue and process it in the main loop.
        Cheap, allocations except for routes are kept minimal.

        Routes are included when ``include_routes`` is True OR periodically
        every ``ROUTES_SNAPSHOT_INTERVAL`` generations (so the user sees
        route evolution live, not only on new global bests).
        """
        if self.progress_callback is None:
            return
        best_cost = (
            self.best_solution.cost
            if self.best_solution is not None
            else float("inf")
        )
        current_best = (
            self.population[0].cost
            if self.population else float("inf")
        )
        elapsed = (
            time.time() - self.run_start_time
            if self.run_start_time > 0 else 0.0
        )
        info: Dict[str, Any] = {
            "event": event,
            "run_id": self.run_id_for_callback,
            "generation": self.generation,
            "fe": self.fe_counter,
            "max_fe": self.max_fe,
            "best_cost": best_cost,
            "current_best": current_best,
            "elapsed_sec": elapsed,
        }
        # Append the freshly recorded history point (only for "step" events
        # it is meaningful; for "new_best" we skip to keep payload small).
        if event == "step" and self.history:
            info["history_point"] = self.history[-1]

        # Include routes on explicit request OR periodic snapshot
        send_routes = include_routes or (
            event == "step"
            and self.generation > 0
            and self.generation % ROUTES_SNAPSHOT_INTERVAL == 0
        )
        if send_routes and self.best_solution is not None:
            # NOTE: routes is a list of lists of ints — cheap to copy.
            info["routes"] = [list(r) for r in self.best_solution.routes]
            info["best_found_at_fe"] = self.best_found_at_fe
            info["best_found_at_gen"] = self.best_found_at_generation
        self.progress_callback(info)

    def run(self) -> Tuple[Solution, List]:
        """
        Run the full CLONALG algorithm until FE budget exhausted or stop_check
        requests termination.

        Returns:
            (best_solution, history)
            where history is list of (fe, best_cost_so_far)
        """
        self.initialize()
        self.run_start_time = time.time()
        # Emit "start" event for the GUI
        self._emit_progress(event="start", include_routes=False)

        stopped = False
        while self.fe_counter < self.max_fe:
            if self.stop_check is not None and self.stop_check():
                stopped = True
                break
            self.step()

        # Ensure final state is recorded
        self.history.append(
            (self.fe_counter, self.best_solution.cost)
        )

        if self.verbose:
            print(
                f"Done | FE: {self.fe_counter}/{self.max_fe} | "
                f"Best: {self.best_solution.cost:.2f} | "
                f"Routes: {self.best_solution.num_routes}"
                f"{' | STOPPED' if stopped else ''}"
            )

        # Final progress event so the UI sees the last state.
        self._emit_progress(
            event="run_stopped" if stopped else "end",
            include_routes=True,
        )

        return self.best_solution, self.history
