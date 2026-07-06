"""
Mutation operators for the Immunological Algorithm applied to CVRP.

Operators work on permutations (giant tours). After mutation,
the Split algorithm in Solution recomputes the optimal feasible routes.

Design: mutation operators are split into two layers:
1. _perm_ functions: work directly on a list (permutation), in-place, O(1)-O(n)
2. Solution-level wrappers: copy + mutate + _split() — used for standalone calls

hypermutation() uses the _perm_ functions to apply N mutations cheaply,
then calls _split() only ONCE at the end.
"""
import random
from typing import List, Callable

from src.solution import Solution


# ============================================================
# Permutation-only mutation primitives (in-place, no _split)
# ============================================================

def _perm_swap(perm: List[int]) -> None:
    """Swap two random positions in-place."""
    n = len(perm)
    if n < 2:
        return
    i, j = random.sample(range(n), 2)
    perm[i], perm[j] = perm[j], perm[i]


def _perm_inversion(perm: List[int]) -> None:
    """Reverse a random segment in-place."""
    n = len(perm)
    if n < 2:
        return
    i, j = sorted(random.sample(range(n), 2))
    perm[i : j + 1] = reversed(perm[i : j + 1])


def _perm_insertion(perm: List[int]) -> None:
    """Remove a random element and re-insert at a random position, in-place."""
    n = len(perm)
    if n < 2:
        return
    i = random.randrange(n)
    customer = perm.pop(i)
    j = random.randrange(n - 1)  # n-1 after pop
    perm.insert(j, customer)


def _perm_scramble(perm: List[int]) -> None:
    """Scramble (randomly shuffle) a random sub-segment in-place."""
    n = len(perm)
    if n < 2:
        return
    i, j = sorted(random.sample(range(n), 2))
    sub = perm[i : j + 1]
    random.shuffle(sub)
    perm[i : j + 1] = sub


# List of permutation-only operators for hypermutation
PERM_OPERATORS = [
    _perm_swap,
    _perm_inversion,
    _perm_insertion,
    _perm_scramble,
]


# ============================================================
# Solution-level mutation wrappers (copy + mutate + _split)
# These are kept for standalone use / testing
# ============================================================

def swap_mutation(solution: Solution) -> Solution:
    """Swap two random positions in the permutation."""
    new_sol = solution.copy()
    _perm_swap(new_sol.permutation)
    new_sol._split()
    return new_sol


def inversion_mutation(solution: Solution) -> Solution:
    """Reverse a random segment of the permutation."""
    new_sol = solution.copy()
    _perm_inversion(new_sol.permutation)
    new_sol._split()
    return new_sol


def insertion_mutation(solution: Solution) -> Solution:
    """Remove a random customer and re-insert at a random position."""
    new_sol = solution.copy()
    _perm_insertion(new_sol.permutation)
    new_sol._split()
    return new_sol


def scramble_mutation(solution: Solution) -> Solution:
    """Scramble (randomly shuffle) a random sub-segment."""
    new_sol = solution.copy()
    _perm_scramble(new_sol.permutation)
    new_sol._split()
    return new_sol


def two_opt_intra_route(solution: Solution, max_iterations: int = 20) -> Solution:
    """
    2-Opt improvement applied to each route independently.
    Uses first-improvement strategy for speed.
    After improvement, rebuilds the permutation from the modified routes.

    Args:
        max_iterations: Cap on improvement rounds to bound execution time.
    """
    new_sol = solution.copy()
    dist = new_sol.instance.dist_matrix
    depot = new_sol.instance.depot
    improved = True

    iteration = 0
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        for r_idx, route in enumerate(new_sol.routes):
            if len(route) < 3:
                continue
            # Build the full path: depot, route[0], ..., route[-1], depot
            path = [depot] + route + [depot]
            m = len(path)

            found = False
            for i in range(1, m - 2):
                for j in range(i + 1, m - 1):
                    old_cost = (
                        dist[path[i - 1]][path[i]]
                        + dist[path[j]][path[j + 1]]
                    )
                    new_cost = (
                        dist[path[i - 1]][path[j]]
                        + dist[path[i]][path[j + 1]]
                    )
                    gain = old_cost - new_cost

                    if gain > 1e-9:
                        # Reverse segment between i and j (first-improvement)
                        path[i : j + 1] = reversed(path[i : j + 1])
                        new_sol.routes[r_idx] = path[1:-1]
                        improved = True
                        found = True
                        break
                if found:
                    break

    # Recompute cost from routes
    new_sol.cost = sum(
        _route_cost(r, depot, dist) for r in new_sol.routes
    )

    # Rebuild permutation from the modified routes to keep consistency
    new_sol.permutation = []
    for route in new_sol.routes:
        new_sol.permutation.extend(route)

    return new_sol


def _route_cost(
    route: List[int], depot: int, dist
) -> float:
    if not route:
        return 0.0
    cost = dist[depot][route[0]]
    for i in range(len(route) - 1):
        cost += dist[route[i]][route[i + 1]]
    cost += dist[route[-1]][depot]
    return cost


# ============================================================
# Hypermutation
# ============================================================

def hypermutation(
    solution: Solution,
    mutation_rate: float,
    operators: List[Callable],  # ignored, uses PERM_OPERATORS
) -> Solution:
    """
    Apply hypermutation to a clone.

    The number of mutations is proportional to mutation_rate.
    Each mutation picks a random permutation-level operator.
    Only ONE _split() call is made at the end.

    Args:
        solution: The clone to mutate
        mutation_rate: Value in [0, 1]; higher = more mutations
        operators: Kept for API compatibility; uses PERM_OPERATORS internally

    Returns:
        Mutated solution (with _split already applied)
    """
    current = solution.copy()

    # Number of mutation steps proportional to mutation_rate.
    # With blind mutation (no internal evaluation), keep mutations small
    # to preserve solution structure: 1 mutation for best antibodies,
    # up to MAX_MUTATIONS for worst.
    MAX_MUTATIONS = 5
    n_mutations = max(1, round(mutation_rate * MAX_MUTATIONS))

    # Apply all mutations on the permutation only (no _split)
    for _ in range(n_mutations):
        op = random.choice(PERM_OPERATORS)
        op(current.permutation)

    # Single _split() call at the end — this is the ONE fitness evaluation
    current._split()

    return current
