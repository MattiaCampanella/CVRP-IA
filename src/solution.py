"""
Solution representation for CVRP using permutation encoding with Split algorithm.

The Split algorithm (Prins, 2004) optimally partitions a giant tour
(permutation of customers) into feasible vehicle routes respecting
capacity constraints.
"""
import math
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

from src.instance import Instance


@dataclass
class Solution:
    """
    A CVRP solution represented as a permutation of customers.
    The Split algorithm transforms the permutation into feasible routes.
    """
    permutation: List[int]
    instance: Instance
    routes: List[List[int]] = field(default_factory=list)
    cost: float = field(default=float("inf"), init=False)
    feasible: bool = field(default=True, init=False)

    def __post_init__(self):
        self._split()

    def _split(self) -> None:
        """
        Split algorithm (Prins, 2004).

        Given a giant tour (permutation of customers without depot),
        optimally partitions it into feasible routes using DP.

        Time complexity: O(n^2) where n = number of customers.
        """
        n = len(self.permutation)
        if n == 0:
            self.routes = []
            self.cost = 0.0
            self.feasible = True
            return

        depot = self.instance.depot
        capacity = self.instance.capacity
        demands = self.instance.demands
        dist = self.instance.dist_matrix

        # V[i] = minimum cost to serve first i customers (indices 0..i-1)
        V = [float("inf")] * (n + 1)
        pred = [-1] * (n + 1)
        V[0] = 0.0

        for i in range(1, n + 1):
            load = 0.0
            route_cost = 0.0

            for j in range(i - 1, -1, -1):
                customer = self.permutation[j]
                load += demands[customer]
                if load > capacity:
                    break

                if j == i - 1:
                    # Route: depot -> perm[j] -> depot
                    route_cost = (
                        dist[depot][customer]
                        + dist[customer][depot]
                    )
                else:
                    # Extend route: depot -> perm[j] -> perm[j+1] -> ... -> depot
                    prev_first = self.permutation[j + 1]
                    route_cost = (
                        route_cost
                        - dist[depot][prev_first]
                        + dist[depot][customer]
                        + dist[customer][prev_first]
                    )

                prev_cost = V[j]  # j = 0 means no previous customers
                total = prev_cost + route_cost

                if total < V[i]:
                    V[i] = total
                    pred[i] = j

        if V[n] == float("inf"):
            self.cost = float("inf")
            self.routes = []
            self.feasible = False
            return

        self.cost = V[n]
        self.feasible = True

        # Reconstruct routes by backtracking
        routes: List[List[int]] = []
        idx = n
        while idx > 0:
            start = pred[idx]
            route = self.permutation[start:idx]
            routes.append(route)
            idx = start

        routes.reverse()
        self.routes = routes

    def copy(self) -> "Solution":
        """Create a deep copy of this solution."""
        new_sol = Solution.__new__(Solution)
        new_sol.permutation = list(self.permutation)
        new_sol.instance = self.instance
        new_sol.routes = [list(r) for r in self.routes]
        new_sol.cost = self.cost
        new_sol.feasible = self.feasible
        return new_sol

    @property
    def num_routes(self) -> int:
        return len(self.routes)

    @property
    def num_customers_served(self) -> int:
        return sum(len(r) for r in self.routes)

    @property
    def satisfability(self) -> bool:
        """Check if all customers are served."""
        n_customers = self.instance.dimension - 1  # exclude depot
        return self.num_customers_served == n_customers
