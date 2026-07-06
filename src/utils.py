"""
Utility functions for CVRP Immunological Algorithm.
"""
import math
import random
from typing import List, Tuple

import numpy as np


def euclidean_distance(
    x1: float, y1: float, x2: float, y2: float
) -> float:
    """Compute Euclidean distance between two points."""
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def compute_distance_matrix(
    coordinates: List[Tuple[float, float]],
    edge_weight_type: str = "EUC_2D",
) -> np.ndarray:
    """
    Compute the distance matrix for a set of nodes.
    For EUC_2D, the distance is the Euclidean distance rounded to the
    nearest integer (as per TSPLIB convention).
    """
    n = len(coordinates)
    dist = np.zeros((n, n), dtype=float)

    for i in range(n):
        for j in range(i + 1, n):
            d = euclidean_distance(
                coordinates[i][0], coordinates[i][1],
                coordinates[j][0], coordinates[j][1],
            )
            if edge_weight_type == "EUC_2D":
                d = round(d)
            dist[i][j] = d
            dist[j][i] = d

    return dist


def route_cost(
    route: List[int], depot: int, dist_matrix: np.ndarray
) -> float:
    """
    Compute the cost of a single route:
    depot -> first customer -> ... -> last customer -> depot
    """
    if not route:
        return 0.0

    cost = dist_matrix[depot][route[0]]
    for i in range(len(route) - 1):
        cost += dist_matrix[route[i]][route[i + 1]]
    cost += dist_matrix[route[-1]][depot]
    return cost


def total_cost(
    routes: List[List[int]], depot: int, dist_matrix: np.ndarray
) -> float:
    """Compute total cost of all routes."""
    return sum(route_cost(r, depot, dist_matrix) for r in routes)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
