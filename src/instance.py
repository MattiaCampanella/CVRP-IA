"""
Instance parser for CVRP instances in TSPLIB format.
"""
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from src.utils import compute_distance_matrix


@dataclass
class Instance:
    """
    Represents a CVRP instance.
    """
    name: str
    comment: str
    dimension: int
    capacity: int
    edge_weight_type: str
    coordinates: List[Tuple[float, float]]
    demands: List[float]
    depot: int
    dist_matrix: np.ndarray | None = None

    def __post_init__(self):
        if self.dist_matrix is None:
            self.dist_matrix = compute_distance_matrix(
                self.coordinates, self.edge_weight_type
            )

    @staticmethod
    def from_file(filepath: str) -> "Instance":
        """Parse a CVRP instance from a .vrp file (TSPLIB format)."""
        with open(filepath, "r") as f:
            lines = f.readlines()

        # Default values
        name = ""
        comment = ""
        dimension = 0
        capacity = 0
        edge_weight_type = "EUC_2D"
        depot_idx = 0

        section = None
        coordinates: dict[int, Tuple[float, float]] = {}
        demands: dict[int, float] = {}
        depot_section_data: list[int] = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if line.startswith("NAME"):
                parts = line.split(":", 1)
                name = parts[1].strip() if len(parts) > 1 else ""
            elif line.startswith("COMMENT"):
                parts = line.split(":", 1)
                comment = parts[1].strip() if len(parts) > 1 else ""
            elif line.startswith("DIMENSION"):
                parts = line.split(":", 1)
                dimension = int(parts[1].strip()) if len(parts) > 1 else 0
            elif line.startswith("CAPACITY"):
                parts = line.split(":", 1)
                capacity = int(parts[1].strip()) if len(parts) > 1 else 0
            elif line.startswith("EDGE_WEIGHT_TYPE"):
                parts = line.split(":", 1)
                edge_weight_type = (
                    parts[1].strip() if len(parts) > 1 else "EUC_2D"
                )
            elif line.startswith("NODE_COORD_SECTION"):
                section = "COORDS"
                continue
            elif line.startswith("DEMAND_SECTION"):
                section = "DEMANDS"
                continue
            elif line.startswith("DEPOT_SECTION"):
                section = "DEPOT"
                continue
            elif line.startswith("EOF"):
                break

            if section == "COORDS":
                parts = line.split()
                if len(parts) >= 3:
                    node_id = int(parts[0])
                    x = float(parts[1])
                    y = float(parts[2])
                    coordinates[node_id] = (x, y)

            elif section == "DEMANDS":
                parts = line.split()
                if len(parts) >= 2:
                    node_id = int(parts[0])
                    demand = float(parts[1])
                    demands[node_id] = demand

            elif section == "DEPOT":
                if line.strip() == "-1":
                    section = None
                    continue
                try:
                    depot_section_data.append(int(line.strip()))
                except ValueError:
                    pass

        # The depot is typically node 1
        # TSPLIB files are 1-indexed, we convert to 0-indexed
        if depot_section_data:
            depot_idx = depot_section_data[0] - 1  # 0-indexed
        else:
            depot_idx = 0

        # Build coordinate list (0-indexed)
        n = len(coordinates)
        coord_list: List[Tuple[float, float]] = [(0.0, 0.0)] * n
        for node_id, (x, y) in coordinates.items():
            coord_list[node_id - 1] = (x, y)

        # Build demand list (0-indexed)
        demand_list: List[float] = [0.0] * n
        for node_id, d in demands.items():
            demand_list[node_id - 1] = d

        # Compute distance matrix
        dist_matrix = compute_distance_matrix(coord_list, edge_weight_type)

        return Instance(
            name=name,
            comment=comment,
            dimension=dimension,
            capacity=capacity,
            edge_weight_type=edge_weight_type,
            coordinates=coord_list,
            demands=demand_list,
            depot=depot_idx,
            dist_matrix=dist_matrix,
        )
