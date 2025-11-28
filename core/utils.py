"""
NEXUS Core Module: Utility Functions
Helper functions for grid operations, conversions, and common tasks.
Uses (row, col) coordinate ordering consistently across the project.
"""

import numpy as np
import random
from typing import Tuple, List, Optional, Set
import math


# -------------------------------------------------------------
# DISTANCE FUNCTIONS (row, col)
# -------------------------------------------------------------

def manhattan_distance(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    """Manhattan distance between (row, col) positions."""
    r1, c1 = a
    r2, c2 = b
    return abs(r2 - r1) + abs(c2 - c1)


def euclidean_distance(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    """Euclidean distance between (row, col) positions."""
    r1, c1 = a
    r2, c2 = b
    dr = r2 - r1
    dc = c2 - c1
    return math.sqrt(dr * dr + dc * dc)


def chebyshev_distance(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    """Chebyshev distance between (row, col) positions."""
    r1, c1 = a
    r2, c2 = b
    return max(abs(r2 - r1), abs(c2 - c1))


# -------------------------------------------------------------
# RANDOM POSITION HELPERS
# -------------------------------------------------------------

def generate_random_position(height: int, width: int,
                             exclude: Optional[Set[Tuple[int, int]]] = None
                             ) -> Tuple[int, int]:
    """
    Generate a random (row, col) position inside the grid.
    """
    exclude = exclude or set()

    while True:
        r = random.randint(0, height - 1)
        c = random.randint(0, width - 1)
        pos = (r, c)

        if pos not in exclude:
            return pos


def generate_walkable_position(city, exclude: Optional[Set[Tuple[int, int]]] = None
                               ) -> Optional[Tuple[int, int]]:
    """
    Generate random walkable (row, col) position inside the city.
    """
    exclude = exclude or set()
    max_attempts = 1000

    for _ in range(max_attempts):
        pos = generate_random_position(city.height, city.width, exclude)
        if city.is_walkable(*pos):  # pos = (row, col)
            return pos

    return None


# -------------------------------------------------------------
# NEIGHBOR FUNCTIONS (row, col)
# -------------------------------------------------------------

def get_neighbors_4(row: int, col: int) -> List[Tuple[int, int]]:
    """Return 4-way neighbors: N, E, S, W (row-major)."""
    return [
        (row - 1, col),  # North
        (row, col + 1),  # East
        (row + 1, col),  # South
        (row, col - 1),  # West
    ]


def get_neighbors_8(row: int, col: int) -> List[Tuple[int, int]]:
    """Return 8-way neighbors including diagonals."""
    return [
        (row - 1, col),      # N
        (row - 1, col + 1),  # NE
        (row, col + 1),      # E
        (row + 1, col + 1),  # SE
        (row + 1, col),      # S
        (row + 1, col - 1),  # SW
        (row, col - 1),      # W
        (row - 1, col - 1),  # NW
    ]


def is_adjacent(a: Tuple[int, int], b: Tuple[int, int], diagonal: bool = False) -> bool:
    """Check adjacency between two nodes."""
    if diagonal:
        return chebyshev_distance(a, b) == 1
    else:
        return manhattan_distance(a, b) == 1


# -------------------------------------------------------------
# GRID/GRAPH COORDINATE HELPERS
# -------------------------------------------------------------

def grid_to_graph_coords(row: int, col: int) -> Tuple[int, int]:
    """Identity transform (kept for API consistency)."""
    return (row, col)


def graph_to_grid_coords(row: int, col: int) -> Tuple[int, int]:
    """Identity transform (kept for API consistency)."""
    return (row, col)


# -------------------------------------------------------------
# PATH RECONSTRUCTION
# -------------------------------------------------------------

def reconstruct_path(came_from: dict, start: Tuple[int, int],
                     goal: Tuple[int, int]) -> List[Tuple[int, int]]:
    """Reconstruct a path from A* or Dijkstra came_from map."""
    if goal not in came_from:
        return []

    path = []
    cur = goal

    while cur != start:
        path.append(cur)
        cur = came_from[cur]

    path.append(start)
    path.reverse()
    return path


# -------------------------------------------------------------
# GRID CREATION
# -------------------------------------------------------------

def create_empty_grid(width: int, height: int, fill_value: int = 0) -> np.ndarray:
    """Create empty 2D grid with (height, width)."""
    return np.full((height, width), fill_value, dtype=np.int32)


def create_random_grid(width: int, height: int,
                       tile_types: List[int],
                       probabilities: Optional[List[float]] = None) -> np.ndarray:
    """Create random 2D grid."""
    if probabilities is None:
        probabilities = [1.0 / len(tile_types)] * len(tile_types)

    return np.random.choice(tile_types, size=(height, width), p=probabilities)


# -------------------------------------------------------------
# FLOOD FILL (row, col)
# -------------------------------------------------------------

def flood_fill(grid: np.ndarray, start: Tuple[int, int],
               target_value: int, fill_value: int) -> np.ndarray:
    """
    Flood fill on a grid (row-major).
    """
    h, w = grid.shape
    grid = grid.copy()

    sr, sc = start
    if grid[sr, sc] != target_value:
        return grid

    stack = [(sr, sc)]

    while stack:
        r, c = stack.pop()

        if not (0 <= r < h and 0 <= c < w):
            continue

        if grid[r, c] != target_value:
            continue

        grid[r, c] = fill_value

        for nr, nc in get_neighbors_4(r, c):
            stack.append((nr, nc))

    return grid


# -------------------------------------------------------------
# MISC HELPERS
# -------------------------------------------------------------

def get_random_subset(items: List, count: int) -> List:
    return random.sample(items, min(count, len(items)))


def normalize_vector(vector: Tuple[float, float]) -> Tuple[float, float]:
    x, y = vector
    mag = math.sqrt(x * x + y * y)
    return (0.0, 0.0) if mag == 0 else (x / mag, y / mag)


def interpolate_positions(a: Tuple[int, int], b: Tuple[int, int], steps: int)\
        -> List[Tuple[float, float]]:
    """Linear interpolation between two positions (visual helper)."""
    if steps <= 1:
        return [a, b]

    r1, c1 = a
    r2, c2 = b
    out = []

    for i in range(steps + 1):
        t = i / steps
        out.append((r1 + (r2 - r1) * t, c1 + (c2 - c1) * t))

    return out


def calculate_path_length(path: List[Tuple[int, int]]) -> float:
    if len(path) < 2:
        return 0.0
    return sum(euclidean_distance(path[i], path[i + 1])
               for i in range(len(path) - 1))


def smooth_path(path: List[Tuple[int, int]], window: int = 3) -> List[Tuple[int, int]]:
    if len(path) <= window:
        return path

    result = []
    half = window // 2

    for i in range(len(path)):
        s = max(0, i - half)
        e = min(len(path), i + half + 1)
        avg_r = sum(p[0] for p in path[s:e]) / (e - s)
        avg_c = sum(p[1] for p in path[s:e]) / (e - s)
        result.append((int(round(avg_r)), int(round(avg_c))))

    return result


def get_grid_bounds(positions: List[Tuple[int, int]]) -> Tuple[int, int, int, int]:
    if not positions:
        return (0, 0, 0, 0)

    rows = [p[0] for p in positions]
    cols = [p[1] for p in positions]

    return (min(rows), min(cols), max(rows), max(cols))


def clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))


def random_chance(prob: float) -> bool:
    return random.random() < prob
