"""
NEXUS Core Module: City Map Manager
Handles loading, querying, and managing the city grid environment.
"""

import numpy as np
from typing import Tuple, Set, Optional, List, Dict
from pathlib import Path


class City:
    """
    Represents the city environment as a 2D grid.

    COORDINATE SYSTEM (NEXUS STANDARD):
        All coordinates are (row, col)
        This matches NumPy indexing and makes AI modules consistent.

    Nexus Tile Encoding:
        0 = Road (walkable)
        1 = Building (blocked)
        2 = Grass/Park (walkable)
        3 = Water (blocked)
        4 = Highway (walkable)
        5 = Restricted Zone (blocked)
    """

    # Tile type constants
    ROAD = 0
    BUILDING = 1
    GRASS = 2
    WATER = 3
    HIGHWAY = 4
    RESTRICTED = 5

    DEFAULT_WALKABLE = {ROAD, GRASS, HIGHWAY}

    def __init__(self, map_path: Optional[str] = None):
        self.width = 0
        self.height = 0
        self.grid: Optional[np.ndarray] = None
        self._base_grid: Optional[np.ndarray] = None
        self.dynamic_blocked: Set[Tuple[int, int]] = set()
        self.walkable_values = set(self.DEFAULT_WALKABLE)
        self.tile_labels: Dict[int, str] = {
            self.ROAD: "road",
            self.BUILDING: "building",
            self.GRASS: "grass",
            self.WATER: "water",
            self.HIGHWAY: "highway",
            self.RESTRICTED: "restricted_zone",
        }

        if map_path:
            self.load(map_path)

    # -------------------- Loading / Saving --------------------

    def load(self, map_path: str) -> None:
        """Load city map from .npy file."""
        path = Path(map_path)
        if not path.exists():
            raise FileNotFoundError(f"City map not found: {map_path}")

        arr = np.load(str(path))
        if arr.ndim != 2:
            raise ValueError("City map must be a 2D matrix")
        arr = arr.astype(np.int32, copy=False)

        allowed = set(self.tile_labels.keys())
        unique = set(np.unique(arr).tolist())
        invalid = unique - allowed
        if invalid:
            raise ValueError(f"Invalid tile values in map: {invalid}")

        self._base_grid = arr.copy()
        self._base_grid.setflags(write=False)
        self.grid = arr.copy()
        self.height, self.width = self.grid.shape
        self.dynamic_blocked.clear()

    def save(self, path: str) -> None:
        """Save current working grid to .npy file."""
        np.save(path, self.grid)

    # -------------------- Reset / Dynamic Tiles --------------------

    def reset_to_base(self) -> None:
        """Reset working grid to original base map."""
        if self._base_grid is None:
            raise RuntimeError("No base map loaded")
        self.grid = self._base_grid.copy()
        self.reset_dynamic_blocked()

    def reset_dynamic_blocked(self) -> None:
        """Clear all dynamic blocks."""
        self.dynamic_blocked.clear()

    # -------------------- Queries --------------------

    def in_bounds(self, row: int, col: int) -> bool:
        """Check if position is within grid bounds."""
        return 0 <= row < self.height and 0 <= col < self.width

    def is_walkable(self, row: int, col: int) -> bool:
        """Check if a tile is walkable."""
        if not self.in_bounds(row, col):
            return False
        return (self.grid[row, col] in self.walkable_values and 
                (row, col) not in self.dynamic_blocked)

    def get_value(self, row: int, col: int) -> int:
        """Get tile value at position."""
        if not self.in_bounds(row, col):
            raise IndexError(f"Out of bounds: ({row}, {col})")
        return int(self.grid[row, col])

    def set_value(self, row: int, col: int, value: int) -> None:
        """Set tile value at position."""
        if not self.in_bounds(row, col):
            raise IndexError(f"Out of bounds: ({row}, {col})")
        if value not in self.tile_labels:
            raise ValueError(f"Invalid tile type: {value}")
        self.grid[row, col] = int(value)

    def neighbors(self, row: int, col: int, diagonal: bool = False) -> List[Tuple[int, int]]:
        """Get all neighbors (not filtered by walkability)."""
        deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        if diagonal:
            deltas += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        return [(row+dr, col+dc) for dr, dc in deltas if self.in_bounds(row+dr, col+dc)]

    def walkable_neighbors(self, row: int, col: int, diagonal: bool = False) -> List[Tuple[int, int]]:
        """Return list of walkable neighbor positions."""
        return [(r, c) for r, c in self.neighbors(row, col, diagonal) if self.is_walkable(r, c)]

    def get_walkable_positions(self) -> List[Tuple[int, int]]:
        """Return all currently walkable positions."""
        positions = []
        for row in range(self.height):
            for col in range(self.width):
                if self.is_walkable(row, col):
                    positions.append((row, col))
        return positions

    def find_tiles(self, tile_type: int) -> List[Tuple[int, int]]:
        """Find all positions of a specific tile type."""
        if tile_type not in self.tile_labels:
            return []
        coords = np.argwhere(self.grid == tile_type)
        return [(int(r), int(c)) for r, c in coords]

    def get_stats(self) -> Dict:
        """Get statistics about tile distribution."""
        unique, counts = np.unique(self.grid, return_counts=True)
        total = self.width * self.height
        dist = {
            self.tile_labels[int(v)]: {
                "count": int(c),
                "percentage": round((int(c) / total) * 100, 2),
            }
            for v, c in zip(unique, counts)
        }
        return {
            "dimensions": f"{self.width}x{self.height}",
            "total_tiles": total,
            "tile_distribution": dist,
        }

    # -------------------- Utility --------------------

    def to_matrix(self) -> np.ndarray:
        """Return copy of current working grid."""
        return self.grid.copy()

    def block_tile(self, row: int, col: int) -> None:
        """Temporarily block a tile."""
        if self.in_bounds(row, col):
            self.dynamic_blocked.add((row, col))

    def unblock_tile(self, row: int, col: int) -> None:
        """Remove temporary block from a tile."""
        self.dynamic_blocked.discard((row, col))

    def create_random_obstacles(
        self, 
        obstacle_ratio: float = 0.1, 
        obstacle_type: Optional[int] = None, 
        seed: Optional[int] = None
    ) -> None:
        """Randomly place obstacles on walkable tiles only."""
        if obstacle_type is None:
            obstacle_type = self.BUILDING

        if seed is not None:
            np.random.seed(seed)

        walkable_positions = self.get_walkable_positions()
        num_obstacles = int(len(walkable_positions) * obstacle_ratio)
        if num_obstacles > 0:
            obstacle_indices = np.random.choice(len(walkable_positions), size=num_obstacles, replace=False)
            for idx in obstacle_indices:
                r, c = walkable_positions[idx]
                self.grid[r, c] = obstacle_type

    def __repr__(self) -> str:
        return f"<City {self.width}x{self.height}, walkable_tiles={len(self.get_walkable_positions())}>"
