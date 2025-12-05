"""
NEXUS Core Package
Exposes unified import interface for the simulation engine.
"""

from .city import City
from .graph import CityGraph
from .agent import Agent, AgentState
from .simulation import Simulation, SimulationEvent
from .utils import (
    manhattan_distance,
    euclidean_distance,
    chebyshev_distance,
    generate_random_position,
    generate_walkable_position,
    get_neighbors_4,
    get_neighbors_8,
    is_adjacent,
    grid_to_graph_coords,
    graph_to_grid_coords,
    reconstruct_path,
    create_empty_grid,
    create_random_grid,
    flood_fill,
    get_random_subset,
    normalize_vector,
    interpolate_positions,
    calculate_path_length,
    smooth_path,
    get_grid_bounds,
    clamp,
    random_chance,
)
