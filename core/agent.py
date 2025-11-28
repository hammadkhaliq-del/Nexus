"""
NEXUS Core Module: Agent
Full-featured agent for NEXUS simulation.
"""

from typing import Tuple, List, Optional, Any
from enum import Enum
import math
import uuid
from .utils import euclidean_distance, manhattan_distance, chebyshev_distance, interpolate_positions

class AgentState(Enum):
    """Agent states in simulation."""
    IDLE = 0
    MOVING = 1
    WORKING = 2
    CHARGING = 3
    OFFLINE = 4

class Agent:
    """
    Represents a mobile agent in the city grid.

    COORDINATE SYSTEM:
        All positions are (row, col), matching City grid indexing.
    """

    def __init__(
        self,
        name: str,
        start_pos: Tuple[int, int],
        speed: float = 1.0,
        max_energy: float = 100.0,
        state: AgentState = AgentState.IDLE
    ):
        # Identity
        self.id = str(uuid.uuid4())[:8]
        self.name = name

        # Position and movement
        self.position: Tuple[int, int] = start_pos
        self.previous_position: Tuple[int, int] = start_pos
        self.goal: Optional[Tuple[int, int]] = None
        self.speed: float = speed

        # Path following
        self.path: List[Tuple[int, int]] = []
        self.path_index: int = 0

        # State
        self.state: AgentState = state

        # Resources
        self.energy: float = max_energy
        self.max_energy: float = max_energy
        self.energy_consumption_rate: float = 0.1

        # Sensors and perception
        self.sensor_range: int = 5
        self.visible_area: List[Tuple[int, int]] = []
        self.known_obstacles: set[Tuple[int, int]] = set()

        # Planning and reasoning
        self.goals_history: List[Tuple[int, int]] = []
        self.actions_taken: List[dict] = []
        self.beliefs: dict = {}
        self.plan: List = []

        # Statistics
        self.distance_traveled: float = 0.0
        self.steps_taken: int = 0
        self.goals_completed: int = 0

    # ------------------- Movement -------------------

    def set_path(self, path: List[Tuple[int, int]]) -> None:
        """Assign a path for the agent to follow."""
        self.path = path
        self.path_index = 0
        if path:
            self.state = AgentState.MOVING

    def move_step(self) -> None:
        """Move agent along its path, updating energy, statistics, and previous position."""
        if not self.path or self.path_index >= len(self.path):
            self.state = AgentState.IDLE
            return

        if self.energy <= 0:
            self.state = AgentState.OFFLINE
            return

        target = self.path[self.path_index]
        dist = euclidean_distance(self.position, target)

        if dist <= self.speed:
            # Arrive at target
            self.previous_position = self.position
            self.position = target
            self.path_index += 1

            self.distance_traveled += dist
            self.steps_taken += 1
            self.consume_energy(self.energy_consumption_rate * dist)

            self.actions_taken.append({
                "type": "move",
                "from": self.previous_position,
                "to": self.position,
                "step": self.steps_taken
            })

            if self.path_index >= len(self.path):
                self.state = AgentState.IDLE
                if self.goal and self.position == self.goal:
                    self.goals_completed += 1
        else:
            # Move proportionally toward target
            steps = max(int(dist / self.speed), 1)
            interp_positions = interpolate_positions(self.position, target, steps)
            self.previous_position = self.position
            self.position = interp_positions[1]

            step_dist = euclidean_distance(self.previous_position, self.position)
            self.distance_traveled += step_dist
            self.steps_taken += 1
            self.consume_energy(self.energy_consumption_rate * step_dist)

    def move_to(self, position: Tuple[int, int]) -> bool:
        """Directly move to adjacent position."""
        dist = euclidean_distance(self.position, position)
        if dist > 1.5:
            return False

        if self.energy <= 0:
            self.state = AgentState.OFFLINE
            return False

        self.previous_position = self.position
        self.position = position
        self.steps_taken += 1
        self.distance_traveled += dist
        self.consume_energy(self.energy_consumption_rate * dist)
        return True

    def wait(self) -> None:
        """Idle for one step."""
        self.state = AgentState.IDLE
        self.steps_taken += 1
        self.consume_energy(self.energy_consumption_rate * 0.1)

    # ------------------- Energy -------------------

    def consume_energy(self, amount: float) -> None:
        """Consume energy and update state if depleted."""
        self.energy = max(0, self.energy - amount)
        if self.energy == 0:
            self.state = AgentState.OFFLINE

    def recharge_energy(self, amount: float) -> None:
        """Recharge energy."""
        self.energy = min(self.max_energy, self.energy + amount)
        if self.state == AgentState.OFFLINE and self.energy > 0:
            self.state = AgentState.IDLE

    def get_energy_percent(self) -> float:
        return (self.energy / self.max_energy) * 100

    # ------------------- Goal Management -------------------

    def set_goal(self, goal: Tuple[int, int]) -> None:
        self.goal = goal
        self.goals_history.append(goal)
        self.state = AgentState.MOVING

    def is_at_goal(self) -> bool:
        return self.goal is not None and self.position == self.goal

    def distance_to_goal(self) -> Optional[float]:
        return euclidean_distance(self.position, self.goal) if self.goal else None

    # ------------------- Sensors / Perception -------------------

    def sense(self, city) -> List[Tuple[int, int]]:
        """Sense surrounding area within sensor range."""
        row, col = self.position
        visible = []

        for dr in range(-self.sensor_range, self.sensor_range + 1):
            for dc in range(-self.sensor_range, self.sensor_range + 1):
                nr, nc = row + dr, col + dc
                if city.in_bounds(nr, nc):
                    if euclidean_distance(self.position, (nr, nc)) <= self.sensor_range:
                        visible.append((nr, nc))
                        if not city.is_walkable(nr, nc):
                            self.known_obstacles.add((nr, nc))

        self.visible_area = visible
        return visible

    # ------------------- Beliefs -------------------

    def update_belief(self, key: str, value: Any) -> None:
        self.beliefs[key] = {"value": value, "step": self.steps_taken}

    def get_belief(self, key: str) -> Optional[Any]:
        belief = self.beliefs.get(key)
        return belief["value"] if belief else None

    # ------------------- Reset / Status -------------------

    def reset(self, position: Optional[Tuple[int, int]] = None) -> None:
        """Reset agent to initial state."""
        if position:
            self.position = position
            self.previous_position = position

        self.goal = None
        self.path = []
        self.path_index = 0
        self.state = AgentState.IDLE

        self.energy = self.max_energy
        self.goals_history.clear()
        self.actions_taken.clear()
        self.beliefs.clear()
        self.visible_area.clear()
        self.known_obstacles.clear()
        self.distance_traveled = 0.0
        self.steps_taken = 0
        self.goals_completed = 0

    def reset_to_idle(self) -> None:
        """Reset agent to idle state without clearing history."""
        self.state = AgentState.IDLE
        self.goal = None
        self.path = []
        self.path_index = 0

    def get_status(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "position": self.position,
            "goal": self.goal,
            "state": self.state.name,
            "energy": round(self.energy, 2),
            "energy_percent": round(self.get_energy_percent(), 1),
            "path_progress": f"{self.path_index}/{len(self.path)}",
            "distance_traveled": round(self.distance_traveled, 2),
            "steps_taken": self.steps_taken,
            "goals_completed": self.goals_completed,
            "sensor_range": self.sensor_range,
            "known_obstacles": len(self.known_obstacles)
        }

    def __repr__(self) -> str:
        return f"<Agent {self.name} pos={self.position} state={self.state.name} energy={self.energy:.1f}>"
