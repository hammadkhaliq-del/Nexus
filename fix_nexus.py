import os
from pathlib import Path

# Define the paths
base_path = Path(__file__).parent
core_path = base_path / "core"

# Ensure core directory exists
if not core_path.exists():
    print(f"❌ Error: Could not find 'core' directory at {core_path}")
    exit(1)

# ==========================================
# 1. CONTENT FOR core/simulation.py
# (Contains the missing 'run' method)
# ==========================================
simulation_code = """
import time
from typing import List, Dict, Optional, Callable, Any, Tuple
from dataclasses import dataclass, field
from .city import City
from .graph import CityGraph
from .agent import Agent, AgentState

@dataclass
class SimulationEvent:
    event_type: str
    position: Tuple[int, int]
    start_time: int
    duration: int = -1
    data: Dict[str, Any] = field(default_factory=dict)
    active: bool = False

    def is_active_at(self, current_step: int) -> bool:
        if not self.active:
            return False
        if self.duration == -1:
            return True
        return current_step < (self.start_time + self.duration)

class Simulation:
    def __init__(self, city: City, graph: Optional[CityGraph] = None):
        self.city = city
        self.graph = graph or CityGraph(city)
        self.agents: List[Agent] = []
        self.agent_registry: Dict[str, Agent] = {}
        self.initial_agent_positions: Dict[str, Tuple[int, int]] = {}
        self.current_step: int = 0
        self.is_running: bool = False
        self.max_steps: Optional[int] = None
        self.events: List[SimulationEvent] = []
        self.event_handlers: Dict[str, Dict[str, Optional[Callable]]] = {}
        self.logs: List[Dict[str, Any]] = []
        self.statistics: Dict[str, Any] = {
            "steps_completed": 0, "total_distance_traveled": 0.0,
            "total_goals_reached": 0, "active_agents": 0,
            "events_triggered": 0, "offline_agents": 0,
            "average_energy": 0.0, "average_distance": 0.0,
            "collision_events": 0
        }
        self.step_callbacks: List[Callable] = []
        self.agent_callbacks: List[Callable] = []
        self.auto_terminate: bool = True
        self.pause_on_error: bool = False

    def add_agent(self, agent: Agent) -> bool:
        if agent.id in self.agent_registry: return False
        r, c = agent.position
        if not self.city.in_bounds(r, c) or not self.city.is_walkable(r, c): return False
        self.agents.append(agent)
        self.agent_registry[agent.id] = agent
        self.initial_agent_positions[agent.id] = agent.position
        self.log(f"Agent {agent.name} added", level="info")
        return True

    def remove_agent(self, agent_id: str) -> bool:
        if agent_id not in self.agent_registry: return False
        agent = self.agent_registry.pop(agent_id)
        self.initial_agent_positions.pop(agent_id, None)
        if agent in self.agents: self.agents.remove(agent)
        return True

    def schedule_event(self, event: SimulationEvent) -> None:
        self.events.append(event)

    def register_event_handler(self, event_type: str, on_activate, on_deactivate=None) -> None:
        self.event_handlers[event_type] = {"activate": on_activate, "deactivate": on_deactivate}

    def register_step_callback(self, callback) -> None:
        self.step_callbacks.append(callback)

    def register_agent_callback(self, callback) -> None:
        self.agent_callbacks.append(callback)

    def start(self, max_steps: Optional[int] = None) -> None:
        self.max_steps = max_steps
        self.is_running = True
        self.log("Simulation started", level="info")

    def stop(self) -> None:
        self.is_running = False
        self.log("Simulation stopped", level="info")

    def pause(self) -> None:
        self.is_running = False

    def resume(self) -> None:
        self.is_running = True

    # --- THE CRITICAL RUN METHOD ---
    def run(self, max_steps: Optional[int] = None, delay: float = 0.0) -> None:
        self.start(max_steps=max_steps)
        try:
            while self.is_running:
                if not self.step(): break
                if delay > 0: time.sleep(delay)
        except KeyboardInterrupt:
            self.stop()

    def step(self) -> bool:
        if not self.is_running: return False
        self.current_step += 1
        self._process_events()
        
        for agent in sorted(self.agents, key=lambda a: a.id):
            if agent.state == AgentState.OFFLINE: continue
            try:
                agent.sense(self.city)
                moved = False
                if agent.state == AgentState.MOVING and agent.path:
                    agent.move_step()
                    moved = True
                for cb in self.agent_callbacks: cb(agent, moved)
            except Exception as e:
                self.log(f"Error agent {agent.name}: {e}", "error")
                if self.pause_on_error: self.pause()

        for cb in self.step_callbacks: cb(self, self.current_step)
        self._update_statistics()

        if self.max_steps and self.current_step >= self.max_steps:
            self.stop(); return False
        
        if self.auto_terminate:
            active = [a for a in self.agents if a.state != AgentState.OFFLINE]
            if not any(a.state == AgentState.MOVING or (a.goal and not a.is_at_goal()) for a in active):
                self.stop(); return False
        return True

    def _process_events(self) -> None:
        for event in self.events:
            if event.start_time == self.current_step and not event.active:
                event.active = True
                self.statistics["events_triggered"] += 1
                h = self.event_handlers.get(event.event_type, {}).get("activate")
                if h: h(self, event)
            if event.active and event.duration != -1 and self.current_step >= (event.start_time + event.duration):
                event.active = False
                h = self.event_handlers.get(event.event_type, {}).get("deactivate")
                if h: h(self, event)

    def _update_statistics(self) -> None:
        self.statistics["steps_completed"] = self.current_step
        self.statistics["total_distance_traveled"] = sum(a.distance_traveled for a in self.agents)
        self.statistics["total_goals_reached"] = sum(a.goals_completed for a in self.agents)
        self.statistics["active_agents"] = len([a for a in self.agents if a.state != AgentState.OFFLINE])
        self.statistics["average_energy"] = (sum(a.energy for a in self.agents) / len(self.agents)) if self.agents else 0
    
    def log(self, message: str, level: str = "info") -> None:
        self.logs.append({"step": self.current_step, "level": level, "message": message})

    def get_logs(self, level=None) -> List[Dict]:
        return [l for l in self.logs if l["level"] == level] if level else self.logs

    def get_statistics(self) -> Dict: return self.statistics.copy()
    def get_status(self) -> Dict: return self.get_statistics()
"""

# ==========================================
# 2. CONTENT FOR core/demo_simulation.py
# (Contains absolute imports and path fix)
# ==========================================
demo_code = """
import numpy as np
import sys
from pathlib import Path

# FIX: Add project root to path
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from core.city import City
from core.graph import CityGraph
from core.agent import Agent
from core.simulation import Simulation, SimulationEvent
from core.utils import (
    manhattan_distance, euclidean_distance,
    generate_walkable_position, calculate_path_length
)

def run_all_tests():
    print("🚀 NEXUS SYSTEM TEST")
    
    # 1. Setup
    grid = np.zeros((20, 20), dtype=np.int32)
    grid[5:8, 5:8] = 1 
    np.save("test_city_map.npy", grid)
    
    city = City("test_city_map.npy")
    graph = CityGraph(city)
    sim = Simulation(city, graph)
    
    # 2. Add Agent
    agent = Agent("TestBot", start_pos=(2, 2))
    agent.set_path([(2,2), (2,3), (2,4), (3,4)])
    sim.add_agent(agent)
    
    # 3. Run
    print("▶️ Running simulation...")
    sim.run(max_steps=10, delay=0.05)
    
    print("✅ Simulation finished.")
    stats = sim.get_statistics()
    print(f"Steps: {stats['steps_completed']}")
    return True

if __name__ == "__main__":
    run_all_tests()
"""

# Write the files
print("⏳ Fixing Nexus Core files...")

with open(core_path / "simulation.py", "w", encoding='utf-8') as f:
    f.write(simulation_code.strip())
print("   ✓ Fixed core/simulation.py (Added 'run' method)")

with open(core_path / "demo_simulation.py", "w", encoding='utf-8') as f:
    f.write(demo_code.strip())
print("   ✓ Fixed core/demo_simulation.py (Fixed imports)")

print("\n✨ REPAIR COMPLETE. You can now run:")
print("python -m core.demo_simulation")