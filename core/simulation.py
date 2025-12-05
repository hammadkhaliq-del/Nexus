"""
NEXUS Core Module: Simulation Engine
"""

from typing import List, Dict, Optional, Callable, Any, Tuple
from dataclasses import dataclass, field
import time
from .city import City
from .graph import CityGraph
from .agent import Agent, AgentState

@dataclass
class SimulationEvent:
    event_type: str
    position: Tuple[int, int]
    start_time: int
    duration: int = -1
    active: bool = False

class Simulation:
    def __init__(self, city: City, graph: Optional[CityGraph] = None):
        self.city = city
        self.graph = graph or CityGraph(city)
        self.agents: List[Agent] = []
        self.agent_registry: Dict[str, Agent] = {}
        
        # Simulation State
        self.current_step: int = 0
        self.is_running: bool = False
        self.max_steps: Optional[int] = None
        
        # Events & Stats
        self.events: List[SimulationEvent] = []
        self.event_handlers: Dict[str, Dict] = {}
        self.statistics: Dict[str, Any] = {
            "steps_completed": 0,
            "events_triggered": 0,
            "active_agents": 0,
            "total_distance_traveled": 0.0
        }
        
        # Callbacks
        self.step_callbacks: List[Callable] = []
        self.agent_callbacks: List[Callable] = []

    def add_agent(self, agent: Agent) -> bool:
        # Check if position is valid (Casting float pos to int for grid check)
        r, c = int(agent.position[0]), int(agent.position[1])
        if not self.city.in_bounds(r, c) or not self.city.is_walkable(r, c):
            return False
        self.agents.append(agent)
        self.agent_registry[agent.id] = agent
        return True

    def schedule_event(self, event: SimulationEvent):
        self.events.append(event)

    def register_event_handler(self, name, activate, deactivate=None):
        self.event_handlers[name] = {"activate": activate, "deactivate": deactivate}

    def register_step_callback(self, cb):
        self.step_callbacks.append(cb)

    def register_agent_callback(self, cb):
        self.agent_callbacks.append(cb)

    def start(self, max_steps=None):
        self.max_steps = max_steps
        self.is_running = True

    def stop(self):
        self.is_running = False

    # --- THE MAIN LOOP ---
    def run(self, max_steps: Optional[int] = None, delay: float = 0.0):
        self.start(max_steps)
        try:
            while self.is_running:
                # 1. Advance one step
                keep_going = self.step()
                if not keep_going:
                    break
                
                # 2. Optional visual delay
                if delay > 0:
                    time.sleep(delay)
        except KeyboardInterrupt:
            self.stop()

    def step(self) -> bool:
        if not self.is_running: return False
        self.current_step += 1

        # 1. Process Events
        for event in self.events:
            # Activate
            if event.start_time == self.current_step and not event.active:
                event.active = True
                self.statistics["events_triggered"] += 1
                h = self.event_handlers.get(event.event_type, {}).get("activate")
                if h: h(self, event)
            # Deactivate
            if event.active and event.duration != -1 and self.current_step >= (event.start_time + event.duration):
                event.active = False
                h = self.event_handlers.get(event.event_type, {}).get("deactivate")
                if h: h(self, event)

        # 2. Update Agents
        for agent in self.agents:
            if agent.state == AgentState.OFFLINE: continue
            
            # Safe sensing (float -> int handled inside agent usually, but wrapped here)
            try:
                agent.sense(self.city)
                moved = False
                if agent.state == AgentState.MOVING:
                    agent.move_step()
                    moved = True
                # Trigger callbacks
                for cb in self.agent_callbacks: cb(agent, moved)
            except Exception:
                pass # Prevent crash on small math errors

        # 3. Update Stats
        self.statistics["steps_completed"] = self.current_step
        self.statistics["active_agents"] = len(self.agents)
        self.statistics["total_distance_traveled"] = sum(a.distance_traveled for a in self.agents)

        # 4. Trigger Step Callbacks
        for cb in self.step_callbacks:
            cb(self, self.current_step)

        # 5. Check Termination
        if self.max_steps and self.current_step >= self.max_steps:
            self.stop()
            return False

        return True

    def get_statistics(self):
        return self.statistics.copy()