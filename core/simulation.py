"""
NEXUS Core Module: Simulation Engine — Full Upgrade

Full-featured, Nexus production-ready simulation engine:
- Separate activate/deactivate handlers
- Track initial agent positions for exact resets
- Pause / resume methods
- Configurable auto_terminate logic and improved termination checks
- Richer statistics (offline, average energy, collisions placeholder)
- Clear documented event lifecycle and safety wrappers for handlers/callbacks
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
    data: Dict[str, Any] = field(default_factory=dict)
    active: bool = False

    def is_active_at(self, current_step: int) -> bool:
        if not self.active:
            return False
        if self.duration == -1:
            return True
        return current_step < (self.start_time + self.duration)

class Simulation:
    """
    Full Nexus-ready simulation engine.

    Event handlers:
      - register_event_handler(event_type, on_activate, on_deactivate=None)
      Activation calls on_activate(sim, event) with event.active == True.
      Deactivation calls on_deactivate(sim, event) with event.active == False (if provided).
    """

    def __init__(self, city: City, graph: Optional[CityGraph] = None):
        self.city = city
        self.graph = graph or CityGraph(city)

        self.agents: List[Agent] = []
        self.agent_registry: Dict[str, Agent] = {}
        # initial positions used for exact reset
        self.initial_agent_positions: Dict[str, Tuple[int, int]] = {}

        self.current_step: int = 0
        self.is_running: bool = False
        self.max_steps: Optional[int] = None

        self.events: List[SimulationEvent] = []
        # event handlers now support separate activate/deactivate
        self.event_handlers: Dict[str, Dict[str, Optional[Callable[[ "Simulation", SimulationEvent ], None]]]] = {}

        self.logs: List[Dict[str, Any]] = []
        self.statistics: Dict[str, Any] = {
            "steps_completed": 0,
            "total_distance_traveled": 0.0,
            "total_goals_reached": 0,
            "active_agents": 0,
            "events_triggered": 0,
            # extras
            "offline_agents": 0,
            "average_energy": 0.0,
            "average_distance": 0.0,
            "collision_events": 0  # placeholder for future collision detection
        }

        self.step_callbacks: List[Callable[[ "Simulation", int ], None]] = []
        self.agent_callbacks: List[Callable[[Agent, bool], None]] = []

        # control flags
        self.auto_terminate: bool = True  # can be disabled
        self.pause_on_error: bool = False  # optional

    # Agent management
    def add_agent(self, agent: Agent) -> bool:
        if agent.id in self.agent_registry:
            self.log(f"Agent {agent.name} already registered", level="warning")
            return False
        r, c = agent.position
        if not self.city.in_bounds(r, c) or not self.city.is_walkable(r, c):
            self.log(f"Agent {agent.name} starting position not walkable: {agent.position}", level="error")
            return False
        self.agents.append(agent)
        self.agent_registry[agent.id] = agent
        # store initial positions for exact reset
        self.initial_agent_positions[agent.id] = agent.position
        self.log(f"Agent {agent.name} (id={agent.id}) added at {agent.position}", level="info")
        return True

    def remove_agent(self, agent_id: str) -> bool:
        if agent_id not in self.agent_registry:
            return False
        agent = self.agent_registry.pop(agent_id)
        self.initial_agent_positions.pop(agent_id, None)
        try:
            self.agents.remove(agent)
        except ValueError:
            pass
        self.log(f"Agent {agent.name} removed", level="info")
        return True

    # Event management
    def schedule_event(self, event: SimulationEvent) -> None:
        self.events.append(event)
        self.log(f"Event scheduled: {event.event_type} at {event.position} (start={event.start_time})", level="info")

    def register_event_handler(self, event_type: str,
                               on_activate: Optional[Callable[[ "Simulation", SimulationEvent ], None]],
                               on_deactivate: Optional[Callable[[ "Simulation", SimulationEvent ], None]] = None) -> None:
        self.event_handlers[event_type] = {"activate": on_activate, "deactivate": on_deactivate}

    # Callbacks
    def register_step_callback(self, callback: Callable[[ "Simulation", int ], None]) -> None:
        self.step_callbacks.append(callback)

    def register_agent_callback(self, callback: Callable[[Agent, bool], None]) -> None:
        self.agent_callbacks.append(callback)

    # Control: start/pause/resume/stop
    def start(self, max_steps: Optional[int] = None) -> None:
        self.max_steps = max_steps
        self.is_running = True
        self.log("Simulation started", level="info")

    def pause(self) -> None:
        self.is_running = False
        self.log("Simulation paused", level="info")

    def resume(self) -> None:
        self.is_running = True
        self.log("Simulation resumed", level="info")

    def stop(self) -> None:
        self.is_running = False
        self.log("Simulation stopped", level="info")

    # Main loop
    def step(self) -> bool:
        if not self.is_running:
            return False

        self.current_step += 1
        self._process_events()

        for agent in sorted(self.agents, key=lambda a: a.id):
            if agent.state == AgentState.OFFLINE:
                continue

            try:
                agent.sense(self.city)
            except Exception as e:
                self.log(f"Agent {agent.name} sensing error: {e}", level="error")
                if self.pause_on_error:
                    self.pause()

            moved = False
            if agent.state == AgentState.MOVING and agent.path:
                try:
                    agent.move_step()
                    moved = True
                except Exception as e:
                    self.log(f"Agent {agent.name} move error: {e}", level="error")
                    if self.pause_on_error:
                        self.pause()

            elif agent.state == AgentState.IDLE:
                # Idle agents could be picked up by planners outside of simulation loop
                pass

            for cb in self.agent_callbacks:
                try:
                    cb(agent, moved)
                except Exception:
                    pass

        for cb in self.step_callbacks:
            try:
                cb(self, self.current_step)
            except Exception:
                pass

        self._update_statistics()

        if self.max_steps is not None and self.current_step >= self.max_steps:
            self.log("Max steps reached", level="info")
            self.stop()
            return False

        if self.auto_terminate:
            active_agents = [a for a in self.agents if a.state != AgentState.OFFLINE]
            has_work = any(
                a.state == AgentState.MOVING or (a.goal and not a.is_at_goal()) or (a.path and a.path_index < len(a.path))
                for a in active_agents
            )
            if not has_work:
                self.log("No agents have remaining work", level="info")
                self.stop()
                return False

        return True

    # Event processing
    def _process_events(self) -> None:
        for event in self.events:
            if event.start_time == self.current_step and not event.active:
                event.active = True
                self.statistics["events_triggered"] += 1
                self.log(f"Event activated: {event.event_type} at {event.position}", level="event")
                entry = self.event_handlers.get(event.event_type)
                handler = entry.get("activate") if entry else None
                if handler:
                    try:
                        handler(self, event)
                    except Exception as e:
                        self.log(f"Event activate handler error: {e}", level="error")

            if event.active and event.duration != -1 and self.current_step >= (event.start_time + event.duration):
                event.active = False
                self.log(f"Event ended: {event.event_type} at {event.position}", level="event")
                entry = self.event_handlers.get(event.event_type)
                handler = entry.get("deactivate") if entry else None
                if handler:
                    try:
                        handler(self, event)
                    except Exception as e:
                        self.log(f"Event deactivate handler error: {e}", level="error")

    # Stats / logging
    def _update_statistics(self) -> None:
        self.statistics["steps_completed"] = self.current_step
        self.statistics["total_distance_traveled"] = sum(a.distance_traveled for a in self.agents)
        self.statistics["total_goals_reached"] = sum(a.goals_completed for a in self.agents)
        active = [a for a in self.agents if a.state != AgentState.OFFLINE]
        self.statistics["active_agents"] = len(active)
        self.statistics["offline_agents"] = len([a for a in self.agents if a.state == AgentState.OFFLINE])
        self.statistics["average_energy"] = (sum(a.energy for a in self.agents) / len(self.agents)) if self.agents else 0.0
        self.statistics["average_distance"] = (sum(a.distance_traveled for a in self.agents) / len(self.agents)) if self.agents else 0.0
        # collision detection is domain-specific; leave placeholder increments for now

    def log(self, message: str, level: str = "info") -> None:
        entry = {"step": self.current_step, "level": level, "message": message, "time": time.time()}
        self.logs.append(entry)

    def get_logs(self, level: Optional[str] = None, last_n: Optional[int] = None) -> List[Dict[str, Any]]:
        items = self.logs
        if level:
            items = [l for l in items if l["level"] == level]
        if last_n:
            items = items[-last_n:]
        return items

    def get_statistics(self) -> Dict[str, Any]:
        return dict(self.statistics)

    def get_agent(self, agent_id: str) -> Optional[Agent]:
        return self.agent_registry.get(agent_id)

    def get_agents_at(self, position: Tuple[int, int]) -> List[Agent]:
        return [a for a in self.agents if a.position == position]

    def reset(self) -> None:
        """Reset simulation to initial state (restores initial agent positions)."""
        self.current_step = 0
        self.is_running = False
        self.events.clear()
        self.logs.clear()
        for agent in self.agents:
            initial = self.initial_agent_positions.get(agent.id)
            agent.reset(position=initial)
        self.statistics = {
            "steps_completed": 0,
            "total_distance_traveled": 0.0,
            "total_goals_reached": 0,
            "active_agents": len(self.agents),
            "events_triggered": 0,
            "offline_agents": 0,
            "average_energy": 0.0,
            "average_distance": 0.0,
            "collision_events": 0
        }
        self.log("Simulation reset to initial state", level="info")

    def __repr__(self) -> str:
        return f"<Simulation step={self.current_step} agents={len(self.agents)} running={self.is_running}>"
