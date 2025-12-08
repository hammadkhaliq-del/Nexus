"""
NEXUS PRIME: THE INTEGRATED SIMULATION (FULL ARCHITECTURE)
Combines Core, AI (Search, Logic, Planner), and UI into a fully autonomous system.
"""
import sys
import random
import numpy as np
from pathlib import Path

# --- FIX PATHS ---
sys.path.insert(0, str(Path(__file__).resolve().parent))

# --- CORE IMPORTS ---
from core.city import City
from core.graph import CityGraph
from core.simulation import Simulation, SimulationEvent
from core.agent import Agent, AgentState
from ui.visualizer import Visualizer

# --- AI IMPORTS ---
from ai.search import SearchEngine
from ai.logic_engine import LogicEngine, AgentRules
from ai.planner import Planner, Action, State

# ==============================================================================
# 1. ENVIRONMENT SETUP
# ==============================================================================
def create_prime_map():
    """Generates a complex city layout."""
    width, height = 30, 20
    grid = np.zeros((height, width), dtype=np.int32)
    
    # City Blocks (Walls = 1)
    grid[2:6, 2:8] = 1
    grid[2:6, 10:18] = 1
    grid[2:6, 20:28] = 1
    grid[8:12, 2:8] = 1
    grid[8:12, 10:18] = 1
    grid[8:12, 20:28] = 1
    grid[14:18, 2:8] = 1
    grid[14:18, 10:18] = 1
    grid[14:18, 20:28] = 1
    
    # Special Zones
    grid[8:12, 10:18] = 2  # Park (Walkable cost 1)
    
    np.save("nexus_prime_map.npy", grid)
    return "nexus_prime_map.npy"

# ==============================================================================
# 2. INTELLIGENT AGENT (THE BRAIN)
# ==============================================================================
class IntelligentAgent(Agent):
    def __init__(self, name, start_pos, graph):
        super().__init__(name, start_pos)
        self.graph = graph
        
        # --- AI SUBSYSTEMS ---
        self.logic_engine = LogicEngine()
        self.planner = Planner()
        
        # Agent Memory
        self.high_level_plan = []  # List of Planner Actions
        self.current_action = None # What I am doing right now
        self.path_calculated = False # State flag
        
        # Initialize Logic Rules (Reflexes)
        self.logic_engine.add_rule(AgentRules.low_energy_recharge(30))
        self.logic_engine.add_rule(AgentRules.obstacle_detected_replan())

    def assign_mission(self, pickup_loc, dropoff_loc):
        """
        Uses the PLANNER to generate a sequence of actions.
        """
        print(f"\n[{self.name}] 🤖 COMPUTING MISSION PLAN...")
        
        # 1. Define Actions
        act_goto_pickup = Action("goto_pickup", {"at_start"}, {"at_pickup"}, {"at_start"})
        act_pickup = Action("pickup_package", {"at_pickup", "hands_empty"}, {"has_package"}, {"hands_empty"})
        act_goto_drop = Action("goto_dropoff", {"at_pickup", "has_package"}, {"at_dropoff"}, {"at_pickup"})
        act_deliver = Action("deliver_package", {"at_dropoff", "has_package"}, {"mission_complete"}, {"has_package"})

        self.planner.add_action(act_goto_pickup)
        self.planner.add_action(act_pickup)
        self.planner.add_action(act_goto_drop)
        self.planner.add_action(act_deliver)

        # 2. Generate Plan
        initial = State({"at_start", "hands_empty"})
        goal = {"mission_complete"}
        plan = self.planner.plan(initial, goal)
        
        if plan:
            self.high_level_plan = []
            for step in plan:
                meta = {"name": step.name}
                if step.name == "goto_pickup": meta["target"] = pickup_loc
                if step.name == "goto_dropoff": meta["target"] = dropoff_loc
                self.high_level_plan.append(meta)
            print(f"[{self.name}] 📜 PLAN GENERATED: {[s['name'] for s in self.high_level_plan]}")
        else:
            print(f"[{self.name}] ❌ PLAN FAILED. Impossible goal.")

    def execute_ai(self, city):
        """
        The Main AI Loop. Called every tick.
        """
        # 1. Run Reflexes
        self.logic_engine.update_memory("agent", self)
        alerts = self.logic_engine.forward_chain()
        if alerts and self.state == AgentState.CHARGING:
            return

        # 2. Pick Next Action from Plan
        if not self.current_action and self.high_level_plan:
            self.current_action = self.high_level_plan.pop(0)
            self.path_calculated = False # Reset path flag for new action
            print(f"[{self.name}] ▶️ STARTING ACTION: {self.current_action['name']}")

        # 3. Execute Current Action
        if self.current_action:
            action_name = self.current_action["name"]

            # --- CASE A: MOVEMENT ---
            if "goto" in action_name:
                target = self.current_action["target"]
                
                # Only calculate path ONCE per action
                if not self.path_calculated:
                    start_node = (int(self.position[0]), int(self.position[1]))
                    path = SearchEngine.a_star(self.graph, start_node, target)
                    if path:
                        self.set_path(path)
                        self.path_calculated = True
                    else:
                        print(f"[{self.name}] ❌ PATHFINDING ERROR to {target}. Retrying...")
                        return # Try again next tick

                # Check arrival using Math (Integer comparison)
                curr_pos = (int(self.position[0]), int(self.position[1]))
                if curr_pos == target:
                    print(f"[{self.name}] ✓ Arrived at waypoint.")
                    self.current_action = None # Done
            
            # --- CASE B: WORK (Pickup/Deliver) ---
            elif "pickup" in action_name or "deliver" in action_name:
                self.wait() # Simulate 1 tick of work
                print(f"[{self.name}] 🔨 Working...")
                self.current_action = None # Done immediately after waiting

# ==============================================================================
# 3. MAIN SIMULATION LAUNCHER
# ==============================================================================
def main():
    print("🚀 INITIALIZING NEXUS PRIME (FULL ARCHITECTURE)...")
    
    # 1. Setup World
    map_file = create_prime_map()
    city = City(map_file)
    graph = CityGraph(city)
    sim = Simulation(city, graph)
    
    # 2. Deploy Intelligent Agents
    # FIX: Ensure coordinates are valid walkable roads (0), NOT buildings (1)
    
    # Courier: Start(0,0) -> Pickup(1,1) -> Dropoff(19,29)
    # (1,1) is a road corner near the first building block
    courier = IntelligentAgent("Courier-AI", (0, 0), graph)
    sim.add_agent(courier)
    courier.assign_mission(pickup_loc=(1, 1), dropoff_loc=(19, 29))
    
    # SecBot: Start(9,9) -> Patrol Point A(9,19) -> Patrol Point B(9,1)
    # Using (9,19) and (9,1) keeps it on the horizontal road between buildings
    sec_bot = IntelligentAgent("SecBot", (9, 9), graph)
    sim.add_agent(sec_bot)
    sec_bot.assign_mission(pickup_loc=(9, 19), dropoff_loc=(9, 1))

    # 3. Events
    sim.schedule_event(SimulationEvent("traffic_jam", (5, 9), start_time=40, duration=50))
    
    def on_jam(sim, event):
        sim.city.block_tile(*event.position)
        print(f"\n⚠️  EVENT: TRAFFIC JAM at {event.position}")
    
    def off_jam(sim, event):
        sim.city.unblock_tile(*event.position)
        print(f"\n✅ EVENT: TRAFFIC CLEARED")

    sim.register_event_handler("traffic_jam", on_jam, off_jam)

    # 4. Hook AI Update
    def update_agents(sim, step):
        for agent in sim.agents:
            if isinstance(agent, IntelligentAgent):
                agent.execute_ai(sim.city)

    sim.register_step_callback(update_agents)

    # 5. Launch UI
    print("✅ SYSTEM READY. Launching Pro Dashboard...")
    viz = Visualizer(sim)
    viz.run()

if __name__ == "__main__":
    main()