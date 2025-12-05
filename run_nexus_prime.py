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

# --- AI IMPORTS (The Full Stack) ---
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
    
    # City Blocks
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
    grid[8:12, 10:18] = 2  # Park
    grid[0, 0] = 0         # Depot A (Top Left)
    grid[19, 29] = 0       # Dropoff B (Bottom Right)
    
    np.save("nexus_prime_map.npy", grid)
    return "nexus_prime_map.npy"

# ==============================================================================
# 2. INTELLIGENT AGENT (THE BRAIN)
# ==============================================================================
class IntelligentAgent(Agent):
    """
    An Agent that uses:
    1. PLANNER: To break tasks into steps (Goto -> Pickup -> Deliver)
    2. SEARCH: To navigate the grid (A*)
    3. LOGIC: To monitor health and rules (Energy check)
    """
    def __init__(self, name, start_pos, graph):
        super().__init__(name, start_pos)
        self.graph = graph
        
        # --- AI SUBSYSTEMS ---
        self.logic_engine = LogicEngine()
        self.planner = Planner()
        
        # Agent Memory
        self.high_level_plan = []  # List of Planner Actions
        self.current_action = None # What I am doing right now
        self.delivery_state = "IDLE" 
        
        # Initialize Logic Rules (Reflexes)
        self.logic_engine.add_rule(AgentRules.low_energy_recharge(30))
        self.logic_engine.add_rule(AgentRules.obstacle_detected_replan())

    def assign_mission(self, pickup_loc, dropoff_loc):
        """
        Uses the PLANNER to generate a sequence of actions.
        Task: Get package from A and take it to B.
        """
        print(f"\n[{self.name}] 🤖 COMPUTING MISSION PLAN...")
        
        # 1. Define the Planning Domain (STRIPS)
        # Action: Go to Pickup
        act_goto_pickup = Action(
            name="goto_pickup",
            preconditions={"at_start"},
            add_effects={"at_pickup"},
            delete_effects={"at_start"}
        )
        # Action: Pickup Package
        act_pickup = Action(
            name="pickup_package",
            preconditions={"at_pickup", "hands_empty"},
            add_effects={"has_package"},
            delete_effects={"hands_empty"}
        )
        # Action: Go to Dropoff
        act_goto_drop = Action(
            name="goto_dropoff",
            preconditions={"at_pickup", "has_package"},
            add_effects={"at_dropoff"},
            delete_effects={"at_pickup"}
        )
        # Action: Deliver
        act_deliver = Action(
            name="deliver_package",
            preconditions={"at_dropoff", "has_package"},
            add_effects={"mission_complete"},
            delete_effects={"has_package"}
        )

        self.planner.add_action(act_goto_pickup)
        self.planner.add_action(act_pickup)
        self.planner.add_action(act_goto_drop)
        self.planner.add_action(act_deliver)

        # 2. Define Initial State and Goal
        initial = State({"at_start", "hands_empty"})
        goal = {"mission_complete"}

        # 3. Generate Plan
        plan = self.planner.plan(initial, goal)
        
        if plan:
            # Store metadata for execution
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
        # 1. Run Reflexes (Logic Engine)
        self.logic_engine.update_memory("agent", self)
        alerts = self.logic_engine.forward_chain()
        if alerts:
            print(f"[{self.name}] ⚡ REFLEX TRIGGERED: {alerts}")
            # If logic says CHARGING, we stop moving
            if self.state == AgentState.CHARGING:
                return

        # 2. Execute High-Level Plan
        if not self.path and not self.current_action and self.high_level_plan:
            # Pop next action
            self.current_action = self.high_level_plan.pop(0)
            action_name = self.current_action["name"]
            print(f"[{self.name}] ▶️ EXECUTING ACTION: {action_name}")

            if "goto" in action_name:
                # Use SEARCH ENGINE (A*) for movement
                target = self.current_action["target"]
                start_node = (int(self.position[0]), int(self.position[1]))
                path = SearchEngine.a_star(self.graph, start_node, target)
                if path:
                    self.set_path(path)
                else:
                    print(f"[{self.name}] ❌ PATHFINDING ERROR.")
            
            elif "pickup" in action_name or "deliver" in action_name:
                # Simulate work time
                self.wait() # Pause for 1 tick
                print(f"[{self.name}] 🔨 Working...")

        # 3. Clear Action when done
        if self.current_action:
            if "goto" in self.current_action["name"] and not self.path:
                print(f"[{self.name}] ✓ Arrived at waypoint.")
                self.current_action = None
            elif "pickup" in self.current_action["name"]:
                self.current_action = None # Instant action for now

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
    
    # 2. Deploy Intelligent Agent
    # Agent: Courier-01
    # Mission: Go from Top-Left (0,0) -> Pick up at (2,2) -> Deliver to Bottom-Right (19,29)
    courier = IntelligentAgent("Courier-AI", (0, 0), graph)
    sim.add_agent(courier)
    
    # Assign the complex mission
    courier.assign_mission(pickup_loc=(2, 2), dropoff_loc=(19, 29))
    
    # Agent 2: Security Bot (Patrol Pattern)
    sec_bot = IntelligentAgent("SecBot", (9, 9), graph)
    sim.add_agent(sec_bot)
    sec_bot.assign_mission(pickup_loc=(9, 20), dropoff_loc=(9, 2)) # Patrol route

    # 3. Schedule Events
    sim.schedule_event(SimulationEvent("traffic_jam", (5, 9), start_time=40, duration=50))
    
    def on_jam(sim, event):
        sim.city.block_tile(*event.position)
        print(f"\n⚠️  EVENT: TRAFFIC JAM at {event.position}")
    
    def off_jam(sim, event):
        sim.city.unblock_tile(*event.position)
        print(f"\n✅ EVENT: TRAFFIC CLEARED")

    sim.register_event_handler("traffic_jam", on_jam, off_jam)

    # 4. Hook AI Update Loop
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