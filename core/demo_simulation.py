"""
NEXUS Core Test Simulation
"""
import numpy as np
import sys
from pathlib import Path

# --- FIX PATHS (Crucial step) ---
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from core.city import City
from core.graph import CityGraph
from core.agent import Agent, AgentState
from core.simulation import Simulation, SimulationEvent

def run_test():
    print("\n" + "="*50)
    print("🚀 NEXUS SYSTEM LAUNCH")
    print("="*50)

    # 1. Create Map
    print("1. Building City Environment...")
    grid = np.zeros((20, 20), dtype=np.int32)
    grid[5:8, 5:8] = 1 # Block some tiles
    np.save("test_city_map.npy", grid)
    
    city = City("test_city_map.npy")
    sim = Simulation(city)
    print(f"   ✓ City loaded ({city.width}x{city.height})")

    # 2. Add Agents
    print("2. Deploying Agents...")
    
    # Bot A
    a1 = Agent("Patrol-Bot", start_pos=(2, 2))
    a1.set_path([(2,2), (2,10), (10,10)])
    sim.add_agent(a1)
    
    # Bot B
    a2 = Agent("Runner-Bot", start_pos=(15, 15), speed=1.5)
    a2.set_path([(15,15), (5,5)])
    sim.add_agent(a2)
    
    print(f"   ✓ Agents active: {[a.name for a in sim.agents]}")

    # 3. Schedule Event
    print("3. Scheduling Traffic Event...")
    def on_traffic(sim, event):
        print(f"   [EVENT] 🚗 Traffic Jam started at {event.position}!")
    
    sim.register_event_handler("traffic", on_traffic)
    sim.schedule_event(SimulationEvent("traffic", (5,5), start_time=3))
    print("   ✓ Event queued for Step 3")

    # 4. Run Loop
    print("\n▶️  RUNNING SIMULATION (Max 10 steps)...")
    print("-" * 50)
    
    # Define what to print every step
    def on_step(sim, step):
        # Build a string showing where every agent is
        positions = [f"{a.name}: ({a.position[0]:.1f}, {a.position[1]:.1f})" for a in sim.agents]
        print(f"   Step {step:02d}: " + " | ".join(positions))

    sim.register_step_callback(on_step)
    
    # START!
    sim.run(max_steps=10, delay=0.05)
    
    print("-" * 50)
    print("✅ SIMULATION COMPLETE")
    print("Final Stats:", sim.get_statistics())

if __name__ == "__main__":
    run_test()