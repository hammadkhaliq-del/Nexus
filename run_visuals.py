"""
NEXUS VISUAL LAUNCHER
Sets up a complex scenario and launches the Graphical Interface.
"""
import sys
import os
import random
import numpy as np
from core.city import City
from core.simulation import Simulation, SimulationEvent
from core.agent import Agent
from ui.visualizer import Visualizer

def create_complex_map():
    # 25x20 Grid
    width, height = 25, 20
    grid = np.zeros((height, width), dtype=np.int32)
    
    # 1. Add "Buildings" (Blocks of 1s)
    grid[3:6, 3:6] = 1
    grid[3:6, 10:15] = 1
    grid[10:15, 3:6] = 1
    grid[10:15, 10:15] = 1
    grid[5:15, 18:22] = 1 # Tall building on right
    
    # 2. Add "Parks" (Grass - 2s)
    grid[0:2, :] = 2
    grid[-2:, :] = 2
    grid[7:9, 7:9] = 2 # Central Park
    
    np.save("visual_city.npy", grid)
    return "visual_city.npy"

def main():
    print("🚀 INITIALIZING NEXUS GRAPHICAL MODE...")
    
    # 1. Load Environment
    map_file = create_complex_map()
    city = City(map_file)
    sim = Simulation(city)
    
    # 2. Deploy Agents (Complex Paths)
    # Agent 1: The Patroller (Blue) - Loops the center
    a1 = Agent("Police", (2, 2), speed=0.5)
    a1.set_path([(2,2), (2,16), (16,16), (16,2), (2,2)])
    sim.add_agent(a1)
    
    # Agent 2: The Courier (Orange) - Zig Zags
    a2 = Agent("FedEx", (18, 2), speed=0.8)
    a2.set_path([(18,2), (8,8), (18,14), (2,20)])
    sim.add_agent(a2)
    
    # Agent 3: The Jogger (Green) - Parks only
    a3 = Agent("Jogger", (1, 1), speed=0.3)
    a3.set_path([(1,1), (1,23), (18,23), (18,1)])
    sim.add_agent(a3)

    # 3. Schedule Dynamic Events
    # Event: Car Accident (Traffic Jam) at the central intersection
    # Lasts from Step 20 to Step 60
    evt = SimulationEvent("car_crash", (8, 8), start_time=20, duration=40)
    sim.schedule_event(evt)
    
    # Define what happens during the event
    def block_road(sim, e): sim.city.block_tile(*e.position)
    def open_road(sim, e): sim.city.unblock_tile(*e.position)
    sim.register_event_handler("car_crash", block_road, open_road)

    # 4. Launch Visualizer
    print("✅ SYSTEM READY. Opening Window...")
    viz = Visualizer(sim)
    viz.run()

if __name__ == "__main__":
    main()