# demo_simulation.py
from core import Agent, Simulation, SimulationEvent
from city import City
from utils import generate_walkable_position
import random

# ----------------------
# 1. Load city
# ----------------------
city = City("data/map.npy")  # make sure you have a valid map.npy
print(f"City loaded: {city.width}x{city.height}")

# ----------------------
# 2. Create simulation
# ----------------------
sim = Simulation(city)

# ----------------------
# 3. Spawn multiple agents
# ----------------------
num_agents = 5
agents = []

for i in range(num_agents):
    start_pos = generate_walkable_position(city)
    agent = Agent(f"Bot-{i+1}", start_pos=start_pos)
    
    # Set a random goal somewhere else in the city
    goal_pos = generate_walkable_position(city, exclude={start_pos})
    if goal_pos:
        agent.set_goal(goal_pos)
    
    sim.add_agent(agent)
    agents.append(agent)
    print(f"Spawned {agent.name} at {start_pos} with goal {goal_pos}")

# ----------------------
# 4. Add a sample event
# ----------------------
def traffic_event_handler(sim: Simulation, event: SimulationEvent):
    if event.active:
        print(f"Traffic started at {event.position}")
        city.block_tile(*event.position)
    else:
        print(f"Traffic cleared at {event.position}")
        city.unblock_tile(*event.position)

sim.register_event_handler("traffic", traffic_event_handler)

# Schedule traffic at random location
traffic_pos = generate_walkable_position(city)
if traffic_pos:
    traffic_event = SimulationEvent(
        event_type="traffic",
        position=traffic_pos,
        start_time=5,
        duration=5
    )
    sim.schedule_event(traffic_event)
    print(f"Scheduled traffic event at {traffic_pos} starting step 5")

# ----------------------
# 5. Run simulation
# ----------------------
max_steps = 20
for _ in range(max_steps):
    sim.step()
    
    # Print agent status
    for agent in agents:
        status = agent.get_status()
        print(f"Step {sim.current_step} | {agent.name}: Pos={status['position']}, "
              f"Goal={status['goal']}, Energy={status['energy_percent']}%, State={status['state']}")
    
    print("---")

# ----------------------
# 6. Print final statistics
# ----------------------
stats = sim.get_statistics()
print("\nFinal Simulation Statistics:")
for key, value in stats.items():
    print(f"{key}: {value}")
