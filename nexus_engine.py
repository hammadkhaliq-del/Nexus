"""
NEXUS ENGINE - Production Backend
Complete AI + Core Integration
Version: 3.0 FINAL
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Core imports
from core.city import City
from core.graph import CityGraph
from core.simulation import Simulation, SimulationEvent
from core.agent import Agent, AgentState
from core.utils import euclidean_distance, generate_walkable_position

# AI imports
from ai.search import a_star, dijkstra, bfs, SearchResult
from ai.logic_engine import LogicEngine, AgentRules
from ai.explainability import ExplainabilityEngine
from ai.bayesian import BayesianNetwork


class NexusEngine:
    """
    Complete NEXUS Engine - Brain + Body Integration
    """
    
    def __init__(self, grid_size: int = 80, num_agents: int = 15):
        self.grid_size = grid_size
        self.num_agents = num_agents
        
        # Core components (Body)
        self.city: Optional[City] = None
        self.graph: Optional[CityGraph] = None
        self.simulation: Optional[Simulation] = None
        
        # AI components (Brain)
        self.logic_engine = LogicEngine()
        self.explainer = ExplainabilityEngine()
        self.bayesian = BayesianNetwork()
        
        # Statistics
        self.stats = {
            "tick": 0,
            "total_paths_planned": 0,
            "successful_paths": 0,
            "failed_paths": 0,
            "replans": 0,
            "energy_alerts": 0,
            "goals_completed": 0
        }
        
        # Event log
        self.events: List[str] = []
        
        # Initialize
        self._initialize_system()
    
    def _initialize_system(self):
        """Initialize complete NEXUS system"""
        self._log("🚀 NEXUS ENGINE INITIALIZING...")
        
        # 1. Create city
        self._create_city()
        
        # 2. Build graph
        self._log("🕸️ Building navigation graph...")
        self.graph = CityGraph(self.city)
        
        # 3. Initialize simulation
        self._log("⚙️ Initializing simulation engine...")
        self.simulation = Simulation(self.city, self.graph)
        
        # 4. Setup AI brain
        self._setup_ai_brain()
        
        # 5. Deploy agents
        self._deploy_agents()
        
        self._log("✅ NEXUS ENGINE READY")
    
    def _create_city(self):
        """Create realistic city map"""
        self._log("🏙️ Generating city environment...")
        
        grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        
        # Add buildings (commercial districts)
        num_buildings = int(self.grid_size * 0.4)
        for _ in range(num_buildings):
            bx = np.random.randint(5, self.grid_size - 10)
            by = np.random.randint(5, self.grid_size - 10)
            bw = np.random.randint(3, 8)
            bh = np.random.randint(3, 8)
            grid[by:by+bh, bx:bx+bw] = 1  # Building
        
        # Add parks (green spaces)
        num_parks = int(self.grid_size * 0.15)
        for _ in range(num_parks):
            px = np.random.randint(5, self.grid_size - 8)
            py = np.random.randint(5, self.grid_size - 8)
            pw = np.random.randint(4, 7)
            ph = np.random.randint(4, 7)
            grid[py:py+ph, px:px+pw] = 2  # Grass/Park
        
        # Add water features
        num_water = 3
        for _ in range(num_water):
            wx = np.random.randint(10, self.grid_size - 15)
            wy = np.random.randint(10, self.grid_size - 15)
            wsize = np.random.randint(5, 10)
            grid[wy:wy+wsize, wx:wx+wsize] = 3  # Water
        
        # Save and load
        np.save("nexus_city.npy", grid)
        self.city = City("nexus_city.npy")
        
        stats = self.city.get_stats()
        self._log(f"   📊 City: {stats['dimensions']}")
        for tile, data in stats['tile_distribution'].items():
            self._log(f"      • {tile}: {data['count']} tiles ({data['percentage']}%)")
    
    def _setup_ai_brain(self):
        """Configure AI reasoning systems"""
        self._log("🧠 Configuring AI brain...")
        
        # Add intelligent rules
        low_energy_rule = AgentRules.low_energy_recharge(threshold=25.0)
        self.logic_engine.add_rule(low_energy_rule)
        
        goal_reached_rule = AgentRules.goal_reached_idle()
        self.logic_engine.add_rule(goal_reached_rule)
        
        self._log("   ✓ Logic rules installed")
        self._log("   ✓ Explainability engine ready")
        self._log("   ✓ Bayesian network initialized")
    
    def _deploy_agents(self):
        """Deploy intelligent agents with AI pathfinding"""
        self._log(f"🤖 Deploying {self.num_agents} agents...")
        
        # Get valid spawn points
        spawn_points = []
        max_attempts = 1000
        attempts = 0
        
        while len(spawn_points) < self.num_agents and attempts < max_attempts:
            pos = generate_walkable_position(self.city)
            if pos and pos not in spawn_points:
                spawn_points.append(pos)
            attempts += 1
        
        # Deploy agents
        for i, spawn in enumerate(spawn_points):
            # Create agent
            agent = Agent(
                name=f"NEXUS-{i:02d}",
                start_pos=spawn,
                speed=np.random.uniform(1.0, 2.0),
                max_energy=np.random.uniform(80, 120)
            )
            
            # Assign goal
            goal_idx = (i + len(spawn_points)//2) % len(spawn_points)
            goal = spawn_points[goal_idx]
            agent.set_goal(goal)
            
            # AI Pathfinding
            self._plan_path_for_agent(agent)
            
            # Add to simulation
            self.simulation.add_agent(agent)
        
        self._log(f"   ✓ {len(self.simulation.agents)} agents deployed")
        self._log(f"   ✓ {self.stats['successful_paths']}/{self.stats['total_paths_planned']} paths successful")
    
    def _plan_path_for_agent(self, agent: Agent):
        """Use AI to plan path for agent"""
        if not agent.goal:
            return
        
        start = (int(agent.position[0]), int(agent.position[1]))
        
        try:
            # Use A* algorithm
            result = a_star(self.graph, start, agent.goal)
            
            self.stats['total_paths_planned'] += 1
            
            if result.success:
                agent.set_path(result.path)
                self.stats['successful_paths'] += 1
                
                # Generate explanation
                explanation = self.explainer.explain_path(
                    agent, result.path, result.algorithm, result.cost
                )
                self._log(f"   [AI] {agent.name}: Path found ({len(result.path)} steps, cost: {result.cost:.1f})")
            else:
                self.stats['failed_paths'] += 1
                self._log(f"   [AI] {agent.name}: No valid path to goal", "warning")
        
        except Exception as e:
            self.stats['failed_paths'] += 1
            self._log(f"   [ERROR] {agent.name}: Pathfinding error - {str(e)}", "error")
    
    def _run_ai_decisions(self):
        """Execute AI reasoning for all agents"""
        for agent in self.simulation.agents:
            # Logic engine evaluation
            context = {
                "agent": agent,
                "city": self.city,
                "detected_obstacles": list(agent.known_obstacles)
            }
            
            actions = self.logic_engine.forward_chain(context)
            
            # Log AI decisions
            for action in actions:
                if "recharge" in action.lower():
                    self.stats['energy_alerts'] += 1
                    self._log(f"[AI-LOGIC] {agent.name}: Energy critical, initiating recharge", "ai")
                elif "goal" in action.lower() and "reached" in action.lower():
                    self.stats['goals_completed'] += 1
                    self._log(f"[AI-LOGIC] {agent.name}: Goal achieved", "success")
            
            # Check for replanning needs
            if agent.path and agent.path_index < len(agent.path):
                next_pos = agent.path[agent.path_index]
                
                if not self.city.is_walkable(*next_pos):
                    self._replan_agent(agent)
    
    def _replan_agent(self, agent: Agent):
        """Replan path for agent (AI replanning)"""
        if not agent.goal:
            return
        
        old_path_len = len(agent.path) if agent.path else 0
        
        try:
            start = (int(agent.position[0]), int(agent.position[1]))
            result = a_star(self.graph, start, agent.goal)
            
            if result.success:
                agent.set_path(result.path)
                self.stats['replans'] += 1
                
                explanation = self.explainer.explain_replanning(
                    agent, [], result.path, "obstacle detected on route"
                )
                self._log(f"[AI-REPLAN] {agent.name}: Route recalculated ({old_path_len} → {len(result.path)} steps)", "ai")
        except:
            pass
    
    def step(self):
        """Execute one simulation step"""
        # AI decision making (every 3 ticks)
        if self.stats['tick'] % 3 == 0:
            self._run_ai_decisions()
        
        # Physics simulation
        try:
            self.simulation.step()
        except Exception as e:
            self._log(f"[ERROR] Simulation step failed: {str(e)}", "error")
        
        self.stats['tick'] += 1
    
    def run(self, steps: int = 100, verbose: bool = True):
        """Run simulation for N steps"""
        if verbose:
            self._log(f"▶️ Running simulation for {steps} steps...")
        
        for i in range(steps):
            self.step()
            
            if verbose and (i + 1) % 20 == 0:
                self._log(f"   Tick {self.stats['tick']}: {self._get_active_agents()} agents active")
        
        if verbose:
            self._log("⏸️ Simulation paused")
            self.print_statistics()
    
    def reset(self):
        """Reset entire system"""
        self._log("🔄 Resetting NEXUS engine...")
        
        self.stats = {
            "tick": 0,
            "total_paths_planned": 0,
            "successful_paths": 0,
            "failed_paths": 0,
            "replans": 0,
            "energy_alerts": 0,
            "goals_completed": 0
        }
        
        self.events.clear()
        self.simulation = None
        
        self._initialize_system()
    
    def assign_random_goals(self):
        """AI auto-goal assignment"""
        if not self.simulation:
            return
        
        walkable = self.city.get_walkable_positions()
        if not walkable:
            return
        
        for agent in self.simulation.agents:
            new_goal = walkable[np.random.randint(0, len(walkable))]
            agent.set_goal(new_goal)
            self._plan_path_for_agent(agent)
        
        self._log(f"[AI-AUTO] Random goals assigned to {len(self.simulation.agents)} agents", "ai")
    
    def recharge_all_agents(self, amount: float = 50.0):
        """Recharge all agents"""
        if not self.simulation:
            return
        
        for agent in self.simulation.agents:
            agent.recharge_energy(amount)
        
        self._log(f"[SYSTEM] All agents recharged +{amount} energy", "success")
    
    def get_agent_by_name(self, name: str) -> Optional[Agent]:
        """Get agent by name"""
        if not self.simulation:
            return None
        
        for agent in self.simulation.agents:
            if agent.name == name:
                return agent
        return None
    
    def get_statistics(self) -> Dict:
        """Get comprehensive statistics"""
        if not self.simulation:
            return self.stats
        
        agents = self.simulation.agents
        
        state_counts = {
            "moving": sum(1 for a in agents if a.state == AgentState.MOVING),
            "idle": sum(1 for a in agents if a.state == AgentState.IDLE),
            "charging": sum(1 for a in agents if a.state == AgentState.CHARGING),
            "offline": sum(1 for a in agents if a.state == AgentState.OFFLINE)
        }
        
        avg_energy = sum(a.energy for a in agents) / len(agents) if agents else 0
        total_distance = sum(a.distance_traveled for a in agents)
        
        return {
            **self.stats,
            "agents": {
                "total": len(agents),
                "states": state_counts,
                "avg_energy": avg_energy,
                "total_distance": total_distance
            },
            "graph": self.graph.get_stats() if self.graph else {},
            "city": self.city.get_stats() if self.city else {}
        }
    
    def print_statistics(self):
        """Print formatted statistics"""
        stats = self.get_statistics()
        
        print("\n" + "="*60)
        print("📊 NEXUS ENGINE STATISTICS")
        print("="*60)
        print(f"⏱️  Simulation Tick: {stats['tick']}")
        print(f"🤖 Total Agents: {stats['agents']['total']}")
        print(f"   • Moving: {stats['agents']['states']['moving']}")
        print(f"   • Idle: {stats['agents']['states']['idle']}")
        print(f"   • Charging: {stats['agents']['states']['charging']}")
        print(f"   • Offline: {stats['agents']['states']['offline']}")
        print(f"⚡ Average Energy: {stats['agents']['avg_energy']:.1f}%")
        print(f"📏 Total Distance: {stats['agents']['total_distance']:.1f}")
        print(f"🧠 AI Pathfinding:")
        print(f"   • Total Plans: {stats['total_paths_planned']}")
        print(f"   • Successful: {stats['successful_paths']}")
        print(f"   • Success Rate: {stats['successful_paths']/stats['total_paths_planned']*100:.1f}%")
        print(f"   • Replans: {stats['replans']}")
        print(f"🎯 Goals Completed: {stats['goals_completed']}")
        print(f"⚠️  Energy Alerts: {stats['energy_alerts']}")
        print("="*60 + "\n")
    
    def _get_active_agents(self) -> int:
        """Count active agents"""
        if not self.simulation:
            return 0
        return sum(1 for a in self.simulation.agents if a.state != AgentState.OFFLINE)
    
    def _log(self, message: str, level: str = "info"):
        """Add event to log"""
        from datetime import datetime
        timestamp = datetime.now().strftime('[%H:%M:%S]')
        
        level_icons = {
            "info": "ℹ️",
            "success": "✅",
            "warning": "⚠️",
            "error": "❌",
            "ai": "🧠"
        }
        
        icon = level_icons.get(level, "•")
        formatted = f"{timestamp} {icon} {message}"
        
        self.events.insert(0, formatted)
        self.events = self.events[:200]  # Keep last 200 events
        
        print(formatted)


# -------------------------
# STANDALONE DEMO
# -------------------------
if __name__ == "__main__":
    print("\n🚀 NEXUS ENGINE - Standalone Demo\n")
    
    # Create engine
    engine = NexusEngine(grid_size=60, num_agents=10)
    
    # Run simulation
    engine.run(steps=50, verbose=True)
    
    # Demonstrate features
    print("\n🎯 Testing AI Features...")
    engine.assign_random_goals()
    engine.run(steps=30, verbose=False)
    
    print("\n⚡ Testing Energy System...")
    engine.recharge_all_agents(30)
    
    # Final stats
    engine.print_statistics()
    
    print("\n✅ Demo Complete!")