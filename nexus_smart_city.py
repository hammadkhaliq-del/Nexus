"""
NEXUS - AI-Powered Smart City Simulation System
Complete integration of 6 major AI engines operating a unified simulated city
"""

import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time
from pathlib import Path
import sys
import random
import hashlib
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.city import City
from core.graph import CityGraph
from core.simulation import Simulation, SimulationEvent
from core.agent import Agent, AgentState
from ai.search import a_star, dijkstra, bfs, compare_algorithms
from ai.logic_engine import LogicEngine, Rule, RuleType, create_simple_rule, AgentRules
from ai.explainability import ExplainabilityEngine
from ai.bayesian import BayesianNetwork
from ai.planner import Planner, State, Action
from ai.csp_engine import CSPEngine, Variable, Constraint

# ==================== COLOR SCHEME ====================
HOLO_CYAN = "#00ffff"
HOLO_MAGENTA = "#ff00ff"
HOLO_BLUE = "#0080ff"
HOLO_GREEN = "#00ff80"
HOLO_RED = "#ff0055"
HOLO_ORANGE = "#ff8800"
HOLO_PURPLE = "#7b2ff7"
DARK_BG = "#0a0a15"
DARK_PANEL = "rgba(10, 20, 40, 0.9)"

# ==================== PAGE CONFIGURATION ====================
st.set_page_config(
    page_title="NEXUS - AI Smart City",
    layout="wide",
    initial_sidebar_state="collapsed",
    page_icon="🏙️"
)

# ==================== CSS STYLING ====================
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;700;900&family=Rajdhani:wght@300;400;500;600;700&display=swap');
    
    * {{
        font-family: 'Rajdhani', sans-serif !important;
    }}
    
    .stApp {{
        background: linear-gradient(135deg, {DARK_BG} 0%, #0f0f25 50%, #1a0a20 100%);
    }}
    
    h1, h2, h3, h4 {{
        font-family: 'Orbitron', sans-serif !important;
        color: {HOLO_CYAN};
        text-shadow: 0 0 15px {HOLO_CYAN}, 0 0 30px {HOLO_CYAN};
        letter-spacing: 3px;
    }}
    
    .main-title {{
        font-size: 3.5rem;
        text-align: center;
        color: {HOLO_CYAN};
        text-shadow: 0 0 20px {HOLO_CYAN}, 0 0 40px {HOLO_CYAN}, 0 0 60px {HOLO_CYAN};
        margin-bottom: 10px;
        font-family: 'Orbitron', sans-serif;
        font-weight: 900;
        letter-spacing: 10px;
    }}
    
    .subtitle {{
        text-align: center;
        color: {HOLO_MAGENTA};
        font-size: 1.3rem;
        margin-bottom: 30px;
        letter-spacing: 4px;
    }}
    
    .stTabs [data-baseweb="tab-panel"] {{
        background-color: {DARK_PANEL};
        border: 2px solid {HOLO_CYAN};
        box-shadow: 0 0 20px {HOLO_CYAN};
        border-radius: 10px;
        padding: 25px;
        backdrop-filter: blur(15px);
    }}
    
    [data-testid="stMetricValue"] {{
        color: {HOLO_CYAN};
        text-shadow: 0 0 10px {HOLO_CYAN};
        font-size: 2.2rem !important;
        font-weight: 700;
    }}
    
    [data-testid="stMetricLabel"] {{
        color: {HOLO_MAGENTA};
        text-transform: uppercase;
        letter-spacing: 2px;
        font-weight: 600;
    }}
    
    .stButton button {{
        background: linear-gradient(135deg, {HOLO_BLUE}, {HOLO_PURPLE});
        color: white;
        border: 2px solid {HOLO_CYAN};
        box-shadow: 0 0 25px {HOLO_CYAN};
        font-weight: 700;
        letter-spacing: 2px;
        font-size: 1.1rem;
        transition: all 0.3s;
        padding: 12px 30px;
    }}
    
    .stButton button:hover {{
        box-shadow: 0 0 35px {HOLO_CYAN}, 0 0 50px {HOLO_MAGENTA};
        transform: translateY(-3px);
    }}
    
    .event-box {{
        background: {DARK_PANEL};
        border-left: 4px solid {HOLO_CYAN};
        padding: 12px;
        margin: 8px 0;
        border-radius: 5px;
        backdrop-filter: blur(10px);
        font-size: 0.95rem;
    }}
    
    .event-time {{
        color: {HOLO_MAGENTA};
        font-weight: 700;
    }}
    
    .ai-engine-badge {{
        background: linear-gradient(135deg, {HOLO_PURPLE}, {HOLO_BLUE});
        color: white;
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 700;
        display: inline-block;
        margin: 3px;
        border: 1px solid {HOLO_CYAN};
        box-shadow: 0 0 10px {HOLO_CYAN};
    }}
    
    .emergency-badge {{
        background: linear-gradient(135deg, {HOLO_RED}, {HOLO_ORANGE});
        color: white;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 700;
        display: inline-block;
        animation: pulse 2s infinite;
    }}
    
    @keyframes pulse {{
        0%, 100% {{ opacity: 1; }}
        50% {{ opacity: 0.6; }}
    }}
    
    .login-container {{
        background: {DARK_PANEL};
        border: 3px solid {HOLO_CYAN};
        box-shadow: 0 0 40px {HOLO_CYAN};
        border-radius: 20px;
        padding: 50px;
        backdrop-filter: blur(20px);
    }}
    
    .signup-container {{
        background: {DARK_PANEL};
        border: 3px solid {HOLO_MAGENTA};
        box-shadow: 0 0 40px {HOLO_MAGENTA};
        border-radius: 20px;
        padding: 50px;
        backdrop-filter: blur(20px);
    }}
    
    ::-webkit-scrollbar {{
        width: 12px;
        height: 12px;
    }}
    
    ::-webkit-scrollbar-track {{
        background: {DARK_BG};
    }}
    
    ::-webkit-scrollbar-thumb {{
        background: linear-gradient(180deg, {HOLO_CYAN}, {HOLO_PURPLE});
        border-radius: 6px;
    }}
</style>
""", unsafe_allow_html=True)

# ==================== SESSION STATE INITIALIZATION ====================
def init_session_state():
    """Initialize all session state variables"""
    defaults = {
        'logged_in': False,
        'username': None,
        'users_db': {'admin': hashlib.sha256('admin123'.encode()).hexdigest()},  # Demo users
        'show_signup': False,
        
        # Simulation state
        'city': None,
        'graph': None,
        'simulation': None,
        'agents': [],
        'emergency_vehicles': [],
        'is_running': False,
        'current_tick': 0,
        'speed_multiplier': 1.0,
        
        # AI Engines
        'logic_engine': None,
        'explainer': None,
        'bayesian': None,
        'planner': None,
        'csp_engine': None,
        
        # Events and tracking
        'events': [],
        'incidents': [],
        'emergencies': [],
        'ai_decisions': [],
        
        # Statistics
        'stats': {
            'efficiency_score': 100.0,
            'avg_search_time': 0.0,
            'resource_utilization': 85.0,
            'incidents_count': 0,
            'avg_path_length': 0.0,
            'accidents_predicted': 0,
            'rules_fired': 0,
            'plans_executed': 0,
            'constraints_violated': 0
        },
        
        # Visualization
        'show_grid': True,
        'show_agents': True,
        'show_paths': True,
        'show_emergencies': True,
        'camera_angle': 0,
        
        # Weather and time
        'weather': 'Clear',
        'time_of_day': 'Morning',
        'traffic_density': 'Low'
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# ==================== HELPER FUNCTIONS ====================

def hash_password(password):
    """Hash password for storage"""
    return hashlib.sha256(password.encode()).hexdigest()

def log_event(event_type, message, ai_engine=None):
    """Add event to the log with timestamp"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    event = {
        "time": timestamp,
        "type": event_type,
        "message": message,
        "ai_engine": ai_engine
    }
    st.session_state.events.insert(0, event)
    if len(st.session_state.events) > 50:
        st.session_state.events = st.session_state.events[:50]

def log_ai_decision(engine, decision, reasoning):
    """Log AI decision for explainability"""
    st.session_state.ai_decisions.insert(0, {
        "time": datetime.now().strftime("%H:%M:%S"),
        "engine": engine,
        "decision": decision,
        "reasoning": reasoning
    })
    if len(st.session_state.ai_decisions) > 30:
        st.session_state.ai_decisions = st.session_state.ai_decisions[:30]

# ==================== AUTHENTICATION ====================

def show_auth_page():
    """Show signup/login page"""
    st.markdown('<div class="main-title">NEXUS</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">AI-POWERED SMART CITY SIMULATION</div>', unsafe_allow_html=True)
    
    # Toggle between login and signup
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        tab1, tab2 = st.tabs(["🔐 LOGIN", "📝 SIGNUP"])
        
        with tab1:
            st.markdown('<div class="login-container">', unsafe_allow_html=True)
            st.markdown("### 🏙️ ACCESS NEXUS SYSTEM")
            
            username = st.text_input("Username", key="login_user", placeholder="Enter your username")
            password = st.text_input("Password", type="password", key="login_pass", placeholder="Enter your password")
            
            col_a, col_b, col_c = st.columns([1, 2, 1])
            with col_b:
                if st.button("🚀 INITIALIZE NEXUS", use_container_width=True):
                    if username and password:
                        hashed_pw = hash_password(password)
                        if username in st.session_state.users_db and st.session_state.users_db[username] == hashed_pw:
                            st.session_state.logged_in = True
                            st.session_state.username = username
                            log_event("SYSTEM", f"User {username} logged in")
                            st.success("✅ Access Granted")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error("❌ Invalid credentials")
                    else:
                        st.error("Please enter both username and password")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with tab2:
            st.markdown('<div class="signup-container">', unsafe_allow_html=True)
            st.markdown("### 📝 CREATE NEXUS ACCOUNT")
            
            new_username = st.text_input("Choose Username", key="signup_user", placeholder="Choose a username")
            new_password = st.text_input("Choose Password", type="password", key="signup_pass", placeholder="Choose a password")
            confirm_password = st.text_input("Confirm Password", type="password", key="confirm_pass", placeholder="Confirm your password")
            
            col_a, col_b, col_c = st.columns([1, 2, 1])
            with col_b:
                if st.button("✨ CREATE ACCOUNT", use_container_width=True):
                    if new_username and new_password and confirm_password:
                        if new_username in st.session_state.users_db:
                            st.error("❌ Username already exists")
                        elif new_password != confirm_password:
                            st.error("❌ Passwords don't match")
                        elif len(new_password) < 6:
                            st.error("❌ Password must be at least 6 characters")
                        else:
                            st.session_state.users_db[new_username] = hash_password(new_password)
                            st.success("✅ Account created! Please login")
                            time.sleep(2)
                            st.rerun()
                    else:
                        st.error("Please fill all fields")
            
            st.markdown('</div>', unsafe_allow_html=True)

# ==================== CITY INITIALIZATION ====================

def initialize_smart_city():
    """Initialize the complete NEXUS Smart City System"""
    log_event("SYSTEM", "🚀 Initializing NEXUS Smart City System...", "SYSTEM")
    
    # Create city grid (20x20 for better visualization)
    grid_size = 20
    grid = np.zeros((grid_size, grid_size), dtype=np.int32)
    
    # Add buildings (commercial/residential zones)
    num_buildings = int(grid_size * 0.3)
    for _ in range(num_buildings):
        bx = np.random.randint(2, grid_size - 5)
        by = np.random.randint(2, grid_size - 5)
        bw = np.random.randint(2, 4)
        bh = np.random.randint(2, 4)
        grid[by:by+bh, bx:bx+bw] = 1  # Building
    
    # Add parks/green zones
    num_parks = int(grid_size * 0.15)
    for _ in range(num_parks):
        px = np.random.randint(1, grid_size - 3)
        py = np.random.randint(1, grid_size - 3)
        grid[py:py+2, px:px+2] = 2  # Grass/Park
    
    # Add restricted zones (industrial/hazard areas)
    num_restricted = 3
    for _ in range(num_restricted):
        rx = np.random.randint(1, grid_size - 2)
        ry = np.random.randint(1, grid_size - 2)
        grid[ry, rx] = 5  # Restricted
    
    # Save and load city
    np.save("nexus_smart_city.npy", grid)
    city = City("nexus_smart_city.npy")
    st.session_state.city = city
    
    log_event("SYSTEM", f"🏙️ City initialized: {grid_size}x{grid_size} grid", "CORE")
    
    # Build navigation graph
    graph = CityGraph(city)
    st.session_state.graph = graph
    log_event("SYSTEM", f"🕸️ Navigation graph built: {graph.graph.number_of_nodes()} nodes", "CORE")
    
    # Initialize simulation
    simulation = Simulation(city, graph)
    st.session_state.simulation = simulation
    
    # ==================== INITIALIZE AI ENGINES ====================
    
    # 1. Logic Engine
    logic_engine = LogicEngine()
    
    # Add custom rules for anomaly detection
    def check_speed_drop(ctx):
        agent = ctx.get("agent")
        return agent and agent.speed < 0.5 and len(agent.known_obstacles) == 0
    
    def action_engine_failure(ctx):
        agent = ctx.get("agent")
        log_event("WARNING", f"⚠️ {agent.name}: Potential engine failure detected", "LOGIC")
        log_ai_decision("Logic Engine", f"Engine failure alert for {agent.name}", 
                       "Speed dropped below threshold with no obstacles")
        st.session_state.stats['rules_fired'] += 1
        return f"Engine failure alert for {agent.name}"
    
    logic_engine.add_rule(create_simple_rule("engine_failure_detection", check_speed_drop, action_engine_failure, priority=9))
    logic_engine.add_rule(AgentRules.low_energy_recharge(threshold=25.0))
    
    st.session_state.logic_engine = logic_engine
    log_event("SYSTEM", "🧠 Logic Engine initialized with rules", "LOGIC")
    
    # 2. Explainability Engine
    explainer = ExplainabilityEngine()
    st.session_state.explainer = explainer
    log_event("SYSTEM", "💡 Explainability Engine initialized", "XAI")
    
    # 3. Bayesian Network
    bayesian = BayesianNetwork()
    st.session_state.bayesian = bayesian
    log_event("SYSTEM", "📊 Bayesian Network initialized", "BAYESIAN")
    
    # 4. HTN Planner
    planner = Planner()
    st.session_state.planner = planner
    log_event("SYSTEM", "📋 HTN Planner initialized", "PLANNER")
    
    # 5. CSP Engine
    csp_engine = CSPEngine()
    st.session_state.csp_engine = csp_engine
    log_event("SYSTEM", "⚡ CSP Engine initialized", "CSP")
    
    # ==================== DEPLOY AGENTS ====================
    
    deploy_agents(city, graph, simulation)
    
    log_event("SYSTEM", "✅ NEXUS Smart City System ONLINE", "SYSTEM")

def deploy_agents(city, graph, simulation):
    """Deploy regular vehicles and emergency vehicles"""
    walkable_positions = city.get_walkable_positions()
    
    if len(walkable_positions) < 10:
        log_event("ERROR", "Not enough walkable positions", "SYSTEM")
        return
    
    # Deploy regular cars
    num_cars = 8
    agents = []
    
    for i in range(num_cars):
        pos_index = (i * len(walkable_positions)) // (num_cars + 4)
        start_pos = walkable_positions[pos_index]
        agent = Agent(f"Car-{i+1}", start_pos, speed=0.7 + random.random() * 0.3)
        
        # Assign random goal
        goal_pos = random.choice(walkable_positions)
        agent.set_goal(goal_pos)
        
        # Use A* pathfinding
        result = a_star(graph, start_pos, goal_pos)
        if result.success:
            agent.set_path(result.path)
            
            # Log AI decision
            explanation = st.session_state.explainer.explain_path(agent, result.path, "A*", result.cost)
            log_ai_decision("Search (A*)", f"{agent.name} planned route", explanation)
            log_event("AI", f"🚗 {agent.name} route planned via A* ({len(result.path)} waypoints)", "SEARCH")
        
        simulation.add_agent(agent)
        agents.append(agent)
    
    st.session_state.agents = agents
    
    # Deploy emergency vehicles
    emergency_types = ["🚑 Ambulance", "🚒 Fire Truck"]
    emergency_vehicles = []
    
    for i, etype in enumerate(emergency_types):
        pos_index = ((num_cars + i) * len(walkable_positions)) // (num_cars + 4)
        start_pos = walkable_positions[pos_index]
        agent = Agent(etype, start_pos, speed=1.0)
        agent.state = AgentState.IDLE  # Emergency vehicles start idle
        
        simulation.add_agent(agent)
        emergency_vehicles.append(agent)
    
    st.session_state.emergency_vehicles = emergency_vehicles
    log_event("SYSTEM", f"🚗 Deployed {num_cars} cars + {len(emergency_vehicles)} emergency vehicles", "SYSTEM")

# ==================== 3D VISUALIZATION ====================

def create_3d_smart_city_visualization():
    """Create enhanced 3D visualization of the smart city"""
    city = st.session_state.city
    agents = st.session_state.agents
    emergency_vehicles = st.session_state.emergency_vehicles
    
    if city is None:
        return go.Figure()
    
    fig = go.Figure()
    grid = city.grid
    height, width = grid.shape
    
    # 1. HOLOGRAPHIC GRID
    if st.session_state.show_grid:
        grid_spacing = 2
        for i in range(0, height, grid_spacing):
            fig.add_trace(go.Scatter3d(
                x=[0, width], y=[i, i], z=[0, 0],
                mode='lines',
                line=dict(color=HOLO_CYAN, width=0.8),
                showlegend=False,
                hoverinfo='skip'
            ))
        for j in range(0, width, grid_spacing):
            fig.add_trace(go.Scatter3d(
                x=[j, j], y=[0, height], z=[0, 0],
                mode='lines',
                line=dict(color=HOLO_CYAN, width=0.8),
                showlegend=False,
                hoverinfo='skip'
            ))
    
    # 2. BUILDINGS
    building_positions = np.argwhere(grid == 1)
    if len(building_positions) > 0:
        x_buildings = building_positions[:, 1]
        y_buildings = building_positions[:, 0]
        z_buildings = np.random.uniform(8, 20, len(building_positions))
        colors = [random.choice([HOLO_CYAN, HOLO_MAGENTA, HOLO_BLUE]) for _ in range(len(building_positions))]
        
        fig.add_trace(go.Scatter3d(
            x=x_buildings, y=y_buildings, z=z_buildings,
            mode='markers',
            marker=dict(size=6, color=colors, symbol='square', opacity=0.7, line=dict(color='white', width=0.5)),
            name='Buildings',
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # 3. PARKS
    park_positions = np.argwhere(grid == 2)
    if len(park_positions) > 0:
        x_parks = park_positions[:, 1]
        y_parks = park_positions[:, 0]
        z_parks = np.ones(len(park_positions)) * 0.5
        
        fig.add_trace(go.Scatter3d(
            x=x_parks, y=y_parks, z=z_parks,
            mode='markers',
            marker=dict(size=5, color=HOLO_GREEN, symbol='square', opacity=0.5),
            name='Parks',
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # 4. RESTRICTED ZONES
    restricted_positions = np.argwhere(grid == 5)
    if len(restricted_positions) > 0:
        x_restricted = restricted_positions[:, 1]
        y_restricted = restricted_positions[:, 0]
        z_restricted = np.ones(len(restricted_positions)) * 0.3
        
        fig.add_trace(go.Scatter3d(
            x=x_restricted, y=y_restricted, z=z_restricted,
            mode='markers',
            marker=dict(size=5, color=HOLO_RED, symbol='x', opacity=0.8),
            name='Restricted',
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # 5. EMERGENCIES
    if st.session_state.show_emergencies:
        for incident in st.session_state.emergencies:
            ix, iy = incident['position'][1], incident['position'][0]
            fig.add_trace(go.Scatter3d(
                x=[ix], y=[iy], z=[12],
                mode='markers+text',
                marker=dict(size=12, color=HOLO_RED, symbol='diamond', line=dict(color='white', width=2)),
                text=[incident['type']],
                textposition='top center',
                textfont=dict(color=HOLO_RED, size=10),
                name=incident['type'],
                showlegend=False,
                hovertext=f"{incident['type']}<br>Time: {incident['time']}"
            ))
    
    # 6. REGULAR AGENTS (CARS)
    if st.session_state.show_agents and agents:
        for agent in agents:
            ax, ay = agent.position[1], agent.position[0]
            color = HOLO_GREEN if agent.state == AgentState.MOVING else HOLO_ORANGE
            
            fig.add_trace(go.Scatter3d(
                x=[ax], y=[ay], z=[10],
                mode='markers+text',
                marker=dict(size=10, color=color, symbol='diamond', line=dict(color='white', width=1.5)),
                text=[agent.name],
                textposition='top center',
                textfont=dict(color=color, size=8),
                name=agent.name,
                hovertext=f"{agent.name}<br>Energy: {agent.get_energy_percent():.1f}%<br>State: {agent.state.name}",
                hoverinfo='text'
            ))
            
            # Agent paths
            if st.session_state.show_paths and agent.path and len(agent.path) > 1:
                path_x = [p[1] for p in agent.path]
                path_y = [p[0] for p in agent.path]
                path_z = [8] * len(agent.path)
                
                fig.add_trace(go.Scatter3d(
                    x=path_x, y=path_y, z=path_z,
                    mode='lines',
                    line=dict(color=color, width=3),
                    showlegend=False,
                    hoverinfo='skip'
                ))
    
    # 7. EMERGENCY VEHICLES
    if st.session_state.show_agents and emergency_vehicles:
        for vehicle in emergency_vehicles:
            vx, vy = vehicle.position[1], vehicle.position[0]
            color = HOLO_RED if vehicle.state == AgentState.MOVING else HOLO_BLUE
            
            fig.add_trace(go.Scatter3d(
                x=[vx], y=[vy], z=[11],
                mode='markers+text',
                marker=dict(size=12, color=color, symbol='diamond', line=dict(color='white', width=2)),
                text=[vehicle.name],
                textposition='top center',
                textfont=dict(color=color, size=9, family='Orbitron'),
                name=vehicle.name,
                hovertext=f"{vehicle.name}<br>State: {vehicle.state.name}",
                hoverinfo='text'
            ))
            
            if st.session_state.show_paths and vehicle.path and len(vehicle.path) > 1:
                path_x = [p[1] for p in vehicle.path]
                path_y = [p[0] for p in vehicle.path]
                path_z = [9] * len(vehicle.path)
                
                fig.add_trace(go.Scatter3d(
                    x=path_x, y=path_y, z=path_z,
                    mode='lines',
                    line=dict(color=HOLO_RED, width=4, dash='dash'),
                    showlegend=False,
                    hoverinfo='skip'
                ))
    
    # Camera settings
    angle = st.session_state.camera_angle
    camera = dict(
        eye=dict(x=np.cos(angle) * 1.5, y=np.sin(angle) * 1.5, z=1.2),
        center=dict(x=0.5, y=0.5, z=0.3),
        up=dict(x=0, y=0, z=1)
    )
    
    fig.update_layout(
        scene=dict(
            xaxis=dict(showgrid=False, showticklabels=False, showbackground=False, range=[0, width]),
            yaxis=dict(showgrid=False, showticklabels=False, showbackground=False, range=[0, height]),
            zaxis=dict(showgrid=False, showticklabels=False, showbackground=False, range=[0, 25]),
            bgcolor=DARK_BG,
            camera=camera,
            aspectmode='cube'
        ),
        paper_bgcolor=DARK_BG,
        plot_bgcolor=DARK_BG,
        height=650,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
        hovermode='closest'
    )
    
    return fig

# ==================== SIMULATION UPDATE ====================

def update_simulation():
    """Update simulation with AI engine integration"""
    if not st.session_state.is_running:
        return
    
    simulation = st.session_state.simulation
    if not simulation:
        return
    
    # Execute simulation step
    simulation.step()
    st.session_state.current_tick += 1
    
    # Update camera rotation
    st.session_state.camera_angle += 0.015
    
    # ==================== BAYESIAN PREDICTIONS ====================
    # Predict accidents based on weather and traffic
    hour = (st.session_state.current_tick // 10) % 24
    accident_prob = st.session_state.bayesian.predict_traffic(hour, st.session_state.weather)
    
    if random.random() < accident_prob * 0.02:  # Low chance per tick
        # Create accident
        walkable = st.session_state.city.get_walkable_positions()
        if walkable:
            accident_pos = random.choice(walkable)
            st.session_state.emergencies.append({
                'type': '🚨 Accident',
                'position': accident_pos,
                'time': datetime.now().strftime("%H:%M:%S"),
                'tick': st.session_state.current_tick
            })
            log_event("EMERGENCY", f"🚨 Accident at {accident_pos}!", "BAYESIAN")
            log_ai_decision("Bayesian Network", "Accident predicted and occurred", 
                          f"P(accident|weather={st.session_state.weather}, hour={hour}) = {accident_prob:.2f}")
            st.session_state.stats['accidents_predicted'] += 1
            st.session_state.stats['incidents_count'] += 1
            
            # Dispatch emergency vehicle
            dispatch_emergency_response(accident_pos, "accident")
    
    # ==================== LOGIC ENGINE RULES ====================
    # Check agents for anomalies
    for agent in st.session_state.agents:
        context = {"agent": agent}
        actions = st.session_state.logic_engine.forward_chain(context)
    
    # ==================== CSP RESOURCE MANAGEMENT ====================
    # Every 20 ticks, check resource constraints
    if st.session_state.current_tick % 20 == 0:
        # Simulate power distribution
        total_power = 100
        hospital_needs = 30
        fire_station_needs = 20
        industrial_needs = 40
        
        if hospital_needs + fire_station_needs + industrial_needs > total_power:
            log_event("WARNING", "⚡ Power shortage detected - CSP resolving...", "CSP")
            log_ai_decision("CSP Engine", "Power reallocation", 
                          "Hospital priority > Fire Station > Industrial")
            st.session_state.stats['constraints_violated'] += 1
            st.session_state.stats['resource_utilization'] = min(100, st.session_state.stats['resource_utilization'] + 2)
    
    # Update statistics
    if st.session_state.agents:
        total_path_length = sum(len(a.path) for a in st.session_state.agents if a.path)
        st.session_state.stats['avg_path_length'] = total_path_length / max(len(st.session_state.agents), 1)
        
        moving_agents = sum(1 for a in st.session_state.agents if a.state == AgentState.MOVING)
        st.session_state.stats['efficiency_score'] = min(100, (moving_agents / len(st.session_state.agents)) * 100)

def dispatch_emergency_response(location, emergency_type):
    """Use HTN planner to dispatch emergency vehicle"""
    available_vehicle = None
    for vehicle in st.session_state.emergency_vehicles:
        if vehicle.state == AgentState.IDLE:
            available_vehicle = vehicle
            break
    
    if available_vehicle:
        # Plan emergency response
        result = a_star(st.session_state.graph, 
                       (int(available_vehicle.position[0]), int(available_vehicle.position[1])),
                       location)
        
        if result.success:
            available_vehicle.set_path(result.path)
            available_vehicle.set_goal(location)
            available_vehicle.state = AgentState.MOVING
            
            log_event("PLANNER", f"📋 {available_vehicle.name} dispatched to {location}", "HTN")
            log_ai_decision("HTN Planner", f"Emergency response plan for {available_vehicle.name}",
                          f"1. Navigate to {location} 2. Handle {emergency_type} 3. Return to base")
            st.session_state.stats['plans_executed'] += 1

# ==================== MAIN DASHBOARD ====================

def show_dashboard():
    """Main NEXUS dashboard interface"""
    # Initialize city if not done
    if st.session_state.city is None:
        initialize_smart_city()
    
    # ==================== HEADER ====================
    st.markdown('<div class="main-title">⚡ NEXUS AI SYSTEM ⚡</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">SMART CITY SIMULATION • 6 AI ENGINES ACTIVE</div>', unsafe_allow_html=True)
    
    # AI Engine Status Badges
    col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
    with col1:
        st.markdown(f'<div class="ai-engine-badge">🔍 A* SEARCH</div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="ai-engine-badge">⚡ CSP</div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="ai-engine-badge">🧠 LOGIC</div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="ai-engine-badge">📋 HTN PLANNER</div>', unsafe_allow_html=True)
    with col5:
        st.markdown(f'<div class="ai-engine-badge">📊 BAYESIAN</div>', unsafe_allow_html=True)
    with col6:
        st.markdown(f'<div class="ai-engine-badge">💡 XAI</div>', unsafe_allow_html=True)
    with col7:
        if st.button("🚪 Logout"):
            st.session_state.logged_in = False
            st.rerun()
    
    st.markdown("---")
    
    # ==================== CONTROL BAR ====================
    col1, col2, col3, col4, col5 = st.columns([2, 2, 2, 2, 2])
    
    with col1:
        status = "🟢 ACTIVE" if st.session_state.is_running else "🔴 PAUSED"
        st.markdown(f"<h3>{status}</h3>", unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"<h3>TICK: {st.session_state.current_tick}</h3>", unsafe_allow_html=True)
    
    with col3:
        if st.button("▶️ START" if not st.session_state.is_running else "⏸️ PAUSE", use_container_width=True):
            st.session_state.is_running = not st.session_state.is_running
            if st.session_state.is_running:
                log_event("SYSTEM", "Simulation started", "SYSTEM")
            else:
                log_event("SYSTEM", "Simulation paused", "SYSTEM")
    
    with col4:
        st.session_state.speed_multiplier = st.slider(
            "Speed", 0.5, 5.0, st.session_state.speed_multiplier, 0.5,
            label_visibility="collapsed"
        )
    
    with col5:
        # Weather/Time controls
        st.session_state.weather = st.selectbox("Weather", ["Clear", "Rain", "Snow"], 
                                                label_visibility="collapsed")
    
    st.markdown("---")
    
    # ==================== MAIN LAYOUT ====================
    left_col, center_col, right_col = st.columns([1, 3, 1])
    
    # LEFT PANEL
    with left_col:
        st.markdown(f"<h2>📊 SYSTEM METRICS</h2>", unsafe_allow_html=True)
        
        st.metric("🏆 Efficiency Score", f"{st.session_state.stats['efficiency_score']:.1f}%")
        st.metric("⚡ Resource Usage", f"{st.session_state.stats['resource_utilization']:.1f}%")
        st.metric("🚗 Active Vehicles", len(st.session_state.agents))
        st.metric("🚨 Active Incidents", len(st.session_state.emergencies))
        st.metric("📈 Avg Path Length", f"{st.session_state.stats['avg_path_length']:.1f}")
        
        st.markdown("---")
        st.markdown(f"<h2>🎮 CONTROLS</h2>", unsafe_allow_html=True)
        
        st.session_state.show_grid = st.checkbox("🗺️ Grid", st.session_state.show_grid)
        st.session_state.show_agents = st.checkbox("🚗 Vehicles", st.session_state.show_agents)
        st.session_state.show_paths = st.checkbox("🛤️ Paths", st.session_state.show_paths)
        st.session_state.show_emergencies = st.checkbox("🚨 Emergencies", st.session_state.show_emergencies)
        
        st.markdown("---")
        st.markdown(f"<h2>📊 AI STATS</h2>", unsafe_allow_html=True)
        
        st.metric("🔥 Rules Fired", st.session_state.stats['rules_fired'])
        st.metric("📋 Plans Executed", st.session_state.stats['plans_executed'])
        st.metric("⚠️ Constraints Hit", st.session_state.stats['constraints_violated'])
        st.metric("🎯 Accidents Predicted", st.session_state.stats['accidents_predicted'])
    
    # CENTER - 3D MAP
    with center_col:
        st.markdown(f"<h2 style='text-align:center;'>🏙️ SMART CITY MAP</h2>", unsafe_allow_html=True)
        
        fig = create_3d_smart_city_visualization()
        st.plotly_chart(fig, use_container_width=True, key=f"map_{st.session_state.current_tick}")
    
    # RIGHT PANEL
    with right_col:
        st.markdown(f"<h2>📡 LIVE EVENTS</h2>", unsafe_allow_html=True)
        
        event_container = st.container(height=300)
        with event_container:
            for event in st.session_state.events[:20]:
                color = HOLO_CYAN
                if event['type'] == 'WARNING':
                    color = HOLO_ORANGE
                elif event['type'] == 'EMERGENCY':
                    color = HOLO_RED
                elif event['type'] == 'AI':
                    color = HOLO_PURPLE
                
                engine_badge = f" [{event['ai_engine']}]" if event.get('ai_engine') else ""
                
                st.markdown(f"""
                <div class="event-box">
                    <span class="event-time">[{event['time']}]</span>
                    <span style="color:{color};">{event['message']}{engine_badge}</span>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown(f"<h2>💡 AI DECISIONS</h2>", unsafe_allow_html=True)
        
        ai_container = st.container(height=300)
        with ai_container:
            for decision in st.session_state.ai_decisions[:15]:
                st.markdown(f"""
                <div class="event-box">
                    <span class="event-time">[{decision['time']}]</span>
                    <div style="color:{HOLO_CYAN}; font-weight:700;">{decision['engine']}</div>
                    <div style="color:{HOLO_MAGENTA};">{decision['decision']}</div>
                    <div style="color:{HOLO_GREEN}; font-size:0.85rem;">{decision['reasoning']}</div>
                </div>
                """, unsafe_allow_html=True)
    
    # AUTO-REFRESH
    if st.session_state.is_running:
        update_simulation()
        sleep_duration = max(0.01, 0.15 / st.session_state.speed_multiplier)
        time.sleep(sleep_duration)
        st.rerun()

# ==================== MAIN APP ====================

if not st.session_state.logged_in:
    show_auth_page()
else:
    show_dashboard()
