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

# ==================== COLOR SCHEME (PROFESSIONAL MUTED) ====================
# Professional muted colors for a clean, business-like interface
DARK_BG = "#0d1117"  # Dark navy background
DARK_PANEL = "rgba(22, 27, 34, 0.95)"  # Subtle panel background

# Primary colors - muted and professional
PRIMARY_BLUE = "#2d7dd2"  # Calm professional blue
PRIMARY_GREEN = "#3fb950"  # Success green
PRIMARY_ORANGE = "#d97617"  # Warning orange
PRIMARY_RED = "#e5534b"  # Error/alert red
PRIMARY_PURPLE = "#8256d0"  # Accent purple

# Neutral colors for text and borders
TEXT_PRIMARY = "#e6edf3"  # Light text
TEXT_SECONDARY = "#8b949e"  # Muted text
BORDER_COLOR = "#30363d"  # Subtle borders
ACCENT_BLUE = "#58a6ff"  # Bright but not harsh accent

# Building colors - realistic and professional
BUILDING_COMMERCIAL = "#3a4556"  # Dark gray-blue
BUILDING_RESIDENTIAL = "#445566"  # Medium gray-blue
BUILDING_INDUSTRIAL = "#556677"  # Light gray-blue
PARK_GREEN = "#2ea043"  # Natural green
ROAD_GRAY = "#21262d"  # Asphalt gray

# ==================== PAGE CONFIGURATION ====================
st.set_page_config(
    page_title="NEXUS - AI Smart City",
    layout="wide",
    initial_sidebar_state="collapsed",
    page_icon="🏙️"
)

# ==================== CSS STYLING (PROFESSIONAL) ====================
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');
    
    * {{
        font-family: 'Inter', sans-serif !important;
    }}
    
    .stApp {{
        background: {DARK_BG};
    }}
    
    h1, h2, h3, h4 {{
        font-family: 'Inter', sans-serif !important;
        color: {TEXT_PRIMARY};
        font-weight: 600;
        letter-spacing: -0.5px;
    }}
    
    .main-title {{
        font-size: 2.5rem;
        text-align: center;
        color: {ACCENT_BLUE};
        margin-bottom: 10px;
        font-weight: 700;
        letter-spacing: -1px;
    }}
    
    .subtitle {{
        text-align: center;
        color: {TEXT_SECONDARY};
        font-size: 1rem;
        margin-bottom: 30px;
        font-weight: 500;
        letter-spacing: 0.5px;
    }}
    
    .stTabs [data-baseweb="tab-panel"] {{
        background-color: {DARK_PANEL};
        border: 1px solid {BORDER_COLOR};
        border-radius: 8px;
        padding: 24px;
    }}
    
    [data-testid="stMetricValue"] {{
        color: {PRIMARY_BLUE};
        font-size: 1.8rem !important;
        font-weight: 600;
    }}
    
    [data-testid="stMetricLabel"] {{
        color: {TEXT_SECONDARY};
        text-transform: uppercase;
        font-size: 0.75rem;
        letter-spacing: 0.5px;
        font-weight: 500;
    }}
    
    .stButton button {{
        background: {PRIMARY_BLUE};
        color: white;
        border: none;
        font-weight: 600;
        letter-spacing: 0.3px;
        transition: all 0.2s;
        border-radius: 6px;
    }}
    
    .stButton button:hover {{
        background: {ACCENT_BLUE};
        transform: translateY(-1px);
    }}
    
    .event-box {{
        background: {DARK_PANEL};
        border-left: 3px solid {PRIMARY_BLUE};
        padding: 10px 12px;
        margin: 6px 0;
        border-radius: 4px;
        font-size: 0.875rem;
    }}
    
    .event-time {{
        color: {TEXT_SECONDARY};
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
    }}
    
    .ai-engine-badge {{
        background: {DARK_PANEL};
        color: {TEXT_PRIMARY};
        padding: 6px 14px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
        margin: 3px;
        border: 1px solid {BORDER_COLOR};
    }}
    
    .emergency-badge {{
        background: {PRIMARY_RED};
        color: white;
        padding: 6px 14px;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
    }}
    
    .login-container {{
        background: {DARK_PANEL};
        border: 1px solid {BORDER_COLOR};
        border-radius: 12px;
        padding: 40px;
    }}
    
    .signup-container {{
        background: {DARK_PANEL};
        border: 1px solid {BORDER_COLOR};
        border-radius: 12px;
        padding: 40px;
    }}
    
    ::-webkit-scrollbar {{
        width: 10px;
        height: 10px;
    }}
    
    ::-webkit-scrollbar-track {{
        background: {DARK_BG};
    }}
    
    ::-webkit-scrollbar-thumb {{
        background: {BORDER_COLOR};
        border-radius: 5px;
    }}
    
    ::-webkit-scrollbar-thumb:hover {{
        background: {TEXT_SECONDARY};
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

# ==================== 3D VISUALIZATION (IMPROVED) ====================

def create_3d_building_mesh(x, y, width, depth, height, color):
    """Create a 3D building mesh for realistic visualization"""
    # Define the 8 vertices of the building cuboid
    vertices = [
        [x, y, 0], [x+width, y, 0], [x+width, y+depth, 0], [x, y+depth, 0],  # Bottom
        [x, y, height], [x+width, y, height], [x+width, y+depth, height], [x, y+depth, height]  # Top
    ]
    
    # Define the 12 triangles (2 per face, 6 faces)
    faces = [
        [0, 1, 2], [0, 2, 3],  # Bottom
        [4, 5, 6], [4, 6, 7],  # Top
        [0, 1, 5], [0, 5, 4],  # Front
        [2, 3, 7], [2, 7, 6],  # Back
        [0, 3, 7], [0, 7, 4],  # Left
        [1, 2, 6], [1, 6, 5]   # Right
    ]
    
    x_coords = [v[0] for v in vertices]
    y_coords = [v[1] for v in vertices]
    z_coords = [v[2] for v in vertices]
    
    i_faces = [f[0] for f in faces]
    j_faces = [f[1] for f in faces]
    k_faces = [f[2] for f in faces]
    
    return go.Mesh3d(
        x=x_coords, y=y_coords, z=z_coords,
        i=i_faces, j=j_faces, k=k_faces,
        color=color,
        opacity=0.85,
        showlegend=False,
        hoverinfo='skip'
    )

def create_3d_smart_city_visualization():
    """Create enhanced 3D visualization with proper building structures"""
    city = st.session_state.city
    agents = st.session_state.agents
    emergency_vehicles = st.session_state.emergency_vehicles
    
    if city is None:
        return go.Figure()
    
    fig = go.Figure()
    grid = city.grid
    height, width = grid.shape
    
    # 1. GROUND PLANE (Subtle grid)
    if st.session_state.show_grid:
        grid_spacing = 2
        for i in range(0, height, grid_spacing):
            fig.add_trace(go.Scatter3d(
                x=[0, width], y=[i, i], z=[0, 0],
                mode='lines',
                line=dict(color=BORDER_COLOR, width=0.5),
                showlegend=False,
                hoverinfo='skip'
            ))
        for j in range(0, width, grid_spacing):
            fig.add_trace(go.Scatter3d(
                x=[j, j], y=[0, height], z=[0, 0],
                mode='lines',
                line=dict(color=BORDER_COLOR, width=0.5),
                showlegend=False,
                hoverinfo='skip'
            ))
    
    # 2. 3D BUILDINGS (Proper mesh structures)
    building_positions = np.argwhere(grid == 1)
    if len(building_positions) > 0:
        # Group adjacent buildings
        processed = set()
        for pos in building_positions:
            by, bx = pos
            if (by, bx) in processed:
                continue
            
            # Find building cluster
            bw = 1
            bh = 1
            
            # Check width
            while bx + bw < width and grid[by, bx + bw] == 1:
                bw += 1
            
            # Check height  
            while by + bh < height and all(grid[by + bh, bx + w] == 1 for w in range(bw)):
                bh += 1
            
            # Mark as processed
            for h in range(bh):
                for w in range(bw):
                    processed.add((by + h, bx + w))
            
            # Random building height
            building_height = np.random.uniform(10, 25)
            
            # Choose building color based on type
            building_colors = [BUILDING_COMMERCIAL, BUILDING_RESIDENTIAL, BUILDING_INDUSTRIAL]
            building_color = random.choice(building_colors)
            
            # Add 3D building mesh
            fig.add_trace(create_3d_building_mesh(
                bx, by, bw, bh, building_height, building_color
            ))
    
    # 3. PARKS (Flat green areas)
    park_positions = np.argwhere(grid == 2)
    if len(park_positions) > 0:
        x_parks = park_positions[:, 1]
        y_parks = park_positions[:, 0]
        z_parks = np.ones(len(park_positions)) * 0.2
        
        fig.add_trace(go.Scatter3d(
            x=x_parks, y=y_parks, z=z_parks,
            mode='markers',
            marker=dict(size=6, color=PARK_GREEN, symbol='square', opacity=0.7),
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
            marker=dict(size=5, color=PRIMARY_RED, symbol='x', opacity=0.8),
            name='Restricted',
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # 5. EMERGENCIES
    if st.session_state.show_emergencies:
        for incident in st.session_state.emergencies:
            ix, iy = incident['position'][1], incident['position'][0]
            fig.add_trace(go.Scatter3d(
                x=[ix], y=[iy], z=[15],
                mode='markers+text',
                marker=dict(size=10, color=PRIMARY_RED, symbol='diamond', line=dict(color='white', width=1.5)),
                text=[incident['type']],
                textposition='top center',
                textfont=dict(color=PRIMARY_RED, size=9),
                name=incident['type'],
                showlegend=False,
                hovertext=f"{incident['type']}<br>Time: {incident['time']}"
            ))
    
    # 6. REGULAR AGENTS (CARS)
    if st.session_state.show_agents and agents:
        for agent in agents:
            ax, ay = agent.position[1], agent.position[0]
            color = PRIMARY_GREEN if agent.state == AgentState.MOVING else PRIMARY_ORANGE
            
            fig.add_trace(go.Scatter3d(
                x=[ax], y=[ay], z=[8],
                mode='markers+text',
                marker=dict(size=8, color=color, symbol='circle', line=dict(color='white', width=1)),
                text=[agent.name],
                textposition='top center',
                textfont=dict(color=color, size=7),
                name=agent.name,
                hovertext=f"{agent.name}<br>Energy: {agent.get_energy_percent():.1f}%<br>State: {agent.state.name}",
                hoverinfo='text'
            ))
            
            # Agent paths
            if st.session_state.show_paths and agent.path and len(agent.path) > 1:
                path_x = [p[1] for p in agent.path]
                path_y = [p[0] for p in agent.path]
                path_z = [6] * len(agent.path)
                
                fig.add_trace(go.Scatter3d(
                    x=path_x, y=path_y, z=path_z,
                    mode='lines',
                    line=dict(color=color, width=2),
                    showlegend=False,
                    hoverinfo='skip'
                ))
    
    # 7. EMERGENCY VEHICLES
    if st.session_state.show_agents and emergency_vehicles:
        for vehicle in emergency_vehicles:
            vx, vy = vehicle.position[1], vehicle.position[0]
            color = PRIMARY_RED if vehicle.state == AgentState.MOVING else PRIMARY_BLUE
            
            fig.add_trace(go.Scatter3d(
                x=[vx], y=[vy], z=[9],
                mode='markers+text',
                marker=dict(size=10, color=color, symbol='diamond', line=dict(color='white', width=1.5)),
                text=[vehicle.name],
                textposition='top center',
                textfont=dict(color=color, size=8),
                name=vehicle.name,
                hovertext=f"{vehicle.name}<br>State: {vehicle.state.name}",
                hoverinfo='text'
            ))
            
            if st.session_state.show_paths and vehicle.path and len(vehicle.path) > 1:
                path_x = [p[1] for p in vehicle.path]
                path_y = [p[0] for p in vehicle.path]
                path_z = [7] * len(vehicle.path)
                
                fig.add_trace(go.Scatter3d(
                    x=path_x, y=path_y, z=path_z,
                    mode='lines',
                    line=dict(color=PRIMARY_RED, width=3, dash='dash'),
                    showlegend=False,
                    hoverinfo='skip'
                ))
    
    # Camera settings - subtle rotation
    angle = st.session_state.camera_angle
    camera = dict(
        eye=dict(x=np.cos(angle) * 1.5, y=np.sin(angle) * 1.5, z=1.1),
        center=dict(x=0.5, y=0.5, z=0.25),
        up=dict(x=0, y=0, z=1)
    )
    
    fig.update_layout(
        scene=dict(
            xaxis=dict(showgrid=False, showticklabels=False, showbackground=False, range=[0, width]),
            yaxis=dict(showgrid=False, showticklabels=False, showbackground=False, range=[0, height]),
            zaxis=dict(showgrid=False, showticklabels=False, showbackground=False, range=[0, 30]),
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

def restart_simulation():
    """Restart simulation with new city generation"""
    st.session_state.city = None
    st.session_state.graph = None
    st.session_state.simulation = None
    st.session_state.agents = []
    st.session_state.emergency_vehicles = []
    st.session_state.is_running = False
    st.session_state.current_tick = 0
    st.session_state.events = []
    st.session_state.incidents = []
    st.session_state.emergencies = []
    st.session_state.ai_decisions = []
    st.session_state.camera_angle = 0
    
    # Reset statistics
    st.session_state.stats = {
        'efficiency_score': 100.0,
        'avg_search_time': 0.0,
        'resource_utilization': 85.0,
        'incidents_count': 0,
        'avg_path_length': 0.0,
        'accidents_predicted': 0,
        'rules_fired': 0,
        'plans_executed': 0,
        'constraints_violated': 0
    }
    
    log_event("SYSTEM", "🔄 Simulation restarted with new city generation", "SYSTEM")
    # Re-initialize
    initialize_smart_city()

# ==================== MAIN DASHBOARD ====================

def show_dashboard():
    """Main NEXUS dashboard interface"""
    # Initialize city if not done
    if st.session_state.city is None:
        initialize_smart_city()
    
    # ==================== HEADER ====================
    st.markdown('<div class="main-title">NEXUS AI SYSTEM</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Smart City Simulation • 6 AI Engines Active</div>', unsafe_allow_html=True)
    
    # AI Engine Status Badges
    col1, col2, col3, col4, col5, col6, col7, col8 = st.columns(8)
    with col1:
        st.markdown(f'<div class="ai-engine-badge">🔍 A* Search</div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="ai-engine-badge">⚡ CSP</div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="ai-engine-badge">🧠 Logic</div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="ai-engine-badge">📋 HTN</div>', unsafe_allow_html=True)
    with col5:
        st.markdown(f'<div class="ai-engine-badge">📊 Bayesian</div>', unsafe_allow_html=True)
    with col6:
        st.markdown(f'<div class="ai-engine-badge">💡 XAI</div>', unsafe_allow_html=True)
    with col7:
        if st.button("🔄 Restart", help="Restart simulation with new city"):
            restart_simulation()
            st.rerun()
    with col8:
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
        st.markdown(f"<h2>📡 Live Events</h2>", unsafe_allow_html=True)
        
        event_container = st.container(height=300)
        with event_container:
            for event in st.session_state.events[:20]:
                color = ACCENT_BLUE
                if event['type'] == 'WARNING':
                    color = PRIMARY_ORANGE
                elif event['type'] == 'EMERGENCY':
                    color = PRIMARY_RED
                elif event['type'] == 'AI':
                    color = PRIMARY_PURPLE
                elif event['type'] == 'SYSTEM':
                    color = PRIMARY_GREEN
                
                engine_badge = f" [{event['ai_engine']}]" if event.get('ai_engine') else ""
                
                st.markdown(f"""
                <div class="event-box">
                    <span class="event-time">[{event['time']}]</span>
                    <span style="color:{color};">{event['message']}{engine_badge}</span>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown(f"<h2>💡 AI Decisions</h2>", unsafe_allow_html=True)
        
        ai_container = st.container(height=300)
        with ai_container:
            for decision in st.session_state.ai_decisions[:15]:
                st.markdown(f"""
                <div class="event-box">
                    <span class="event-time">[{decision['time']}]</span>
                    <div style="color:{PRIMARY_BLUE}; font-weight:600;">{decision['engine']}</div>
                    <div style="color:{TEXT_PRIMARY};">{decision['decision']}</div>
                    <div style="color:{TEXT_SECONDARY}; font-size:0.85rem;">{decision['reasoning']}</div>
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
