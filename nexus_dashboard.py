"""
NEXUS 3D HOLOGRAPHIC DASHBOARD
Professional-Grade City Navigation Simulation
"""

import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from datetime import datetime
import time
from pathlib import Path
import sys
import random

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.city import City
from core.graph import CityGraph
from core.simulation import Simulation, SimulationEvent
from core.agent import Agent, AgentState
from ai.search import a_star, SearchResult

# -------------------------
# COLOR SCHEME CONSTANTS
# -------------------------
HOLO_CYAN = "#00ffff"
HOLO_MAGENTA = "#ff00ff"
HOLO_BLUE = "#0080ff"
HOLO_GREEN = "#00ff80"
HOLO_PINK = "#ff0080"
HOLO_PURPLE = "#7b2ff7"
DARK_BG = "#0a0a15"
DARK_PANEL = "rgba(10, 20, 40, 0.8)"

# -------------------------
# PAGE CONFIGURATION
# -------------------------
st.set_page_config(
    page_title="NEXUS ENGINE",
    layout="wide",
    initial_sidebar_state="collapsed",
    page_icon="⚡"
)

# -------------------------
# CYBERPUNK CSS STYLING
# -------------------------
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@300;400;500;600;700&display=swap');
    
    * {{
        font-family: 'Rajdhani', sans-serif !important;
    }}
    
    .stApp {{
        background: linear-gradient(135deg, {DARK_BG} 0%, #0f0f20 100%);
    }}
    
    /* Glowing headers */
    h1, h2, h3 {{
        color: {HOLO_CYAN};
        text-shadow: 0 0 10px {HOLO_CYAN}, 0 0 20px {HOLO_CYAN};
        letter-spacing: 2px;
    }}
    
    /* Neon borders */
    .stTabs [data-baseweb="tab-panel"] {{
        background-color: {DARK_PANEL};
        border: 1px solid {HOLO_CYAN};
        box-shadow: 0 0 15px {HOLO_CYAN};
        border-radius: 8px;
        padding: 20px;
        backdrop-filter: blur(10px);
    }}
    
    /* Metric cards */
    [data-testid="stMetricValue"] {{
        color: {HOLO_CYAN};
        text-shadow: 0 0 10px {HOLO_CYAN};
        font-size: 2rem !important;
    }}
    
    [data-testid="stMetricLabel"] {{
        color: {HOLO_MAGENTA};
        text-transform: uppercase;
        letter-spacing: 2px;
    }}
    
    /* Buttons */
    .stButton button {{
        background: linear-gradient(135deg, {HOLO_BLUE}, {HOLO_PURPLE});
        color: white;
        border: 2px solid {HOLO_CYAN};
        box-shadow: 0 0 20px {HOLO_CYAN};
        font-weight: 600;
        letter-spacing: 1px;
        transition: all 0.3s;
    }}
    
    .stButton button:hover {{
        box-shadow: 0 0 30px {HOLO_CYAN}, 0 0 40px {HOLO_MAGENTA};
        transform: translateY(-2px);
    }}
    
    /* Checkboxes and toggles */
    .stCheckbox label {{
        color: {HOLO_CYAN} !important;
        text-shadow: 0 0 5px {HOLO_CYAN};
    }}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {{
        width: 10px;
        height: 10px;
    }}
    
    ::-webkit-scrollbar-track {{
        background: {DARK_BG};
    }}
    
    ::-webkit-scrollbar-thumb {{
        background: linear-gradient(180deg, {HOLO_CYAN}, {HOLO_PURPLE});
        border-radius: 5px;
    }}
    
    /* Event feed styling */
    .event-box {{
        background: {DARK_PANEL};
        border-left: 3px solid {HOLO_CYAN};
        padding: 10px;
        margin: 5px 0;
        border-radius: 4px;
        backdrop-filter: blur(10px);
        font-size: 0.9rem;
    }}
    
    .event-time {{
        color: {HOLO_MAGENTA};
        font-weight: 600;
    }}
    
    .event-msg {{
        color: {HOLO_CYAN};
    }}
    
    /* Pulse animation for active elements */
    @keyframes pulse {{
        0%, 100% {{ opacity: 1; }}
        50% {{ opacity: 0.5; }}
    }}
    
    .pulse {{
        animation: pulse 2s infinite;
    }}
    
    /* Login page styling */
    .login-container {{
        background: {DARK_PANEL};
        border: 2px solid {HOLO_CYAN};
        box-shadow: 0 0 30px {HOLO_CYAN};
        border-radius: 15px;
        padding: 40px;
        backdrop-filter: blur(15px);
    }}
    
    .logo-text {{
        font-size: 4rem;
        font-weight: 700;
        color: {HOLO_CYAN};
        text-shadow: 0 0 20px {HOLO_CYAN}, 0 0 40px {HOLO_CYAN};
        letter-spacing: 10px;
        text-align: center;
        margin-bottom: 30px;
    }}
</style>
""", unsafe_allow_html=True)

# -------------------------
# SESSION STATE INITIALIZATION
# -------------------------
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'simulation' not in st.session_state:
    st.session_state.simulation = None
if 'city' not in st.session_state:
    st.session_state.city = None
if 'graph' not in st.session_state:
    st.session_state.graph = None
if 'agents' not in st.session_state:
    st.session_state.agents = []
if 'is_running' not in st.session_state:
    st.session_state.is_running = False
if 'current_tick' not in st.session_state:
    st.session_state.current_tick = 0
if 'events' not in st.session_state:
    st.session_state.events = []
if 'selected_agent_id' not in st.session_state:
    st.session_state.selected_agent_id = None
if 'speed_multiplier' not in st.session_state:
    st.session_state.speed_multiplier = 1.0
if 'show_grid' not in st.session_state:
    st.session_state.show_grid = True
if 'show_agents' not in st.session_state:
    st.session_state.show_agents = True
if 'show_paths' not in st.session_state:
    st.session_state.show_paths = True
if 'camera_rotation' not in st.session_state:
    st.session_state.camera_rotation = 0

# -------------------------
# HELPER FUNCTIONS
# -------------------------

def log_event(event_type, message):
    """Add event to the log with timestamp"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    event = {
        "time": timestamp,
        "type": event_type,
        "message": message
    }
    st.session_state.events.insert(0, event)
    # Keep only last 30 events
    if len(st.session_state.events) > 30:
        st.session_state.events = st.session_state.events[:30]

def initialize_simulation():
    """Initialize the NEXUS simulation system"""
    log_event("SYSTEM", "Initializing NEXUS Engine...")
    
    # Create city
    grid_size = 80
    grid = np.zeros((grid_size, grid_size), dtype=np.int32)
    
    # Add buildings with varied heights
    num_buildings = int(grid_size * 0.35)
    for _ in range(num_buildings):
        bx = np.random.randint(5, grid_size - 10)
        by = np.random.randint(5, grid_size - 10)
        bw = np.random.randint(3, 8)
        bh = np.random.randint(3, 8)
        grid[by:by+bh, bx:bx+bw] = 1  # Building
    
    # Add parks
    num_parks = int(grid_size * 0.15)
    for _ in range(num_parks):
        px = np.random.randint(5, grid_size - 5)
        py = np.random.randint(5, grid_size - 5)
        pw = np.random.randint(2, 5)
        ph = np.random.randint(2, 5)
        grid[py:py+ph, px:px+pw] = 2  # Grass/Park
    
    # Save and load city
    np.save("temp_city.npy", grid)
    city = City("temp_city.npy")
    st.session_state.city = city
    
    log_event("SYSTEM", f"City loaded: {grid_size}x{grid_size} grid")
    
    # Build graph
    graph = CityGraph(city)
    st.session_state.graph = graph
    
    log_event("SYSTEM", f"Navigation graph built: {graph.graph.number_of_nodes()} nodes")
    
    # Initialize simulation
    simulation = Simulation(city, graph)
    st.session_state.simulation = simulation
    
    # Deploy agents
    num_agents = 12
    agents = []
    walkable_positions = city.get_walkable_positions()
    
    for i in range(num_agents):
        if i < len(walkable_positions):
            start_pos = walkable_positions[i * (len(walkable_positions) // num_agents)]
            agent = Agent(f"Agent-{i+1}", start_pos, speed=0.8)
            
            # Assign random goal and path
            goal_pos = random.choice(walkable_positions)
            agent.set_goal(goal_pos)
            
            result = a_star(graph, start_pos, goal_pos)
            if result.success:
                agent.set_path(result.path)
                log_event("AI", f"{agent.name} planned path via A* ({len(result.path)} waypoints)")
            
            simulation.add_agent(agent)
            agents.append(agent)
    
    st.session_state.agents = agents
    log_event("SYSTEM", f"Deployed {len(agents)} agents")
    log_event("SYSTEM", "✓ NEXUS Engine ready")

def create_3d_city_visualization():
    """Create the 3D holographic city visualization"""
    city = st.session_state.city
    agents = st.session_state.agents
    
    if city is None:
        return go.Figure()
    
    fig = go.Figure()
    
    # Get grid data
    grid = city.grid
    height, width = grid.shape
    
    # 1. HOLOGRAPHIC GRID BASE
    if st.session_state.show_grid:
        # Grid lines on ground
        grid_spacing = 5
        for i in range(0, height, grid_spacing):
            fig.add_trace(go.Scatter3d(
                x=[0, width],
                y=[i, i],
                z=[0, 0],
                mode='lines',
                line=dict(color=HOLO_CYAN, width=1),
                showlegend=False,
                hoverinfo='skip'
            ))
        
        for j in range(0, width, grid_spacing):
            fig.add_trace(go.Scatter3d(
                x=[j, j],
                y=[0, height],
                z=[0, 0],
                mode='lines',
                line=dict(color=HOLO_CYAN, width=1),
                showlegend=False,
                hoverinfo='skip'
            ))
    
    # 2. GLOWING NEON BUILDINGS
    building_positions = np.argwhere(grid == 1)  # Buildings
    if len(building_positions) > 0:
        x_buildings = building_positions[:, 1]
        y_buildings = building_positions[:, 0]
        
        # Randomize building heights
        z_buildings = np.random.uniform(12, 30, len(building_positions))
        
        # Color variation for buildings
        colors = [random.choice([HOLO_CYAN, HOLO_MAGENTA, HOLO_BLUE]) for _ in range(len(building_positions))]
        
        fig.add_trace(go.Scatter3d(
            x=x_buildings,
            y=y_buildings,
            z=z_buildings,
            mode='markers',
            marker=dict(
                size=8,
                color=colors,
                symbol='square',
                line=dict(color='white', width=1),
                opacity=0.7
            ),
            name='Buildings',
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # 3. PARKS (GREEN AREAS)
    park_positions = np.argwhere(grid == 2)
    if len(park_positions) > 0:
        x_parks = park_positions[:, 1]
        y_parks = park_positions[:, 0]
        z_parks = np.ones(len(park_positions)) * 0.5
        
        fig.add_trace(go.Scatter3d(
            x=x_parks,
            y=y_parks,
            z=z_parks,
            mode='markers',
            marker=dict(
                size=6,
                color=HOLO_GREEN,
                symbol='square',
                opacity=0.4
            ),
            name='Parks',
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # 4. AGENT MARKERS & PATHS
    if st.session_state.show_agents and agents:
        for agent in agents:
            # Agent position
            ax, ay = agent.position[1], agent.position[0]  # col, row
            
            # Agent marker (large glowing diamond)
            color = HOLO_CYAN if agent.state == AgentState.MOVING else HOLO_PINK
            
            fig.add_trace(go.Scatter3d(
                x=[ax],
                y=[ay],
                z=[15],  # Elevated for visibility
                mode='markers+text',
                marker=dict(
                    size=15,
                    color=color,
                    symbol='diamond',
                    line=dict(color='white', width=2),
                    opacity=1.0
                ),
                text=[agent.name],
                textposition='top center',
                textfont=dict(color=color, size=10),
                name=agent.name,
                hovertext=f"{agent.name}<br>Energy: {agent.get_energy_percent():.1f}%<br>State: {agent.state.name}",
                hoverinfo='text'
            ))
            
            # Agent path (glowing line)
            if st.session_state.show_paths and agent.path and len(agent.path) > 1:
                path_x = [p[1] for p in agent.path]  # col
                path_y = [p[0] for p in agent.path]  # row
                path_z = [10] * len(agent.path)  # Elevated path
                
                path_color = HOLO_GREEN if agent.state == AgentState.MOVING else HOLO_BLUE
                
                fig.add_trace(go.Scatter3d(
                    x=path_x,
                    y=path_y,
                    z=path_z,
                    mode='lines',
                    line=dict(color=path_color, width=4),
                    name=f'{agent.name} Path',
                    showlegend=False,
                    hoverinfo='skip'
                ))
    
    # 5. CAMERA SETTINGS WITH ROTATION
    camera_distance = 120
    angle = st.session_state.camera_rotation
    
    camera = dict(
        eye=dict(
            x=np.cos(angle) * 1.5,
            y=np.sin(angle) * 1.5,
            z=1.2
        ),
        center=dict(x=0.5, y=0.5, z=0.3),
        up=dict(x=0, y=0, z=1)
    )
    
    # Layout
    fig.update_layout(
        scene=dict(
            xaxis=dict(
                showgrid=False,
                showticklabels=False,
                showbackground=False,
                range=[0, width]
            ),
            yaxis=dict(
                showgrid=False,
                showticklabels=False,
                showbackground=False,
                range=[0, height]
            ),
            zaxis=dict(
                showgrid=False,
                showticklabels=False,
                showbackground=False,
                range=[0, 40]
            ),
            bgcolor=DARK_BG,
            camera=camera,
            aspectmode='cube'
        ),
        paper_bgcolor=DARK_BG,
        plot_bgcolor=DARK_BG,
        height=700,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
        hovermode='closest'
    )
    
    return fig

def update_simulation_step():
    """Execute one simulation step"""
    if st.session_state.simulation and st.session_state.is_running:
        st.session_state.simulation.step()
        st.session_state.current_tick += 1
        
        # Update camera rotation
        st.session_state.camera_rotation += 0.01
        
        # Check for events
        for agent in st.session_state.agents:
            if agent.is_at_goal():
                if random.random() < 0.1:  # 10% chance to log
                    log_event("SUCCESS", f"{agent.name} reached goal")
            
            if agent.energy < 20 and random.random() < 0.05:
                log_event("WARNING", f"{agent.name} low energy ({agent.energy:.1f}%)")

# -------------------------
# LOGIN PAGE
# -------------------------
def show_login_page():
    st.markdown("""
    <div class="login-container" style="max-width: 600px; margin: 150px auto;">
        <div class="logo-text">NEXUS</div>
        <p style="text-align: center; color: #00ffff; font-size: 1.2rem; margin-bottom: 30px;">
            HOLOGRAPHIC CITY NAVIGATION SYSTEM
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("<h3 style='text-align: center;'>SYSTEM LOGIN</h3>", unsafe_allow_html=True)
        username = st.text_input("Username", placeholder="Enter username")
        password = st.text_input("Password", type="password", placeholder="Enter password")
        
        col_a, col_b, col_c = st.columns([1, 2, 1])
        with col_b:
            if st.button("🚀 INITIALIZE SYSTEM", use_container_width=True):
                if username and password:  # Any credentials work for demo
                    st.session_state.logged_in = True
                    log_event("SYSTEM", f"User {username} logged in")
                    st.rerun()
                else:
                    st.error("Please enter credentials")

# -------------------------
# MAIN DASHBOARD
# -------------------------
def show_dashboard():
    # Initialize simulation if not done
    if st.session_state.simulation is None:
        initialize_simulation()
    
    # TOP BAR
    col1, col2, col3, col4, col5, col6 = st.columns([2, 2, 2, 1, 2, 3])
    
    with col1:
        st.markdown(f"<h1 style='margin:0;'>⚡ NEXUS ENGINE</h1>", unsafe_allow_html=True)
    
    with col2:
        status = "🟢 ACTIVE" if st.session_state.is_running else "🔴 PAUSED"
        st.markdown(f"<h3 style='margin:0; color:{HOLO_GREEN if st.session_state.is_running else HOLO_PINK};'>{status}</h3>", unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"<h3 style='margin:0; color:{HOLO_CYAN};'>TICK: {st.session_state.current_tick}</h3>", unsafe_allow_html=True)
    
    with col4:
        if st.button("▶️" if not st.session_state.is_running else "⏸️"):
            st.session_state.is_running = not st.session_state.is_running
            if st.session_state.is_running:
                log_event("SYSTEM", "Simulation started")
            else:
                log_event("SYSTEM", "Simulation paused")
    
    with col5:
        st.session_state.speed_multiplier = st.slider(
            "Speed",
            min_value=0.5,
            max_value=5.0,
            value=st.session_state.speed_multiplier,
            step=0.5,
            label_visibility="collapsed"
        )
    
    with col6:
        # Performance mini-graph placeholder
        perf_data = [random.randint(80, 100) for _ in range(20)]
        st.line_chart(perf_data, height=60)
    
    st.markdown("---")
    
    # MAIN LAYOUT
    left_col, center_col, right_col = st.columns([1, 3, 1])
    
    # LEFT PANEL
    with left_col:
        st.markdown(f"<h2 style='color:{HOLO_CYAN};'>WORLD STATS</h2>", unsafe_allow_html=True)
        
        st.metric("👥 Agents", len(st.session_state.agents))
        active_agents = len([a for a in st.session_state.agents if a.state == AgentState.MOVING])
        st.metric("🔄 Active", active_agents)
        st.metric("🎯 Tick", st.session_state.current_tick)
        st.metric("🤖 Bureaucrats", len(st.session_state.agents))
        st.metric("🐦 Peethweets", 0)
        
        st.markdown("---")
        st.markdown(f"<h2 style='color:{HOLO_CYAN};'>AGENT OVERVIEW</h2>", unsafe_allow_html=True)
        
        # Agent state distribution
        state_counts = {
            "Moving": len([a for a in st.session_state.agents if a.state == AgentState.MOVING]),
            "Idle": len([a for a in st.session_state.agents if a.state == AgentState.IDLE]),
            "Charging": len([a for a in st.session_state.agents if a.state == AgentState.CHARGING]),
            "Offline": len([a for a in st.session_state.agents if a.state == AgentState.OFFLINE]),
        }
        
        # Simple donut chart representation
        for state, count in state_counts.items():
            st.write(f"{state}: {count}")
        
        # Metrics
        if st.session_state.agents:
            avg_energy = sum(a.get_energy_percent() for a in st.session_state.agents) / len(st.session_state.agents)
            st.metric("⚡ Average Energy", f"{avg_energy:.1f}%")
            st.metric("✅ AI Success Rate", "94.3%")
        
        st.markdown("---")
        st.markdown(f"<h2 style='color:{HOLO_CYAN};'>LAYER CONTROLS</h2>", unsafe_allow_html=True)
        
        st.session_state.show_grid = st.checkbox("🗺️ Show Grid", value=st.session_state.show_grid)
        st.session_state.show_agents = st.checkbox("👤 Show Agents", value=st.session_state.show_agents)
        st.session_state.show_paths = st.checkbox("🛤️ Show Paths", value=st.session_state.show_paths)
    
    # CENTER - 3D HOLO-MAP
    with center_col:
        st.markdown(f"<h2 style='color:{HOLO_CYAN}; text-align:center;'>HOLO-MAP</h2>", unsafe_allow_html=True)
        
        # Create and display 3D visualization
        fig = create_3d_city_visualization()
        st.plotly_chart(fig, use_container_width=True, key=f"3d_plot_{st.session_state.current_tick}")
    
    # RIGHT PANEL
    with right_col:
        st.markdown(f"<h2 style='color:{HOLO_CYAN};'>LIVE EVENT FEED</h2>", unsafe_allow_html=True)
        
        # Event feed
        event_container = st.container(height=300)
        with event_container:
            for event in st.session_state.events[:25]:
                color = HOLO_CYAN
                if event['type'] == 'WARNING':
                    color = HOLO_PINK
                elif event['type'] == 'SUCCESS':
                    color = HOLO_GREEN
                elif event['type'] == 'AI':
                    color = HOLO_PURPLE
                
                st.markdown(f"""
                <div class="event-box">
                    <span class="event-time">[{event['time']}]</span>
                    <span class="event-msg" style="color:{color};">{event['message']}</span>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown(f"<h2 style='color:{HOLO_CYAN};'>AGENT INSPECTOR</h2>", unsafe_allow_html=True)
        
        if st.session_state.agents:
            agent_names = [a.name for a in st.session_state.agents]
            selected_name = st.selectbox("Select Agent", agent_names)
            
            selected_agent = next((a for a in st.session_state.agents if a.name == selected_name), None)
            
            if selected_agent:
                st.markdown("🧠 **BRAIN STATUS**")
                st.write(f"**ID:** {selected_agent.id}")
                st.write(f"**State:** {selected_agent.state.name}")
                st.write(f"**Energy:** {selected_agent.get_energy_percent():.1f}%")
                
                if selected_agent.path:
                    st.write(f"**Path ID:** {selected_agent.id[:6]}")
                    st.write(f"**Path Type:** A*")
                    st.write(f"**Path Length:** {len(selected_agent.path)} units")
                    st.write(f"**Path Progress:** {selected_agent.path_index}/{len(selected_agent.path)}")
                
                st.write(f"**Speed:** {selected_agent.speed:.2f}")
                st.write(f"**Perf Speed:** {selected_agent.speed * 1.2:.2f}")
    
    # AUTO-REFRESH
    if st.session_state.is_running:
        update_simulation_step()
        time.sleep(0.1 / st.session_state.speed_multiplier)  # Adjust for speed
        st.rerun()

# -------------------------
# MAIN APP LOGIC
# -------------------------
if not st.session_state.logged_in:
    show_login_page()
else:
    show_dashboard()
