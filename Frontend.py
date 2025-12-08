"""
NEXUS ENGINE - Holographic Dashboard
Replicating the advanced holographic interface with moving agents
Run: streamlit run nexus_dashboard.py
"""

import streamlit as st
import plotly.graph_objects as go
import numpy as np
from pathlib import Path
import sys
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent))

from nexus_engine import NexusEngine
from core.agent import AgentState

# CONFIG
st.set_page_config(page_title="NEXUS ENGINE", layout="wide", page_icon="⚡")

# Holographic Colors
HOLO_CYAN = "#00ffff"
HOLO_MAGENTA = "#ff00ff"
HOLO_BLUE = "#0080ff"
HOLO_GREEN = "#00ff80"
HOLO_PINK = "#ff0080"
DARK_BG = "#0a0a15"

# Advanced CSS - Holographic Style
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@300;500;700&display=swap');

.stApp {{
    background: linear-gradient(135deg, #0a0a15 0%, #0f0f20 50%, #1a1a2e 100%);
    color: {HOLO_CYAN};
    font-family: 'Rajdhani', sans-serif;
}}

header, footer, #MainMenu {{ visibility: hidden; }}
.block-container {{ padding: 0.5rem 1rem !important; max-width: 100% !important; }}

/* TOP BAR */
.top-bar {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 12px 20px;
    background: rgba(0, 255, 255, 0.05);
    border: 1px solid {HOLO_CYAN};
    border-radius: 8px;
    margin-bottom: 15px;
    box-shadow: 0 0 20px rgba(0, 255, 255, 0.3);
}}

.logo {{ 
    font-size: 20px; 
    font-weight: 700; 
    color: {HOLO_CYAN}; 
    letter-spacing: 3px;
    text-shadow: 0 0 10px {HOLO_CYAN};
}}

/* PANELS */
.holo-panel {{
    background: rgba(0, 20, 40, 0.6);
    border: 1px solid {HOLO_CYAN};
    border-radius: 8px;
    padding: 15px;
    margin-bottom: 12px;
    box-shadow: 0 0 15px rgba(0, 255, 255, 0.2);
    backdrop-filter: blur(10px);
}}

.panel-header {{
    font-size: 12px;
    font-weight: 700;
    color: {HOLO_CYAN};
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 12px;
    padding-bottom: 6px;
    border-bottom: 1px solid {HOLO_CYAN};
}}

.stat-item {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 0;
    border-bottom: 1px solid rgba(0, 255, 255, 0.1);
}}

.stat-label {{ color: rgba(0, 255, 255, 0.7); font-size: 12px; }}
.stat-value {{ color: {HOLO_CYAN}; font-size: 16px; font-weight: 700; }}

/* EVENT FEED */
.event-feed {{
    height: 250px;
    overflow-y: auto;
    font-family: 'Courier New', monospace;
    font-size: 10px;
    color: {HOLO_GREEN};
    background: rgba(0, 0, 0, 0.4);
    padding: 10px;
    border-radius: 6px;
}}

.event-line {{
    padding: 4px 6px;
    margin-bottom: 2px;
    border-left: 2px solid {HOLO_GREEN};
    opacity: 0.8;
}}

/* AGENT INSPECTOR */
.inspector {{
    text-align: center;
    padding: 20px;
}}

.brain-icon {{
    width: 100px;
    height: 100px;
    margin: 0 auto 15px;
    background: radial-gradient(circle, {HOLO_CYAN} 0%, transparent 70%);
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 50px;
    animation: pulse 2s infinite;
}}

@keyframes pulse {{
    0%, 100% {{ opacity: 1; box-shadow: 0 0 20px {HOLO_CYAN}; }}
    50% {{ opacity: 0.7; box-shadow: 0 0 40px {HOLO_CYAN}; }}
}}

.inspector-stat {{
    display: flex;
    justify-content: space-between;
    padding: 6px 0;
    border-bottom: 1px solid rgba(0, 255, 255, 0.2);
    font-size: 11px;
}}

/* BUTTONS */
.stButton > button {{
    background: rgba(0, 255, 255, 0.1) !important;
    color: {HOLO_CYAN} !important;
    border: 2px solid {HOLO_CYAN} !important;
    border-radius: 6px !important;
    padding: 10px 20px !important;
    font-weight: 700 !important;
    font-size: 12px !important;
    transition: all 0.3s !important;
}}

.stButton > button:hover {{
    background: rgba(0, 255, 255, 0.2) !important;
    box-shadow: 0 0 20px {HOLO_CYAN} !important;
}}

/* SCROLLBAR */
::-webkit-scrollbar {{ width: 6px; }}
::-webkit-scrollbar-track {{ background: rgba(0, 0, 0, 0.3); }}
::-webkit-scrollbar-thumb {{ background: {HOLO_CYAN}; border-radius: 3px; }}

/* TOGGLE */
.stCheckbox label {{ color: {HOLO_CYAN} !important; font-weight: 600 !important; }}

/* SLIDER */
.stSlider label {{ color: {HOLO_CYAN} !important; font-weight: 600 !important; }}
</style>
""", unsafe_allow_html=True)

# SESSION STATE
if 'engine' not in st.session_state:
    st.session_state.engine = None
    st.session_state.is_running = False
    st.session_state.logged_in = False
    st.session_state.username = ''
    st.session_state.camera_angle = 45
    st.session_state.speed = 1.0
    st.session_state.events = []
    st.session_state.selected_agent = None

def add_event(message):
    """Add event to feed"""
    timestamp = datetime.now().strftime('[%H:%M:%S]')
    st.session_state.events.insert(0, f"{timestamp} {message}")
    st.session_state.events = st.session_state.events[:50]

# LOGIN
def login_page():
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        st.markdown("<br><br><br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style='text-align: center;'>
            <div style='font-size: 80px; color: {HOLO_CYAN}; text-shadow: 0 0 30px {HOLO_CYAN};'>⚡</div>
            <div style='font-size: 40px; font-weight: 700; color: {HOLO_CYAN}; 
                        letter-spacing: 5px; margin: 20px 0;'>
                NEXUS ENGINE
            </div>
            <div style='color: rgba(0, 255, 255, 0.7); font-size: 14px; letter-spacing: 3px;'>
                HOLOGRAPHIC AI SIMULATION SYSTEM
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br><br>", unsafe_allow_html=True)
        username = st.text_input("OPERATOR ID", key="user")
        password = st.text_input("ACCESS CODE", type="password", key="pass")
        
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("⚡ INITIALIZE SYSTEM", use_container_width=True):
            if username and password:
                st.session_state.logged_in = True
                st.session_state.username = username
                with st.spinner("Initializing..."):
                    st.session_state.engine = NexusEngine(grid_size=80, num_agents=15)
                    add_event("System initialized successfully")
                st.rerun()

# CREATE HOLOGRAPHIC 3D MAP
def create_holographic_map():
    """Create holographic 3D visualization with VISIBLE MOVING AGENTS"""
    engine = st.session_state.engine
    if not engine or not engine.city or not engine.simulation:
        return go.Figure()
    
    city = engine.city
    agents = engine.simulation.agents
    fig = go.Figure()
    grid_size = city.width
    
    # 1. HOLOGRAPHIC GRID (more visible)
    grid_lines = np.linspace(0, grid_size, 25)
    for i, pos in enumerate(grid_lines):
        opacity = 0.3 if i % 5 == 0 else 0.15
        # X lines
        fig.add_trace(go.Scatter3d(
            x=[pos, pos], y=[0, grid_size], z=[0, 0],
            mode='lines',
            line=dict(color=HOLO_CYAN, width=2 if i % 5 == 0 else 1),
            opacity=opacity,
            showlegend=False,
            hoverinfo='skip'
        ))
        # Y lines
        fig.add_trace(go.Scatter3d(
            x=[0, grid_size], y=[pos, pos], z=[0, 0],
            mode='lines',
            line=dict(color=HOLO_CYAN, width=2 if i % 5 == 0 else 1),
            opacity=opacity,
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # 2. BUILDINGS (holographic style)
    building_positions = city.find_tiles(1)
    np.random.seed(42)
    
    for i, (row, col) in enumerate(building_positions[::4]):
        if i > 80:
            break
        
        height = np.random.uniform(12, 30)
        width = 2.0
        
        # Building vertices
        x = [col-width/2, col+width/2, col+width/2, col-width/2,
             col-width/2, col+width/2, col+width/2, col-width/2]
        y = [row-width/2, row-width/2, row+width/2, row+width/2,
             row-width/2, row-width/2, row+width/2, row+width/2]
        z = [0, 0, 0, 0, height, height, height, height]
        
        # Color based on height
        if height > 24:
            color = HOLO_CYAN
        elif height > 18:
            color = HOLO_BLUE
        else:
            color = HOLO_MAGENTA
        
        # Building mesh
        fig.add_trace(go.Mesh3d(
            x=x, y=y, z=z,
            i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
            j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
            k=[0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
            color=color,
            opacity=0.4,
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # Glowing edges
        fig.add_trace(go.Scatter3d(
            x=[col-width/2, col+width/2, col+width/2, col-width/2, col-width/2],
            y=[row-width/2, row-width/2, row+width/2, row+width/2, row-width/2],
            z=[height, height, height, height, height],
            mode='lines',
            line=dict(color=color, width=3),
            opacity=0.8,
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # 3. AGENT PATHS (highly visible trails)
    for agent in agents:
        if agent.path and len(agent.path) > 1:
            path_x = [p[1] for p in agent.path]
            path_y = [p[0] for p in agent.path]
            path_z = [3.0] * len(agent.path)
            
            # Bright colored paths
            if agent.state == AgentState.MOVING:
                color = HOLO_GREEN
                width = 5
            else:
                color = HOLO_CYAN
                width = 3
            
            fig.add_trace(go.Scatter3d(
                x=path_x, y=path_y, z=path_z,
                mode='lines',
                line=dict(color=color, width=width),
                opacity=0.7,
                showlegend=False,
                hoverinfo='skip'
            ))
    
    # 4. AGENTS - EXTRA LARGE AND VISIBLE
    agent_x = [a.position[1] for a in agents]
    agent_y = [a.position[0] for a in agents]
    agent_z = [4.0] * len(agents)  # Higher elevation
    
    agent_colors = []
    agent_sizes = []
    for a in agents:
        if a.state == AgentState.MOVING:
            agent_colors.append(HOLO_GREEN)
            agent_sizes.append(25)  # EXTRA LARGE
        elif a.state == AgentState.CHARGING:
            agent_colors.append(HOLO_PINK)
            agent_sizes.append(20)
        else:
            agent_colors.append(HOLO_CYAN)
            agent_sizes.append(18)
    
    agent_texts = [f"<b>{a.name}</b><br>Status: {a.state.name}<br>Energy: {a.energy:.0f}%<br>Pos: ({a.position[0]:.0f}, {a.position[1]:.0f})" 
                   for a in agents]
    
    # AGENTS with labels
    fig.add_trace(go.Scatter3d(
        x=agent_x, y=agent_y, z=agent_z,
        mode='markers+text',
        marker=dict(
            size=agent_sizes,
            color=agent_colors,
            symbol='diamond',
            line=dict(color='white', width=3),
            opacity=1.0
        ),
        text=[a.name.split('-')[1] for a in agents],  # Show ID number
        textposition='top center',
        textfont=dict(size=10, color='white', family='monospace'),
        hovertext=agent_texts,
        hovertemplate='%{hovertext}<extra></extra>',
        showlegend=False
    ))
    
    # 5. CAMERA
    angle = st.session_state.camera_angle
    eye_x = 1.8 * np.cos(np.radians(angle))
    eye_y = 1.8 * np.sin(np.radians(angle))
    
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False, range=[0, grid_size]),
            yaxis=dict(visible=False, range=[0, grid_size]),
            zaxis=dict(visible=False, range=[0, 40]),
            bgcolor=DARK_BG,
            camera=dict(
                eye=dict(x=eye_x, y=eye_y, z=1.0),
                center=dict(x=0, y=0, z=0.2),
                up=dict(x=0, y=0, z=1)
            ),
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=0.5)
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=750,
        showlegend=False,
        margin=dict(l=0, r=0, t=0, b=0),
        hovermode='closest'
    )
    
    return fig

# DASHBOARD
def dashboard():
    engine = st.session_state.engine
    
    # Run simulation (CRITICAL FOR MOVEMENT)
    if st.session_state.is_running and engine:
        for _ in range(int(st.session_state.speed)):
            engine.step()
        st.session_state.camera_angle = (st.session_state.camera_angle + 0.5) % 360
    
    stats = engine.get_statistics() if engine else {}
    
    # TOP BAR
    status = "🟢 ACTIVE" if st.session_state.is_running else "🔵 STANDBY"
    st.markdown(f"""
    <div class="top-bar">
        <div>
            <span class="logo">⚡ NEXUS ENGINE</span>
            <span style="margin-left: 20px; font-size: 12px; color: rgba(0,255,255,0.7);">
                {status} | TICK: {stats.get('tick', 0):,}
            </span>
        </div>
        <div style="font-size: 14px;">OPERATOR: <b>{st.session_state.username.upper()}</b></div>
    </div>
    """, unsafe_allow_html=True)
    
    # CONTROLS
    col1, col2, col3, col4 = st.columns([1, 1, 1, 6])
    with col1:
        if st.button("⏸️ PAUSE" if st.session_state.is_running else "▶️ PLAY"):
            st.session_state.is_running = not st.session_state.is_running
            add_event(f"Simulation {'paused' if not st.session_state.is_running else 'resumed'}")
            st.rerun()
    with col2:
        if st.button("🔄 RESET"):
            engine.reset()
            add_event("System reset")
            st.rerun()
    with col3:
        if st.button("🎯 GOALS"):
            engine.assign_random_goals()
            add_event("New goals assigned to all agents")
    with col4:
        st.session_state.speed = st.slider("Speed", 0.5, 5.0, st.session_state.speed, 0.5, 
                                           label_visibility="collapsed")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # MAIN LAYOUT
    left, center, right = st.columns([2, 6, 2])
    
    # LEFT PANEL
    with left:
        # WORLD STATS
        st.markdown(f"""
        <div class="holo-panel">
            <div class="panel-header">🌍 WORLD STATS</div>
            <div class="stat-item">
                <span class="stat-label">🤖 Agents</span>
                <span class="stat-value">{len(engine.simulation.agents) if engine else 0}</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">⚡ Active</span>
                <span class="stat-value">{stats.get('agents', {}).get('states', {}).get('moving', 0)}</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">🎯 Goals</span>
                <span class="stat-value">{stats.get('goals_completed', 0)}</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">🔄 Reroutes</span>
                <span class="stat-value">{stats.get('replans', 0)}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # AGENT OVERVIEW
        st.markdown(f"""
        <div class="holo-panel">
            <div class="panel-header">📊 AGENT OVERVIEW</div>
            <div style="text-align: center; padding: 20px;">
                <div style="font-size: 40px; color: {HOLO_GREEN}; font-weight: 900;">
                    {stats.get('successful_paths', 0) / max(stats.get('total_paths_planned', 1), 1) * 100:.0f}%
                </div>
                <div style="font-size: 12px; color: rgba(0,255,255,0.7);">AI SUCCESS RATE</div>
                <br>
                <div style="font-size: 30px; color: {HOLO_CYAN}; font-weight: 900;">
                    {stats.get('agents', {}).get('avg_energy', 0):.0f}%
                </div>
                <div style="font-size: 12px; color: rgba(0,255,255,0.7);">AVG ENERGY</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # CENTER - 3D MAP
    with center:
        st.markdown('<div class="holo-panel" style="height: 800px;">', unsafe_allow_html=True)
        st.markdown('<div class="panel-header">🗺️ HOLO-MAP</div>', unsafe_allow_html=True)
        
        if engine:
            fig = create_holographic_map()
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # RIGHT PANEL
    with right:
        # LIVE EVENT FEED
        st.markdown(f"""
        <div class="holo-panel">
            <div class="panel-header">📡 LIVE EVENT FEED</div>
            <div class="event-feed">
        """, unsafe_allow_html=True)
        
        for event in st.session_state.events[:25]:
            st.markdown(f'<div class="event-line">{event}</div>', unsafe_allow_html=True)
        
        st.markdown('</div></div>', unsafe_allow_html=True)
        
        # AGENT INSPECTOR
        st.markdown(f"""
        <div class="holo-panel">
            <div class="panel-header">🔍 AGENT INSPECTOR</div>
            <div class="inspector">
                <div class="brain-icon">🧠</div>
        """, unsafe_allow_html=True)
        
        if engine and engine.simulation.agents:
            agent = engine.simulation.agents[0]
            st.markdown(f"""
                <div class="inspector-stat">
                    <span style="color: rgba(0,255,255,0.7);">Agent ID:</span>
                    <span style="color: {HOLO_CYAN}; font-weight: 700;">{agent.name}</span>
                </div>
                <div class="inspector-stat">
                    <span style="color: rgba(0,255,255,0.7);">Status:</span>
                    <span style="color: {HOLO_GREEN};">{agent.state.name}</span>
                </div>
                <div class="inspector-stat">
                    <span style="color: rgba(0,255,255,0.7);">Energy:</span>
                    <span style="color: {HOLO_CYAN}; font-weight: 700;">{agent.energy:.1f}%</span>
                </div>
                <div class="inspector-stat">
                    <span style="color: rgba(0,255,255,0.7);">Position:</span>
                    <span style="color: {HOLO_CYAN};">({agent.position[0]:.0f}, {agent.position[1]:.0f})</span>
                </div>
                <div class="inspector-stat">
                    <span style="color: rgba(0,255,255,0.7);">Path Length:</span>
                    <span style="color: {HOLO_CYAN};">{len(agent.path)} steps</span>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div></div>', unsafe_allow_html=True)
    
    # Auto-refresh for SMOOTH MOVEMENT
    if st.session_state.is_running:
        import time
        time.sleep(0.08)  # 12 FPS for smooth visible movement
        st.rerun()

# MAIN
if not st.session_state.logged_in:
    login_page()
else:
    dashboard()