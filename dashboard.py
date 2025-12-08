"""
NEXUS ENGINE — Professional Streamlit Dashboard
Multi-Agent AI Simulation Platform
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import random
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Import your modules
try:
    from core.city import City
    from core.graph import CityGraph
    from core.simulation import Simulation, SimulationEvent
    from core.agent import Agent, AgentState
    from ai.search import SearchEngine
    from ai.logic_engine import LogicEngine, AgentRules
except ImportError:
    pass

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
# COLOR SCHEME
# -------------------------
NEON_CYAN = "#00d9ff"
NEON_MAGENTA = "#ff00ff"
NEON_BLUE = "#0080ff"

# -------------------------
# CYBERPUNK CSS
# -------------------------
st.markdown(
    f"""
    <style>
    /* GLOBAL BACKGROUND */
    .stApp {{
        background: linear-gradient(180deg, #0a0e27 0%, #0f1419 50%, #1a0b2e 100%);
        color: #d0e8ff;
        font-family: 'Courier New', 'Consolas', monospace;
    }}
    
    /* HIDE DEFAULT UI */
    header, footer, #MainMenu {{ visibility: hidden; }}
    
    .block-container {{
        padding-top: 1rem !important;
        padding-left: 1.5rem !important;
        padding-right: 1.5rem !important;
        max-width: 100% !important;
    }}
    
    /* TOP NAVIGATION BAR */
    .nexus-topbar {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 12px 20px;
        background: rgba(10, 14, 39, 0.85);
        border: 1px solid rgba(0, 217, 255, 0.3);
        border-radius: 6px;
        margin-bottom: 15px;
        box-shadow: 0 0 20px rgba(0, 217, 255, 0.1);
        backdrop-filter: blur(10px);
    }}
    
    .nexus-logo {{
        font-size: 20px;
        font-weight: 900;
        letter-spacing: 3px;
        color: {NEON_CYAN};
        text-shadow: 0 0 10px {NEON_CYAN};
    }}
    
    .nexus-nav {{
        display: flex;
        gap: 15px;
    }}
    
    .nav-item {{
        color: #8b9dc3;
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 1px;
        padding: 6px 12px;
        border-radius: 4px;
        transition: 0.3s;
        border: 1px solid transparent;
        cursor: pointer;
    }}
    
    .nav-item.active {{
        color: #000;
        background: {NEON_CYAN};
        font-weight: bold;
        box-shadow: 0 0 12px {NEON_CYAN};
    }}

    .nav-item:hover {{
        border: 1px solid {NEON_CYAN};
        color: {NEON_CYAN};
    }}

    /* GLASS CARDS */
    .card {{
        background: rgba(10, 14, 39, 0.6);
        border: 1px solid rgba(0, 217, 255, 0.3);
        border-radius: 6px;
        padding: 12px;
        margin-bottom: 12px;
        box-shadow: 0 0 15px rgba(0, 217, 255, 0.1);
    }}
    
    .panel-title {{
        font-size: 11px;
        color: {NEON_CYAN};
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 10px;
        border-bottom: 1px solid rgba(0, 217, 255, 0.2);
        padding-bottom: 5px;
        font-weight: bold;
    }}

    /* METRICS */
    .metric-big {{
        font-size: 36px;
        font-weight: 900;
        color: {NEON_CYAN};
        text-shadow: 0 0 15px rgba(0, 217, 255, 0.4);
        line-height: 1;
    }}
    
    .metric-sub {{
        font-size: 9px;
        color: #5a6c8f;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-top: 5px;
    }}

    /* LOGS */
    .log-panel {{
        height: 280px;
        overflow-y: auto;
        font-family: 'Consolas', monospace;
        font-size: 10px;
        color: #8b9dc3;
        background: rgba(0,0,0,0.3);
        padding: 10px;
        border-radius: 4px;
        border: 1px solid rgba(0, 217, 255, 0.1);
    }}
    
    .log-line {{
        border-bottom: 1px solid rgba(255,255,255,0.05);
        padding: 4px 0;
        font-family: 'Courier New', monospace;
    }}
    
    /* SCROLLBAR */
    ::-webkit-scrollbar {{ width: 6px; }}
    ::-webkit-scrollbar-track {{ background: rgba(0,0,0,0.2); }}
    ::-webkit-scrollbar-thumb {{ 
        background: {NEON_CYAN}; 
        border-radius: 3px; 
    }}
    
    /* BUTTONS */
    .stButton>button {{
        background: linear-gradient(90deg, {NEON_CYAN} 0%, #7b2ff7 100%) !important;
        color: white !important;
        border: 1px solid {NEON_CYAN} !important;
        border-radius: 4px !important;
        padding: 8px 20px !important;
        font-weight: bold !important;
        font-family: 'Courier New', monospace !important;
        text-transform: uppercase !important;
        font-size: 12px !important;
        letter-spacing: 1px !important;
        box-shadow: 0 0 15px rgba(0, 217, 255, 0.3) !important;
        transition: all 0.3s !important;
    }}
    
    .stButton>button:hover {{
        box-shadow: 0 0 25px rgba(0, 217, 255, 0.6) !important;
        transform: translateY(-2px) !important;
    }}
    
    /* TEXT INPUTS */
    .stTextInput input {{
        background: rgba(10, 14, 39, 0.8) !important;
        border: 1px solid rgba(0, 217, 255, 0.3) !important;
        color: {NEON_CYAN} !important;
        border-radius: 4px !important;
        font-family: 'Courier New', monospace !important;
    }}
    
    .stTextInput input:focus {{
        border: 1px solid {NEON_CYAN} !important;
        box-shadow: 0 0 15px rgba(0, 217, 255, 0.3) !important;
    }}
    
    .stTextInput label {{
        color: {NEON_CYAN} !important;
        font-weight: bold !important;
        letter-spacing: 2px !important;
        font-size: 11px !important;
    }}
    
    /* TABS */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 10px;
        background: transparent !important;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        background: rgba(10, 14, 39, 0.8) !important;
        border: 1px solid rgba(0, 217, 255, 0.3) !important;
        color: {NEON_CYAN} !important;
        border-radius: 4px !important;
        padding: 10px 20px !important;
        font-family: 'Courier New', monospace !important;
    }}
    
    .stTabs [aria-selected="true"] {{
        background: linear-gradient(90deg, rgba(0, 217, 255, 0.2), rgba(123, 47, 247, 0.2)) !important;
        border: 1px solid {NEON_CYAN} !important;
        box-shadow: 0 0 15px rgba(0, 217, 255, 0.3) !important;
    }}
    
    /* METRICS FROM STREAMLIT */
    [data-testid="stMetricValue"] {{
        color: {NEON_CYAN} !important;
        font-size: 1.8rem !important;
        font-weight: bold !important;
    }}
    
    [data-testid="stMetricLabel"] {{
        color: #5a6c8f !important;
        font-size: 0.7rem !important;
        text-transform: uppercase !important;
    }}
    
    /* PLOTLY FIX */
    .js-plotly-plot {{ height: 100% !important; }}
    
    div[data-testid="column"] {{
        padding: 0 0.5rem !important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# SESSION STATE INIT
# -------------------------
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = ''
    st.session_state.simulation = None
    st.session_state.city = None
    st.session_state.graph = None
    st.session_state.is_running = False
    st.session_state.tick = 14502
    st.session_state.current_page = 'dashboard'
    st.session_state.events = []

# -------------------------
# LOGIN PAGE
# -------------------------
def login_page():
    col1, col2, col3 = st.columns([1, 1.5, 1])
    
    with col2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        
        # Logo and Title
        st.markdown(f"""
        <div style='text-align: center; margin-bottom: 50px;'>
            <div style='font-size: 4rem; margin-bottom: 15px;'>⚡</div>
            <div style='color: {NEON_CYAN}; font-size: 2.5rem; font-weight: bold; letter-spacing: 5px; 
                        text-shadow: 0 0 30px rgba(0, 217, 255, 0.8); margin-bottom: 10px;'>
                NEXUS ENGINE
            </div>
            <div style='color: #7b2ff7; font-size: 1rem; letter-spacing: 3px;'>
                AI SIMULATION PLATFORM
            </div>
            <div style='color: #5a6c8f; font-size: 0.8rem; letter-spacing: 2px; margin-top: 10px;'>
                MULTI-AGENT CONTROL SYSTEM
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Login/Signup Tabs
        tab1, tab2 = st.tabs(["🔐 LOGIN", "📝 SIGN UP"])
        
        with tab1:
            st.markdown("<br>", unsafe_allow_html=True)
            username = st.text_input("USERNAME", key="login_user", placeholder="Enter your username")
            password = st.text_input("PASSWORD", type="password", key="login_pass", placeholder="Enter your password")
            st.markdown("<br>", unsafe_allow_html=True)
            
            col_a, col_b, col_c = st.columns([1, 2, 1])
            with col_b:
                if st.button("⚡ INITIALIZE SESSION", use_container_width=True, key="login_btn"):
                    if username and password:
                        st.session_state.logged_in = True
                        st.session_state.username = username
                        st.rerun()
                    else:
                        st.error("⚠️ Please enter both username and password")
        
        with tab2:
            st.markdown("<br>", unsafe_allow_html=True)
            new_user = st.text_input("USERNAME", key="signup_user", placeholder="Choose a username")
            new_email = st.text_input("EMAIL", key="signup_email", placeholder="Enter your email")
            new_pass = st.text_input("PASSWORD", type="password", key="signup_pass", placeholder="Create a password")
            confirm_pass = st.text_input("CONFIRM PASSWORD", type="password", key="confirm_pass", placeholder="Confirm your password")
            st.markdown("<br>", unsafe_allow_html=True)
            
            col_a, col_b, col_c = st.columns([1, 2, 1])
            with col_b:
                if st.button("✨ CREATE NEXUS ACCOUNT", use_container_width=True, key="signup_btn"):
                    if new_user and new_email and new_pass == confirm_pass:
                        st.session_state.logged_in = True
                        st.session_state.username = new_user
                        st.rerun()
                    elif new_pass != confirm_pass:
                        st.error("⚠️ Passwords do not match")
                    else:
                        st.error("⚠️ Please fill all fields")
        
        # Footer
        st.markdown(f"""
        <div style='text-align: center; margin-top: 50px; color: #5a6c8f; font-size: 0.7rem;'>
            NEXUS ENGINE v2.4.0 | SECURE CONNECTION ESTABLISHED<br>
            Powered by Advanced AI • Multi-Agent Pathfinding • Real-time Simulation
        </div>
        """, unsafe_allow_html=True)

# -------------------------
# SIMULATION INIT
# -------------------------
def init_simulation():
    if st.session_state.simulation is None:
        try:
            st.session_state.city = City()
            st.session_state.city.width = 100
            st.session_state.city.height = 100
            st.session_state.city.grid = np.random.choice([0, 1, 2], size=(100, 100), p=[0.6, 0.3, 0.1])
            st.session_state.city._base_grid = st.session_state.city.grid.copy()
            
            st.session_state.graph = CityGraph(st.session_state.city)
            st.session_state.simulation = Simulation(st.session_state.city, st.session_state.graph)
            
            for i in range(20):
                pos = (np.random.randint(0, 100), np.random.randint(0, 100))
                agent = Agent(f"Agent_{i}", pos)
                st.session_state.simulation.add_agent(agent)
        except:
            pass

# -------------------------
# EVENT GENERATOR
# -------------------------
def generate_event():
    events = [
        "Pathfinding algorithm executed successfully",
        "Agent collision avoidance triggered",
        "Waypoint navigation completed",
        "Energy optimization routine active",
        "Logic engine rule evaluation",
        "Sensor array scan complete",
        "A* search algorithm completed",
        "Graph node traversal optimized",
        "Agent state transition detected",
        "Simulation tick processed"
    ]
    agent_id = random.randint(1, 99)
    timestamp = datetime.now().strftime('[%H:%M:%S]')
    return f"{timestamp} [UNIT-{agent_id:02d}] {random.choice(events)}"

# -------------------------
# DATA GENERATORS
# -------------------------
def generate_time_series(num=80):
    values = np.abs(np.cumsum(np.random.randn(num) * 3 + 1)) + 40
    return values

def get_agent_states():
    if st.session_state.simulation:
        sim = st.session_state.simulation
        moving = sum(1 for a in sim.agents if a.state == AgentState.MOVING)
        idle = sum(1 for a in sim.agents if a.state == AgentState.IDLE)
        working = sum(1 for a in sim.agents if a.state == AgentState.WORKING)
        return {"MOVING": moving, "IDLE": idle, "WORKING": working}
    return {"MOVING": 3200, "IDLE": 1200, "WORKING": 600}

# -------------------------
# 3D CITY VISUALIZATION
# -------------------------
def create_3d_city():
    fig = go.Figure()
    
    np.random.seed(42)
    n_buildings = 40
    
    x_pos = np.random.uniform(5, 45, n_buildings)
    y_pos = np.random.uniform(5, 45, n_buildings)
    heights = np.random.uniform(10, 35, n_buildings)
    
    # Building colors
    colors = []
    for _ in range(n_buildings):
        r = np.random.random()
        if r < 0.4:
            colors.append(NEON_CYAN)
        elif r < 0.7:
            colors.append(NEON_MAGENTA)
        else:
            colors.append(NEON_BLUE)
    
    # Create 3D buildings
    for i in range(n_buildings):
        w = 2
        x = [x_pos[i]-w/2, x_pos[i]+w/2, x_pos[i]+w/2, x_pos[i]-w/2,
             x_pos[i]-w/2, x_pos[i]+w/2, x_pos[i]+w/2, x_pos[i]-w/2]
        y = [y_pos[i]-w/2, y_pos[i]-w/2, y_pos[i]+w/2, y_pos[i]+w/2,
             y_pos[i]-w/2, y_pos[i]-w/2, y_pos[i]+w/2, y_pos[i]+w/2]
        z = [0, 0, 0, 0, heights[i], heights[i], heights[i], heights[i]]
        
        fig.add_trace(go.Mesh3d(
            x=x, y=y, z=z,
            i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
            j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
            k=[0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
            color=colors[i],
            opacity=0.85,
            showlegend=False,
            hoverinfo='skip',
            lighting=dict(ambient=0.6, diffuse=0.8, specular=0.3),
        ))
    
    # Grid
    for i in range(0, 51, 5):
        fig.add_trace(go.Scatter3d(
            x=[0, 50], y=[i, i], z=[0, 0],
            mode='lines',
            line=dict(color='rgba(0, 217, 255, 0.15)', width=1),
            showlegend=False, hoverinfo='skip'
        ))
        fig.add_trace(go.Scatter3d(
            x=[i, i], y=[0, 50], z=[0, 0],
            mode='lines',
            line=dict(color='rgba(0, 217, 255, 0.15)', width=1),
            showlegend=False, hoverinfo='skip'
        ))
    
    # Agent paths
    for i in range(8):
        path_len = np.random.randint(8, 15)
        path_x = np.random.uniform(5, 45, path_len)
        path_y = np.random.uniform(5, 45, path_len)
        path_z = np.ones(path_len) * 1.5
        
        fig.add_trace(go.Scatter3d(
            x=path_x, y=path_y, z=path_z,
            mode='lines+markers',
            line=dict(color='rgba(0, 217, 255, 0.5)', width=2),
            marker=dict(size=2, color=NEON_CYAN),
            showlegend=False, hoverinfo='skip'
        ))
    
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False, range=[0, 50]),
            yaxis=dict(visible=False, range=[0, 50]),
            zaxis=dict(visible=False, range=[0, 40]),
            bgcolor='rgba(0,0,0,0)',
            camera=dict(
                eye=dict(x=1.7, y=1.7, z=0.7),
                center=dict(x=0, y=0, z=-0.15)
            ),
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=0.6)
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=600,
        showlegend=False,
        margin=dict(l=0, r=0, t=0, b=0)
    )
    
    return fig

# -------------------------
# MAIN DASHBOARD
# -------------------------
def main_dashboard():
    # Update events if running
    if st.session_state.is_running:
        if len(st.session_state.events) == 0 or random.random() < 0.3:
            st.session_state.events.insert(0, generate_event())
            st.session_state.events = st.session_state.events[:50]
        
        if st.session_state.simulation:
            try:
                st.session_state.simulation.step()
            except:
                pass
        
        st.session_state.tick += 1
    
    init_simulation()
    
    # TOP NAVIGATION BAR
    st.markdown(
        f"""
        <div class="nexus-topbar">
          <div style="display:flex; align-items:center; gap:12px;">
            <div style="font-size:24px;">⚡</div>
            <div>
              <div class="nexus-logo">NEXUS ENGINE</div>
              <div style="font-size:9px; color:#5a6c8f; letter-spacing:1px;">
                AI SIMULATION • TICK: <span style="color:{NEON_CYAN}">{st.session_state.tick:,}</span>
              </div>
            </div>
          </div>

          <div class="nexus-nav">
            <div class="nav-item {'active' if st.session_state.current_page == 'dashboard' else ''}">DASHBOARD</div>
            <div class="nav-item">AGENTS</div>
            <div class="nav-item">ANALYTICS</div>
            <div class="nav-item">CONFIG</div>
          </div>

          <div style="display:flex; gap:12px; align-items:center;">
            <div style="text-align:right;">
                <div style="font-size:9px; color:#5a6c8f;">OPERATOR</div>
                <div style="font-weight:bold; color:#fff; font-size:11px;">{st.session_state.username.upper()}</div>
            </div>
            <div style="width:32px; height:32px; background:{NEON_CYAN}; border-radius:50%; 
                        display:flex; align-items:center; justify-content:center; font-size:14px; color:#000; font-weight:bold;">
                {st.session_state.username[0].upper() if st.session_state.username else 'A'}
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # CONTROL BUTTONS
    col_btn1, col_btn2, col_btn3, col_spacer, col_logout = st.columns([1, 1, 1, 6, 1.5])
    
    with col_btn1:
        if st.button("⏸️ PAUSE" if st.session_state.is_running else "▶️ PLAY", key="play_pause", use_container_width=True):
            st.session_state.is_running = not st.session_state.is_running
            st.rerun()
    
    with col_btn2:
        if st.button("🔄 RESET", key="reset", use_container_width=True):
            st.session_state.tick = 0
            st.session_state.simulation = None
            st.session_state.events = []
            init_simulation()
            st.rerun()
    
    with col_btn3:
        if st.button("📊 EXPORT", key="export", use_container_width=True):
            st.toast("Export functionality coming soon!")
    
    with col_logout:
        if st.button("🚪 LOGOUT", key="logout", use_container_width=True):
            st.session_state.logged_in = False
            st.rerun()
    
    st.markdown("<div style='margin: 10px 0;'></div>", unsafe_allow_html=True)
    
    # MAIN LAYOUT
    left_col, center_col, right_col = st.columns([1.8, 4, 1.8], gap="small")
    
    # --- LEFT COLUMN ---
    with left_col:
        # System Metrics Card
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">SYSTEM METRICS</div>', unsafe_allow_html=True)
        
        c1, c2 = st.columns(2)
        with c1:
            total_agents = len(st.session_state.simulation.agents) if st.session_state.simulation else 5000
            st.markdown(f'''
                <div class="metric-big">{total_agents:,}</div>
                <div class="metric-sub">TOTAL AGENTS</div>
            ''', unsafe_allow_html=True)
        with c2:
            active_events = len(st.session_state.events)
            st.markdown(f'''
                <div class="metric-big">{active_events}</div>
                <div class="metric-sub">ACTIVE EVENTS</div>
            ''', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # CPU Load Chart
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">SYSTEM LOAD</div>', unsafe_allow_html=True)
        
        time_series = generate_time_series()
        fig_line = go.Figure(go.Scatter(
            y=time_series,
            mode='lines',
            line=dict(width=2, color=NEON_CYAN),
            fill='tozeroy',
            fillcolor='rgba(0, 217, 255, 0.1)'
        ))
        fig_line.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            height=130,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(visible=False),
            yaxis=dict(visible=False)
        )
        st.plotly_chart(fig_line, use_container_width=True, key="cpu_chart")
        st.markdown('</div>', unsafe_allow_html=True)

        # Agent Status Donut
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">AGENT STATUS</div>', unsafe_allow_html=True)
        
        agent_states = get_agent_states()
        fig_pie = go.Figure(data=[go.Pie(
            labels=list(agent_states.keys()),
            values=list(agent_states.values()),
            hole=0.65,
            marker=dict(colors=[NEON_CYAN, NEON_MAGENTA, NEON_BLUE]),
            textinfo='none'
        )])
        fig_pie.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            height=170,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            showlegend=False,
            annotations=[dict(
                text="ACTIVE",
                x=0.5, y=0.5,
                font_size=11,
                font_color=NEON_CYAN,
                showarrow=False
            )]
        )
        st.plotly_chart(fig_pie, use_container_width=True, key="status_pie")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Graph Stats
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">GRAPH STATISTICS</div>', unsafe_allow_html=True)
        if st.session_state.graph:
            stats = st.session_state.graph.get_stats()
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("🔵 Nodes", f"{stats['nodes']:,}")
            with col_b:
                st.metric("🔗 Edges", f"{stats['edges']:,}")
        st.markdown('</div>', unsafe_allow_html=True)

    # --- CENTER COLUMN: 3D HOLO-MAP ---
    with center_col:
        st.markdown('<div class="card" style="height: 95%;">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">HOLO-MAP // 3D CITY VISUALIZATION</div>', unsafe_allow_html=True)

        fig = create_3d_city()
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)

    # --- RIGHT COLUMN: LOGS & INSPECTOR ---
    with right_col:
        # Card 1: Event Log
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">LIVE EVENT FEED</div>', unsafe_allow_html=True)
        
        log_html = "<div class='log-panel'>"
        for event in st.session_state.events[:20]:
            log_html += f"<div class='log-line'> > {event}</div>"
        log_html += "</div>"
        
        st.markdown(log_html, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Card 2: Agent Inspector
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">SELECTED AGENT INSPECTOR</div>', unsafe_allow_html=True)
        
        # Brain SVG Graphic
        st.markdown(f"""
        <div style="text-align: center; margin: 15px 0;">
            <svg width="80" height="80" viewBox="0 0 100 100" style="opacity: 0.8;">
                <path d="M20,50 Q20,20 50,20 Q80,20 80,50 Q80,80 50,80 Q20,80 20,50" 
                      fill="none" stroke="{NEON_CYAN}" stroke-width="2"/>
                <path d="M35,40 Q50,30 65,40" fill="none" stroke="{NEON_MAGENTA}" stroke-width="2"/>
                <path d="M30,50 L70,50" fill="none" stroke="{NEON_BLUE}" stroke-width="1" stroke-dasharray="2,2"/>
                <circle cx="50" cy="50" r="4" fill="{NEON_CYAN}" />
            </svg>
        </div>
        """, unsafe_allow_html=True)
        
        # Live Agent Data
        if st.session_state.simulation and st.session_state.simulation.agents:
            agent = st.session_state.simulation.agents[0]
            st.markdown(f"""
            <div style='font-size: 0.75rem; line-height: 1.8; font-family: Consolas;'>
                <div style='display: flex; justify-content: space-between; border-bottom: 1px solid rgba(255,255,255,0.05); padding: 2px 0;'>
                    <span style='color: #5a6c8f;'>UNIT ID</span>
                    <span style='color: {NEON_CYAN}; font-weight: bold;'>{agent.name}</span>
                </div>
                <div style='display: flex; justify-content: space-between; border-bottom: 1px solid rgba(255,255,255,0.05); padding: 2px 0;'>
                    <span style='color: #5a6c8f;'>GRID POS</span>
                    <span style='color: #fff;'>{agent.position}</span>
                </div>
                <div style='display: flex; justify-content: space-between; border-bottom: 1px solid rgba(255,255,255,0.05); padding: 2px 0;'>
                    <span style='color: #5a6c8f;'>STATUS</span>
                    <span style='color: {NEON_MAGENTA}; font-weight: bold;'>{agent.state.name}</span>
                </div>
                <div style='display: flex; justify-content: space-between; border-bottom: 1px solid rgba(255,255,255,0.05); padding: 2px 0;'>
                    <span style='color: #5a6c8f;'>ENERGY</span>
                    <span style='color: {NEON_CYAN};'>{agent.energy:.1f}%</span>
                </div>
                <div style='display: flex; justify-content: space-between; padding: 2px 0;'>
                    <span style='color: #5a6c8f;'>PATH ALG</span>
                    <span style='color: #fff;'>A* SEARCH</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("No agents selected.")
            
        st.markdown('</div>', unsafe_allow_html=True)

# -------------------------
# MAIN ROUTER
# -------------------------
if not st.session_state.logged_in:
    login_page()
else:
    main_dashboard()