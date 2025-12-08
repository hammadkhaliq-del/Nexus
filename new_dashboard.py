"""
NEXUS ENGINE — Fully Integrated Dashboard
Brain (AI) + Body (Core) Working Together
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import random
from datetime import datetime
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Import NEXUS Brain + Body
from core.city import City
from core.graph import CityGraph
from core.simulation import Simulation, SimulationEvent
from core.agent import Agent, AgentState
from core.utils import euclidean_distance, manhattan_distance

# Import AI Brain
from ai.search import a_star, dijkstra, bfs, compare_algorithms
from ai.logic_engine import LogicEngine, create_simple_rule, AgentRules
from ai.explainability import ExplainabilityEngine
from ai.bayesian import BayesianNetwork

# -------------------------
# PAGE CONFIGURATION
# -------------------------
st.set_page_config(
    page_title="NEXUS ENGINE - Integrated",
    layout="wide",
    initial_sidebar_state="collapsed",
    page_icon="⚡"
)

# Color scheme
NEON_CYAN = "#00d9ff"
NEON_MAGENTA = "#ff00ff"
NEON_BLUE = "#0080ff"
NEON_GREEN = "#00ff88"

# [KEEP ALL YOUR EXISTING CSS - I'll just add new styles]
st.markdown(f"""
<style>
/* ... YOUR EXISTING CSS ... */
.stApp {{
    background: linear-gradient(180deg, #0a0e27 0%, #0f1419 50%, #1a0b2e 100%);
    color: #d0e8ff;
    font-family: 'Courier New', 'Consolas', monospace;
}}

header, footer, #MainMenu {{ visibility: hidden; }}

.block-container {{
    padding-top: 1rem !important;
    padding-left: 1.5rem !important;
    padding-right: 1.5rem !important;
    max-width: 100% !important;
}}

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
}}

.nexus-logo {{
    font-size: 20px;
    font-weight: 900;
    letter-spacing: 3px;
    color: {NEON_CYAN};
    text-shadow: 0 0 10px {NEON_CYAN};
}}

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

.metric-big {{
    font-size: 36px;
    font-weight: 900;
    color: {NEON_CYAN};
    text-shadow: 0 0 15px rgba(0, 217, 255, 0.4);
}}

.metric-sub {{
    font-size: 9px;
    color: #5a6c8f;
    letter-spacing: 2px;
    text-transform: uppercase;
}}

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
}}

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
    box-shadow: 0 0 15px rgba(0, 217, 255, 0.3) !important;
}}

/* NEW: Path visualization indicator */
.path-indicator {{
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    margin-right: 5px;
}}

.path-active {{ background: {NEON_GREEN}; box-shadow: 0 0 8px {NEON_GREEN}; }}
.path-planning {{ background: {NEON_CYAN}; box-shadow: 0 0 8px {NEON_CYAN}; }}
.path-blocked {{ background: {NEON_MAGENTA}; box-shadow: 0 0 8px {NEON_MAGENTA}; }}

</style>
""", unsafe_allow_html=True)

# -------------------------
# SESSION STATE INIT
# -------------------------
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = ''
    st.session_state.simulation = None
    st.session_state.city = None
    st.session_state.graph = None
    st.session_state.logic_engine = None
    st.session_state.explainer = None
    st.session_state.bayesian = None
    st.session_state.is_running = False
    st.session_state.tick = 0
    st.session_state.current_page = 'dashboard'
    st.session_state.events = []
    st.session_state.ai_decisions = []
    st.session_state.pathfinding_stats = {"total": 0, "successful": 0, "failed": 0}
    st.session_state.selected_agent_id = None

# -------------------------
# NEXUS BRAIN INITIALIZATION
# -------------------------
def init_nexus_brain():
    """Initialize AI components"""
    if st.session_state.logic_engine is None:
        st.session_state.logic_engine = LogicEngine()
        
        # Add intelligent rules
        low_energy_rule = AgentRules.low_energy_recharge(threshold=25.0)
        st.session_state.logic_engine.add_rule(low_energy_rule)
        
        goal_reached_rule = AgentRules.goal_reached_idle()
        st.session_state.logic_engine.add_rule(goal_reached_rule)
    
    if st.session_state.explainer is None:
        st.session_state.explainer = ExplainabilityEngine()
    
    if st.session_state.bayesian is None:
        st.session_state.bayesian = BayesianNetwork()

# -------------------------
# NEXUS BODY INITIALIZATION
# -------------------------
def init_nexus_body():
    """Initialize core simulation with real map"""
    if st.session_state.simulation is None:
        # Create realistic city map
        grid_size = 80
        grid = np.zeros((grid_size, grid_size), dtype=np.int32)
        
        # Add buildings (1 = blocked)
        for _ in range(25):
            bx, by = np.random.randint(5, grid_size-10), np.random.randint(5, grid_size-10)
            bw, bh = np.random.randint(3, 8), np.random.randint(3, 8)
            grid[by:by+bh, bx:bx+bw] = 1
        
        # Add parks (2 = walkable grass)
        for _ in range(10):
            px, py = np.random.randint(5, grid_size-8), np.random.randint(5, grid_size-8)
            pw, ph = np.random.randint(4, 7), np.random.randint(4, 7)
            grid[py:py+ph, px:px+pw] = 2
        
        # Save and load
        np.save("nexus_city_map.npy", grid)
        
        st.session_state.city = City("nexus_city_map.npy")
        st.session_state.graph = CityGraph(st.session_state.city)
        st.session_state.simulation = Simulation(
            st.session_state.city, 
            st.session_state.graph
        )
        
        # Add intelligent agents with AI-planned paths
        spawn_points = []
        for _ in range(15):
            attempts = 0
            while attempts < 100:
                pos = (np.random.randint(5, grid_size-5), np.random.randint(5, grid_size-5))
                if st.session_state.city.is_walkable(*pos) and pos not in spawn_points:
                    spawn_points.append(pos)
                    break
                attempts += 1
        
        for i, pos in enumerate(spawn_points):
            agent = Agent(f"NEXUS-{i:02d}", pos, speed=1.5, max_energy=100.0)
            
            # Give agent a goal and use A* to plan path
            goal = spawn_points[(i + 7) % len(spawn_points)]
            agent.set_goal(goal)
            
            # AI pathfinding
            try:
                result = a_star(st.session_state.graph, pos, goal)
                if result.success:
                    agent.set_path(result.path)
                    st.session_state.pathfinding_stats["successful"] += 1
                    
                    # Log AI decision
                    explanation = st.session_state.explainer.explain_path(
                        agent, result.path, result.algorithm, result.cost
                    )
                    log_event(f"[AI-BRAIN] {agent.name}: {explanation}", "info")
                else:
                    st.session_state.pathfinding_stats["failed"] += 1
                    log_event(f"[AI-BRAIN] {agent.name}: No valid path found", "warning")
            except Exception as e:
                st.session_state.pathfinding_stats["failed"] += 1
                log_event(f"[ERROR] {agent.name}: Pathfinding error", "error")
            
            st.session_state.pathfinding_stats["total"] += 1
            st.session_state.simulation.add_agent(agent)

# -------------------------
# EVENT LOGGING SYSTEM
# -------------------------
def log_event(message: str, level: str = "info"):
    """Add event to log with timestamp"""
    timestamp = datetime.now().strftime('[%H:%M:%S.%f')[:-3] + ']'
    
    level_icons = {
        "info": "ℹ️",
        "success": "✅",
        "warning": "⚠️",
        "error": "❌",
        "ai": "🧠"
    }
    
    icon = level_icons.get(level, "•")
    formatted = f"{timestamp} {icon} {message}"
    
    st.session_state.events.insert(0, formatted)
    st.session_state.events = st.session_state.events[:100]

# -------------------------
# AI DECISION LOOP
# -------------------------
def run_ai_decisions():
    """Execute AI reasoning for each agent"""
    if not st.session_state.simulation:
        return
    
    logic = st.session_state.logic_engine
    explainer = st.session_state.explainer
    
    for agent in st.session_state.simulation.agents:
        # Logic engine evaluates rules
        context = {
            "agent": agent,
            "city": st.session_state.city,
            "detected_obstacles": list(agent.known_obstacles)
        }
        
        actions = logic.forward_chain(context)
        
        # Log AI decisions
        for action in actions:
            if "recharge" in action.lower():
                log_event(f"[AI-LOGIC] {agent.name}: Low energy detected, initiating recharge protocol", "ai")
            elif "goal" in action.lower():
                log_event(f"[AI-LOGIC] {agent.name}: Goal reached, transitioning to IDLE state", "success")
        
        # Check if replanning needed (obstacle ahead)
        if agent.path and agent.path_index < len(agent.path):
            next_pos = agent.path[agent.path_index]
            if not st.session_state.city.is_walkable(*next_pos):
                # Replan with A*
                try:
                    result = a_star(st.session_state.graph, 
                                  (int(agent.position[0]), int(agent.position[1])), 
                                  agent.goal)
                    if result.success:
                        old_path = agent.path
                        agent.set_path(result.path)
                        
                        explanation = explainer.explain_replanning(
                            agent, old_path, result.path, "obstacle detected on route"
                        )
                        log_event(f"[AI-REPLAN] {explanation}", "warning")
                except:
                    pass

# -------------------------
# 2D MAP WITH AGENTS
# -------------------------
def create_2d_map():
    """Create 2D map showing agents and their paths"""
    if not st.session_state.city or not st.session_state.simulation:
        return go.Figure()
    
    city = st.session_state.city
    sim = st.session_state.simulation
    
    # Create heatmap of city
    fig = go.Figure()
    
    # Base map
    fig.add_trace(go.Heatmap(
        z=city.grid,
        colorscale=[
            [0, 'rgba(0, 50, 100, 0.3)'],      # Roads - dark blue
            [0.5, 'rgba(100, 0, 0, 0.8)'],     # Buildings - red
            [1, 'rgba(0, 100, 50, 0.4)']       # Grass - green
        ],
        showscale=False,
        hoverinfo='skip'
    ))
    
    # Agent paths
    for agent in sim.agents:
        if agent.path:
            path_x = [p[1] for p in agent.path]
            path_y = [p[0] for p in agent.path]
            
            fig.add_trace(go.Scatter(
                x=path_x, y=path_y,
                mode='lines',
                line=dict(color=NEON_CYAN, width=1, dash='dot'),
                opacity=0.4,
                showlegend=False,
                hoverinfo='skip'
            ))
    
    # Agent positions
    agent_x = [agent.position[1] for agent in sim.agents]
    agent_y = [agent.position[0] for agent in sim.agents]
    agent_colors = [NEON_GREEN if a.state == AgentState.MOVING else NEON_MAGENTA 
                    for a in sim.agents]
    agent_names = [a.name for a in sim.agents]
    
    fig.add_trace(go.Scatter(
        x=agent_x, y=agent_y,
        mode='markers',
        marker=dict(
            size=12,
            color=agent_colors,
            symbol='circle',
            line=dict(color='white', width=1)
        ),
        text=agent_names,
        hovertemplate='<b>%{text}</b><br>Position: (%{y:.1f}, %{x:.1f})<extra></extra>',
        showlegend=False
    ))
    
    fig.update_layout(
        height=600,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(visible=False, range=[-1, city.width]),
        yaxis=dict(visible=False, range=[-1, city.height], scaleanchor="x"),
        margin=dict(l=0, r=0, t=0, b=0),
        hovermode='closest'
    )
    
    return fig

# -------------------------
# AI STATISTICS
# -------------------------
def get_ai_stats():
    """Get AI system statistics"""
    if not st.session_state.simulation:
        return {}
    
    total_agents = len(st.session_state.simulation.agents)
    
    states = {
        "moving": sum(1 for a in st.session_state.simulation.agents if a.state == AgentState.MOVING),
        "idle": sum(1 for a in st.session_state.simulation.agents if a.state == AgentState.IDLE),
        "charging": sum(1 for a in st.session_state.simulation.agents if a.state == AgentState.CHARGING),
        "offline": sum(1 for a in st.session_state.simulation.agents if a.state == AgentState.OFFLINE)
    }
    
    avg_energy = sum(a.energy for a in st.session_state.simulation.agents) / total_agents if total_agents else 0
    total_distance = sum(a.distance_traveled for a in st.session_state.simulation.agents)
    
    return {
        "states": states,
        "avg_energy": avg_energy,
        "total_distance": total_distance,
        "pathfinding": st.session_state.pathfinding_stats
    }

# -------------------------
# LOGIN PAGE
# -------------------------
def login_page():
    col1, col2, col3 = st.columns([1, 1.5, 1])
    
    with col2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        
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
            <div style='color: {NEON_GREEN}; font-size: 0.9rem; letter-spacing: 2px; margin-top: 15px; font-weight: bold;'>
                🧠 BRAIN + BODY INTEGRATED
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["🔐 LOGIN", "📝 SIGN UP"])
        
        with tab1:
            st.markdown("<br>", unsafe_allow_html=True)
            username = st.text_input("USERNAME", key="login_user", placeholder="Enter your username")
            password = st.text_input("PASSWORD", type="password", key="login_pass", placeholder="Enter your password")
            st.markdown("<br>", unsafe_allow_html=True)
            
            col_a, col_b, col_c = st.columns([1, 2, 1])
            with col_b:
                if st.button("⚡ INITIALIZE NEXUS", use_container_width=True, key="login_btn"):
                    if username and password:
                        st.session_state.logged_in = True
                        st.session_state.username = username
                        init_nexus_brain()
                        init_nexus_body()
                        log_event(f"NEXUS SYSTEM INITIALIZED - Operator: {username.upper()}", "success")
                        st.rerun()
                    else:
                        st.error("⚠️ Please enter credentials")
        
        with tab2:
            st.markdown("<br>", unsafe_allow_html=True)
            new_user = st.text_input("USERNAME", key="signup_user")
            new_email = st.text_input("EMAIL", key="signup_email")
            new_pass = st.text_input("PASSWORD", type="password", key="signup_pass")
            confirm_pass = st.text_input("CONFIRM PASSWORD", type="password", key="confirm_pass")
            st.markdown("<br>", unsafe_allow_html=True)
            
            col_a, col_b, col_c = st.columns([1, 2, 1])
            with col_b:
                if st.button("✨ CREATE ACCOUNT", use_container_width=True, key="signup_btn"):
                    if new_user and new_email and new_pass == confirm_pass:
                        st.session_state.logged_in = True
                        st.session_state.username = new_user
                        init_nexus_brain()
                        init_nexus_body()
                        st.rerun()
                    else:
                        st.error("⚠️ Please check your inputs")

# -------------------------
# MAIN DASHBOARD
# -------------------------
def main_dashboard():
    # Initialize systems
    init_nexus_brain()
    init_nexus_body()
    
    # Run AI decisions and simulation
    if st.session_state.is_running:
        try:
            # AI decision making
            if st.session_state.tick % 3 == 0:  # Every 3 ticks
                run_ai_decisions()
            
            # Physics simulation
            if st.session_state.simulation:
                st.session_state.simulation.step()
            
            st.session_state.tick += 1
            
            # Generate events
            if random.random() < 0.15:
                events = [
                    "A* pathfinding optimized",
                    "Logic rule triggered",
                    "Bayesian inference updated",
                    "Agent replanning initiated",
                    "Energy management active"
                ]
                log_event(f"[SYSTEM] {random.choice(events)}", "info")
                
        except Exception as e:
            log_event(f"[ERROR] Simulation error: {str(e)}", "error")
    
    # TOP BAR
    st.markdown(f"""
    <div class="nexus-topbar">
      <div style="display:flex; align-items:center; gap:12px;">
        <div style="font-size:24px;">⚡</div>
        <div>
          <div class="nexus-logo">NEXUS ENGINE</div>
          <div style="font-size:9px; color:#5a6c8f;">
            🧠 AI-POWERED • TICK: <span style="color:{NEON_CYAN}">{st.session_state.tick:,}</span>
          </div>
        </div>
      </div>
      
      <div style="display:flex; gap:8px; align-items:center;">
        <div class="path-indicator {'path-active' if st.session_state.is_running else 'path-planning'}"></div>
        <div style="font-size:9px; color:#5a6c8f;">
          {'SIMULATION ACTIVE' if st.session_state.is_running else 'PAUSED'}
        </div>
      </div>

      <div style="display:flex; gap:12px; align-items:center;">
        <div style="text-align:right;">
            <div style="font-size:9px; color:#5a6c8f;">OPERATOR</div>
            <div style="font-weight:bold; color:#fff; font-size:11px;">{st.session_state.username.upper()}</div>
        </div>
        <div style="width:32px; height:32px; background:{NEON_CYAN}; border-radius:50%; 
                    display:flex; align-items:center; justify-content:center; font-size:14px; color:#000; font-weight:bold;">
            {st.session_state.username[0].upper() if st.session_state.username else 'N'}
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)
    
    # CONTROLS
    col1, col2, col3, col4, spacer, col_logout = st.columns([1, 1, 1, 1, 5, 1.5])
    
    with col1:
        if st.button("⏸️ PAUSE" if st.session_state.is_running else "▶️ PLAY", 
                    use_container_width=True, key="play"):
            st.session_state.is_running = not st.session_state.is_running
            log_event(f"Simulation {'STARTED' if st.session_state.is_running else 'PAUSED'}", "info")
            st.rerun()
    
    with col2:
        if st.button("🔄 RESET", use_container_width=True, key="reset"):
            st.session_state.simulation = None
            st.session_state.tick = 0
            st.session_state.events = []
            st.session_state.pathfinding_stats = {"total": 0, "successful": 0, "failed": 0}
            init_nexus_body()
            log_event("NEXUS SYSTEM RESET", "success")
            st.rerun()
    
    with col3:
        if st.button("🧠 REPLAN ALL", use_container_width=True, key="replan"):
            # Trigger AI replanning for all agents
            for agent in st.session_state.simulation.agents:
                if agent.goal:
                    result = a_star(st.session_state.graph, 
                                  (int(agent.position[0]), int(agent.position[1])), 
                                  agent.goal)
                    if result.success:
                        agent.set_path(result.path)
            log_event("AI REPLANNING COMPLETE FOR ALL AGENTS", "ai")
            st.toast("🧠 AI Replanning Complete!")
    
    with col4:
        if st.button("📊 EXPORT", use_container_width=True, key="export"):
            stats = get_ai_stats()
            st.toast(f"📊 Stats: {stats['pathfinding']['successful']}/{stats['pathfinding']['total']} successful paths")
    
    with col_logout:
        if st.button("🚪 LOGOUT", use_container_width=True):
            st.session_state.logged_in = False
            st.rerun()
    
    st.markdown("<div style='margin: 10px 0;'></div>", unsafe_allow_html=True)
    
    # MAIN LAYOUT
    left_col, center_col, right_col = st.columns([1.8, 4, 1.8], gap="small")
    
    # --- LEFT: AI STATS ---
    with left_col:
        ai_stats = get_ai_stats()
        
        # System Status
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">🧠 AI BRAIN STATUS</div>', unsafe_allow_html=True)
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f'''
                <div class="metric-big">{ai_stats["pathfinding"]["successful"]}</div>
                <div class="metric-sub">SUCCESSFUL PATHS</div>
            ''', unsafe_allow_html=True)
        with c2:
            total = ai_stats["pathfinding"]["total"]
            success_rate = (ai_stats["pathfinding"]["successful"] / total * 100) if total else 0
            st.markdown(f'''
                <div class="metric-big">{success_rate:.0f}%</div>
                <div class="metric-sub">SUCCESS RATE</div>
            ''', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Agent States
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">AGENT DISTRIBUTION</div>', unsafe_allow_html=True)
        
        states = ai_stats["states"]
        fig_pie = go.Figure(data=[go.Pie(
            labels=list(states.keys()),
            values=list(states.values()),
            hole=0.65,
            marker=dict(colors=[NEON_GREEN, NEON_CYAN, NEON_MAGENTA, '#ff4444']),
            textinfo='none'
        )])
        fig_pie.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            height=170,
            paper_bgcolor='rgba(0,0,0,0)',
            showlegend=False
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Energy & Distance
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">PERFORMANCE METRICS</div>', unsafe_allow_html=True)
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("⚡ Avg Energy", f"{ai_stats['avg_energy']:.1f}%")
        with col_b:
            st.metric("📏 Total Dist", f"{ai_stats['total_distance']:.1f}")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Graph Info
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">GRAPH NETWORK</div>', unsafe_allow_html=True)
        if st.session_state.graph:
            stats = st.session_state.graph.get_stats()
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("🔵 Nodes", f"{stats['nodes']:,}")
            with col_b:
                st.metric("🔗 Edges", f"{stats['edges']:,}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # --- CENTER: LIVE MAP ---
    with center_col:
        st.markdown('<div class="card" style="height: 95%;">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">🗺️ LIVE AGENT MAP // REAL-TIME TRACKING</div>', unsafe_allow_html=True)
        
        # Show 2D map with agents
        fig_map = create_2d_map()
        st.plotly_chart(fig_map, use_container_width=True, config={'displayModeBar': False})
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # --- RIGHT: LOGS & AGENT INSPECTOR ---
    with right_col:
        # Event Log
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">📡 LIVE EVENT FEED</div>', unsafe_allow_html=True)
        
        log_html = "<div class='log-panel'>"
        for event in st.session_state.events[:30]:
            log_html += f"<div class='log-line'>{event}</div>"
        log_html += "</div>"
        st.markdown(log_html, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Agent Inspector
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">🤖 AGENT INSPECTOR</div>', unsafe_allow_html=True)
        
        if st.session_state.simulation and st.session_state.simulation.agents:
            # Agent selector
            agent_names = [a.name for a in st.session_state.simulation.agents]
            selected = st.selectbox("Select Agent", agent_names, key="agent_select", label_visibility="collapsed")
            
            # Find selected agent
            agent = next((a for a in st.session_state.simulation.agents if a.name == selected), None)
            
            if agent:
                # Brain visualization
                st.markdown(f"""
                <div style="text-align: center; margin: 10px 0;">
                    <svg width="60" height="60" viewBox="0 0 100 100">
                        <circle cx="50" cy="50" r="40" fill="none" stroke="{NEON_CYAN}" stroke-width="2"/>
                        <path d="M30,50 Q50,30 70,50" fill="none" stroke="{NEON_MAGENTA}" stroke-width="2"/>
                        <circle cx="50" cy="50" r="4" fill="{NEON_GREEN}" />
                    </svg>
                </div>
                """, unsafe_allow_html=True)
                
                # Agent details
                st.markdown(f"""
                <div style='font-size: 0.75rem; line-height: 1.8; font-family: Consolas;'>
                    <div style='display: flex; justify-content: space-between; border-bottom: 1px solid rgba(255,255,255,0.05); padding: 2px 0;'>
                        <span style='color: #5a6c8f;'>UNIT ID</span>
                        <span style='color: {NEON_CYAN}; font-weight: bold;'>{agent.name}</span>
                    </div>
                    <div style='display: flex; justify-content: space-between; border-bottom: 1px solid rgba(255,255,255,0.05); padding: 2px 0;'>
                        <span style='color: #5a6c8f;'>POSITION</span>
                        <span style='color: #fff;'>({agent.position[0]:.1f}, {agent.position[1]:.1f})</span>
                    </div>
                    <div style='display: flex; justify-content: space-between; border-bottom: 1px solid rgba(255,255,255,0.05); padding: 2px 0;'>
                        <span style='color: #5a6c8f;'>STATE</span>
                        <span style='color: {NEON_MAGENTA}; font-weight: bold;'>{agent.state.name}</span>
                    </div>
                    <div style='display: flex; justify-content: space-between; border-bottom: 1px solid rgba(255,255,255,0.05); padding: 2px 0;'>
                        <span style='color: #5a6c8f;'>ENERGY</span>
                        <span style='color: {NEON_GREEN if agent.energy > 50 else NEON_MAGENTA};'>{agent.energy:.1f}%</span>
                    </div>
                    <div style='display: flex; justify-content: space-between; border-bottom: 1px solid rgba(255,255,255,0.05); padding: 2px 0;'>
                        <span style='color: #5a6c8f;'>GOAL</span>
                        <span style='color: #fff;'>{agent.goal if agent.goal else "None"}</span>
                    </div>
                    <div style='display: flex; justify-content: space-between; border-bottom: 1px solid rgba(255,255,255,0.05); padding: 2px 0;'>
                        <span style='color: #5a6c8f;'>PATH ALG</span>
                        <span style='color: {NEON_CYAN};'>A* SEARCH</span>
                    </div>
                    <div style='display: flex; justify-content: space-between; border-bottom: 1px solid rgba(255,255,255,0.05); padding: 2px 0;'>
                        <span style='color: #5a6c8f;'>DISTANCE</span>
                        <span style='color: #fff;'>{agent.distance_traveled:.1f}</span>
                    </div>
                    <div style='display: flex; justify-content: space-between; padding: 2px 0;'>
                        <span style='color: #5a6c8f;'>STEPS</span>
                        <span style='color: #fff;'>{agent.steps_taken}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # AI Explanation
                st.markdown("<br>", unsafe_allow_html=True)
                if agent.path and len(agent.path) > 0:
                    explanation = st.session_state.explainer.explain_action(
                        agent, "navigate", {"energy_low": agent.energy < 30}
                    )
                    st.markdown(f"""
                    <div style='background: rgba(0, 217, 255, 0.1); border: 1px solid rgba(0, 217, 255, 0.3); 
                                border-radius: 4px; padding: 8px; font-size: 0.7rem;'>
                        <div style='color: {NEON_CYAN}; font-weight: bold; margin-bottom: 4px;'>🧠 AI REASONING:</div>
                        <div style='color: #8b9dc3;'>{explanation}</div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("No agents available")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # AI Control Panel
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">⚙️ AI CONTROLS</div>', unsafe_allow_html=True)
        
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            if st.button("🎯 Auto-Goal", use_container_width=True, key="auto_goal"):
                # Assign random goals to all agents
                if st.session_state.simulation:
                    for agent in st.session_state.simulation.agents:
                        walkable = st.session_state.city.get_walkable_positions()
                        if walkable:
                            new_goal = random.choice(walkable)
                            agent.set_goal(new_goal)
                            
                            # Plan path with A*
                            result = a_star(
                                st.session_state.graph,
                                (int(agent.position[0]), int(agent.position[1])),
                                new_goal
                            )
                            if result.success:
                                agent.set_path(result.path)
                    
                    log_event("[AI-AUTO] Auto-goal assignment complete for all agents", "ai")
                    st.toast("🎯 Auto-goals assigned!")
        
        with col_c2:
            if st.button("⚡ Recharge", use_container_width=True, key="recharge_all"):
                # Recharge all agents
                if st.session_state.simulation:
                    for agent in st.session_state.simulation.agents:
                        agent.recharge_energy(50)
                    log_event("[AI-AUTO] Energy boost applied to all agents", "success")
                    st.toast("⚡ All agents recharged!")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Auto-refresh if running
    if st.session_state.is_running:
        import time
        time.sleep(0.1)
        st.rerun()

# -------------------------
# MAIN ROUTER
# -------------------------
if not st.session_state.logged_in:
    login_page()
else:
    main_dashboard()