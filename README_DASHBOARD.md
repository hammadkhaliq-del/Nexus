# NEXUS 3D Holographic Dashboard

Professional-grade 3D holographic dashboard for NEXUS city navigation simulation system.

## Features

### 🎨 Visual Design
- **Cyberpunk aesthetic** with neon cyan, magenta, and blue color scheme
- **Holographic effects** with glowing text and borders
- **3D visualization** using Plotly with rotating camera
- **Rajdhani font** for futuristic appearance

### 🗺️ 3D HOLO-MAP
- **Glowing neon buildings** with randomized heights (12-30 units)
- **Holographic grid base** with cyan grid lines
- **Animated agent paths** as bright glowing trails
- **Large agent markers** as glowing diamonds with labels
- **Rotating camera** for cinematic city overview

### 📊 Dashboard Panels

#### Left Panel
- **WORLD STATS**: Agent counts, tick counter, entity tracking
- **AGENT OVERVIEW**: State distribution, AI success rate, average energy
- **LAYER CONTROLS**: Toggle visibility of grid, agents, and paths

#### Center Panel
- **HOLO-MAP**: Main 3D visualization with interactive controls

#### Right Panel
- **LIVE EVENT FEED**: Color-coded event log with timestamps
- **AGENT INSPECTOR**: Detailed info for selected agent (ID, state, path, energy)

### 🎮 Controls
- **Play/Pause**: Start/stop simulation
- **Speed Slider**: Adjust simulation speed (0.5x - 5.0x)
- **Layer Toggles**: Show/hide different visualization layers

## Quick Start

### Installation

```bash
# Install dependencies
pip install streamlit plotly pandas numpy networkx
```

### Run the Dashboard

```bash
# Start the dashboard
streamlit run nexus_dashboard.py
```

### Usage

1. **Login**: Enter any username/password (demo mode)
2. **Initialize**: Click "🚀 INITIALIZE SYSTEM"
3. **Observe**: Watch the system initialize city, graph, and agents
4. **Start**: Press ▶️ to start the simulation
5. **Explore**: Use controls to adjust speed and visibility

## Architecture

### Backend Integration

The dashboard integrates with:

**Core Modules:**
- `core.city` - City grid management
- `core.graph` - Navigation graph (NetworkX)
- `core.agent` - Agent behavior and state
- `core.simulation` - Simulation engine

**AI Modules:**
- `ai.search` - A* pathfinding algorithm
- `ai.logic_engine` - Rule-based decisions
- `ai.explainability` - Decision explanations
- `ai.bayesian` - Probabilistic reasoning

### Configuration

Constants defined at the top of `nexus_dashboard.py`:

```python
# Color scheme
HOLO_CYAN = "#00ffff"
HOLO_MAGENTA = "#ff00ff"
HOLO_BLUE = "#0080ff"
HOLO_GREEN = "#00ff80"

# Simulation parameters
EVENT_LOG_PROBABILITY = 0.1
LOW_ENERGY_THRESHOLD = 20.0
PERF_SPEED_MULTIPLIER = 1.2
MIN_SLEEP_DURATION = 0.01
```

## Features in Detail

### 3D Visualization
- **Buildings**: Random heights (12-30 units) with color variation
- **Parks**: Green areas at ground level
- **Grid**: Cyan glowing lines every 5 units
- **Agents**: Diamond markers elevated at z=15 for visibility
- **Paths**: Elevated trails at z=10 with color coding

### Event Logging
Events are color-coded by type:
- 🔵 **SYSTEM** (Cyan): System initialization and status
- 🟣 **AI** (Purple): AI decisions and pathfinding
- 🟢 **SUCCESS** (Green): Goal achievements
- 🔴 **WARNING** (Pink): Energy alerts and issues

### Agent Inspector
Real-time metrics for selected agent:
- **ID**: Unique identifier
- **State**: MOVING, IDLE, CHARGING, OFFLINE
- **Energy**: Current energy percentage
- **Path**: A* algorithm, length, progress
- **Speed**: Movement speed and performance metrics

## Technical Details

### Performance
- **Auto-refresh**: 10 FPS with adjustable speed multiplier
- **Minimum sleep**: 0.01s to prevent CPU overuse
- **Event buffer**: Last 30 events retained

### Safety Features
- Agent count validation against walkable positions
- Even distribution of agents across city
- Energy threshold warnings
- Camera rotation limiting

## Screenshots

![Login Page](https://github.com/user-attachments/assets/3b0f54be-0d92-49ad-8cbc-9101f72a813b)
*Login page with holographic branding*

![Dashboard](https://github.com/user-attachments/assets/d439f461-83a2-4a1c-af00-99df2a062102)
*Main dashboard with simulation running*

## Requirements

- Python 3.8+
- streamlit >= 1.35
- plotly >= 5.18.0
- numpy >= 2.0.0
- pandas >= 2.0.0
- networkx >= 3.1

## Troubleshooting

### Dashboard won't start
```bash
# Check if all dependencies are installed
pip install -r requirements.txt

# Verify imports
python -c "import streamlit; import plotly; print('OK')"
```

### No agents visible
- Ensure "👤 Show Agents" is checked in Layer Controls
- Click the play button to start simulation
- Check that city initialization succeeded in event log

### 3D visualization not showing
- Check browser console for WebGL errors
- Try a different browser (Chrome/Firefox recommended)
- Ensure graphics drivers are up to date

## License

Part of the NEXUS city navigation simulation system.
