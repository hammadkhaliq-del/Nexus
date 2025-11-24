# NEXUS: Intelligent Urban Simulation Platform

> **NEXUS** is a real-time urban simulation and AI reasoning engine that integrates **Search**, **Logic**, **CSP**, **Probabilistic Models**, and **Planning** into a single interactive city environment.
> Built entirely in **Python**, powered by **Streamlit**, **Plotly**, and a suite of AI modules, NEXUS showcases how multiple AI paradigms collaborate to handle complex city-scale challenges such as traffic flow, disasters, resource allocation, and uncertainty.

---

# 🚀 **Features at a Glance**

### 🏙️ **1. Real-Time City Simulation**

* Dynamic grid-based city model
* Agents that move, reroute, wait, and respond to events
* Real-time tick-based simulation engine
* Deterministic replay (seeded) for grading consistency

### 🤖 **2. Multi-Paradigm AI System**

NEXUS integrates **five AI paradigms**:

* **Search** → A* + NetworkX shortest paths
* **Logic Reasoning** → Custom rule-engine with Horn-like rules
* **CSP Solving** → Resource allocation, power distribution
* **Bayesian Reasoning** → Accident probability modeling
* **HTN Planning** → Disaster response via PyHOP planners

### 🗺️ **3. Interactive Dashboard**

* Full GUI built in **Streamlit**
* Plotly-powered city map visualization
* Side-by-side panels for agents, metrics, logs, and events
* Scenario selector (Rush Hour, Storm, Hospital Crisis, etc.)

### 🧠 **4. Explainable AI (XAI)**

* Decision logs showing why AI took each action
* Pathfinding visualizations
* Bayesian inference traces
* CSP violations & reasoning summaries

### 📦 **5. Scenario Packs**

Configurable JSON/YAML scenario files for:

* City layout
* Initial agent states
* Bayesian parameters
* CSP constraints
* Weather + event probabilities

---

# 🧩 **Project Structure**

```
NEXUS/
│
├── app.py                     # Streamlit main entry point
├── core/
│   ├── city_map.py            # Grid + map utilities
│   ├── graph_builder.py       # Grid → NetworkX graph converter
│   ├── event_bus.py           # Publisher/subscriber system
│   ├── snapshot.py            # Snapshot + replay manager
│
├── simulation/
│   ├── agent.py               # Agent model
│   ├── simulation_loop.py     # Tick manager + world updates
│   ├── events.py              # Accident, fire, roadblock events
│
├── ai/
│   ├── search.py              # A* + NX shortest path interface
│   ├── logic_engine.py        # IF–THEN rules + evaluator
│   ├── csp_solver.py          # python-constraint wrapper
│   ├── planner_hop.py         # PyHOP domain + operators
│   ├── bayes.py               # pgmpy Bayesian model wrapper
│   ├── explainability.py      # XAI utilities
│
├── ui/
│   ├── layout.py              # Streamlit page layout & columns
│   ├── map_renderer.py        # Plotly visualizations
│   ├── metrics_panel.py       # Live metrics
│   ├── logs_panel.py          # XAI explanation logs
│
├── scenario_packs/
│   ├── rush_hour.json
│   ├── storm.json
│   ├── hospital_crisis.json
│   └── stress_test.json
│
├── tests/
│   ├── test_search.py
│   ├── test_csp.py
│   ├── test_graph.py
│   └── test_events.py
│
├── docs/
│   ├── architecture.md
│   ├── uml_diagrams.pdf
│   └── user_guide.md
│
├── requirements.txt
└── README.md
```

---

# 🛠️ **Technologies Used**

### **Core Language**

* **Python 3.10+**

### **Frontend & Dashboard**

* **Streamlit** → GUI, layout, interactivity
* **Plotly** → Live map + agent visualization

### **AI Modules**

* **NetworkX** → Graph + shortest paths
* **Custom A*** → Pathfinding with expansion metrics
* **python-constraint** → CSP solver
* **PyHOP** → HTN planning
* **pgmpy** → Bayesian Networks
* **Custom Rule Engine** → Logic reasoning

### **Data & Utilities**

* **NumPy** → Grid representation
* **JSON/YAML** → Scenario Packs
* **Rich** (optional) → Better logs
* **TQDM** (optional) → Console progress

### **NO Django / Flask / HTML / CSS / JS**

Everything is built directly in Python through Streamlit.

---

# ▶️ **How to Run**

### 1. Clone the Repo

```
git clone https://github.com/your-username/NEXUS.git
cd NEXUS
```

### 2. Install Dependencies

```
pip install -r requirements.txt
```

### 3. Start the Simulation Dashboard

```
streamlit run app.py
```

### 4. Select a Scenario

Choose from the sidebar (Rush Hour, Storm, Hospital Crisis, etc.)

### 5. Press **Start Simulation**

Watch agents navigate, planners dispatch, CSP allocate power, and the Bayesian engine spawn events.

---

# 🎛️ **Key UI Panels**

### **City Map**

Shows:

* Buildings
* Roads
* Agents (moving)
* Accidents, fires, blocked roads
* Live updates every tick

### **Metrics Panel**

Displays:

* Avg travel time
* Planner success rate
* CSP violations
* Accident probability
* System health

### **Event Log (XAI)**

Explains:

* Why path was chosen
* Why CSP reallocated
* Why planner executed tasks
* Why events were generated

---

# 🔬 **AI Components (Detailed)**

### 🔍 Search

* Custom A* (with open/closed set visualization)
* NetworkX fallback (Dijkstra)
* Rerouting around accidents or closures

### 📏 Logic Engine

Simple rule-based system:

```
IF accident_detected AND traffic_congestion_high THEN reroute_agents
```

### 🧩 CSP Solver

Manages:

* Power distribution
* Resource minimization
* Priority zones (hospital, fire station)

### 🌧️ Bayesian Network

Predicts probabilities of events like:

* Accidents
* Road closures
* Emergency spikes
  Based on variables like rain, time, density.

### 🚑 HTN Planning

Ambulance executes multi-step plans such as:

* Drive → Load → Deliver → Return

---

# 📊 **Explainability (XAI)**

NEXUS provides clear insight into the AI’s internal decisions:

* Path decision traces
* Bayesian inference tables
* CSP constraint violation reasons
* HTN plan decomposition
* Human-readable explanations

Example:

> “Ambulance 04 rerouted because Node 12 marked unsafe (P(accident) = 0.82).”

---

# 🧪 **Testing**

Includes:

* **Unit Tests** (Search, CSP, Graph, Event Bus)
* **Integration Tests** (Scenario Packs)
* **Stress Tests** (high-agent load)
* **Deterministic Seeds** for reproducibility

Run all tests:

```
pytest
```

---

# 📚 **Documentation**

Inside `/docs/`:

* Architecture Overview
* Class Diagrams
* Activity Diagrams
* Scenario Pack Format
* User Guide

---

# 🎥 **Demo**

Demo includes:

* Accident generation
* Emergency dispatch (HTN)
* CSP power reallocation
* Bayesian weather impact
* Replay mode
* XAI decision logs

Video available at: *(Add your link here)*

---

# 📌 **Future Improvements**

* Multi-agent negotiation (auction-based resource allocation)
* ML-based traffic prediction model
* Real-world map imports (OpenStreetMap)
* Federated agent intelligence
* Web deployment (Streamlit Cloud)

---

# ⭐ **Contributors**

* **Hammad** — Lead Developer, Architect, AI Modules
* *(Add teammates)*

---

# 📝 **License**

MIT License — free to use, modify, and distribute.

---

# 🏁 **Final Note**

NEXUS isn’t just a simulation — it’s a **multi-intelligence testbed** demonstrating how various AI paradigms collaborate to manage complex, uncertain real-world environments.
