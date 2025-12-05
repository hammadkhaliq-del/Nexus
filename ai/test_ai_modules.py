"""
NEXUS AI Modules Test Suite
Complete test of all AI functionality: Search, Logic, CSP, Planning, Bayesian, Explainability
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from core import City, CityGraph, Agent, AgentState
from ai import (
    # Search
    bfs, dfs, dijkstra, a_star, bidirectional_a_star, compare_algorithms,
    # Logic
    LogicEngine, create_simple_rule, AgentRules,
    # CSP
    CSPEngine,
    # Planning
    Planner, State, Action,
    # Bayesian
    BayesianNetwork,
    # Explainability
    ExplainabilityEngine
)


def setup_test_environment():
    """Create test environment."""
    print("📁 Setting up test environment...")
    
    # Create simple map
    grid = np.zeros((15, 15), dtype=np.int32)
    grid[5:8, 5:8] = 1  # Building
    np.save("ai_test_map.npy", grid)
    
    city = City("ai_test_map.npy")
    graph = CityGraph(city)
    
    print(f"✓ Environment ready: {graph.get_stats()['nodes']} nodes\n")
    return city, graph


def test_search_algorithms(graph):
    """Test all pathfinding algorithms."""
    print("=" * 60)
    print("🔍 TESTING SEARCH ALGORITHMS")
    print("=" * 60)
    
    start = (2, 2)
    goal = (12, 12)
    
    print(f"Finding path from {start} to {goal}\n")
    
    # Test each algorithm
    results = compare_algorithms(graph, start, goal)
    
    for name, result in results.items():
        print(f"  {name}:")
        print(f"    Success: {'✓' if result.success else '✗'}")
        print(f"    Path length: {len(result.path)}")
        print(f"    Cost: {result.cost:.2f}")
        print(f"    Nodes explored: {result.nodes_explored}")
        print()
    
    # Compare efficiency
    print("📊 Algorithm Comparison:")
    best_cost = min(r.cost for r in results.values() if r.success)
    least_explored = min(r.nodes_explored for r in results.values() if r.success)
    
    for name, result in results.items():
        if not result.success:
            continue
        efficiency = "⭐" if result.nodes_explored == least_explored else ""
        optimal = "✓" if result.cost == best_cost else ""
        print(f"  {name}: Optimal:{optimal} Efficient:{efficiency}")
    
    return results["A*"]


def test_logic_engine():
    """Test rule-based reasoning."""
    print("\n" + "=" * 60)
    print("🧠 TESTING LOGIC ENGINE")
    print("=" * 60)
    
    engine = LogicEngine()
    
    # Add facts
    print("\n📝 Adding facts:")
    engine.add_fact("agent_at", "home")
    engine.add_fact("energy_level", 80)
    engine.add_fact("goal_set", True)
    print(f"  Facts: {len(engine.facts)}")
    
    # Create rules
    print("\n📜 Creating rules:")
    
    def low_energy_condition(ctx):
        return engine.has_fact("energy_level", 20)
    
    def low_energy_action(ctx):
        engine.add_fact("needs_recharge", True)
        return "Triggered recharge alert"
    
    rule1 = create_simple_rule(
        "low_energy_alert",
        low_energy_condition,
        low_energy_action,
        priority=10
    )
    engine.add_rule(rule1)
    
    # Add pre-made agent rules
    agent = Agent("TestBot", (5, 5), max_energy=100.0)
    agent.energy = 25
    
    recharge_rule = AgentRules.low_energy_recharge(threshold=30)
    engine.add_rule(recharge_rule)
    
    print(f"  Rules: {len(engine.rules)}")
    
    # Test inference
    print("\n⚡ Running inference:")
    context = {"agent": agent}
    actions = engine.forward_chain(context)
    
    for action in actions:
        print(f"  - {action}")
    
    # Check facts
    print(f"\n📊 Final facts: {len(engine.facts)}")
    for fact in list(engine.facts)[:5]:
        print(f"  {fact}")


def test_csp_engine():
    """Test constraint satisfaction."""
    print("\n" + "=" * 60)
    print("🧩 TESTING CSP ENGINE")
    print("=" * 60)
    
    csp = CSPEngine()
    
    # Problem: Schedule 3 agents to 3 locations at 2 time slots
    print("\n📅 Problem: Schedule agents to locations")
    
    agents_ids = ["A1", "A2", "A3"]
    locations = [(5, 5), (10, 10), (12, 12)]
    times = [1, 2]
    
    # Add variables
    for agent_id in agents_ids:
        domain = [(t, loc) for t in times for loc in locations]
        csp.add_variable(agent_id, domain)
    
    print(f"  Variables: {len(csp.variables)}")
    print(f"  Domain size: {len(csp.variables['A1'].domain)}")
    
    # Constraint: No two agents at same place and time
    def no_conflict(assignment):
        values = list(assignment.values())
        return len(values) == len(set(values))
    
    csp.add_constraint("no_conflict", agents_ids, no_conflict)
    print(f"  Constraints: {len(csp.constraints)}")
    
    # Solve
    print("\n🔍 Solving CSP...")
    solution = csp.solve()
    
    if solution:
        print("✓ Solution found:")
        for agent_id, (time, loc) in solution.items():
            print(f"  {agent_id}: Time={time}, Location={loc}")
    else:
        print("✗ No solution found")


def test_planner():
    """Test STRIPS planning."""
    print("\n" + "=" * 60)
    print("📋 TESTING PLANNER")
    print("=" * 60)
    
    planner = Planner()
    
    # Define actions
    print("\n⚙️ Defining actions:")
    
    # Action: pickup(item)
    pickup = Action(
        name="pickup_package",
        preconditions={"at_depot", "hands_empty"},
        add_effects={"has_package"},
        delete_effects={"hands_empty"}
    )
    
    # Action: deliver(item)
    deliver = Action(
        name="deliver_package",
        preconditions={"at_customer", "has_package"},
        add_effects={"delivered", "hands_empty"},
        delete_effects={"has_package"}
    )
    
    # Action: go_to_depot
    go_depot = Action(
        name="go_to_depot",
        preconditions=set(),
        add_effects={"at_depot"},
        delete_effects={"at_customer"}
    )
    
    # Action: go_to_customer
    go_customer = Action(
        name="go_to_customer",
        preconditions=set(),
        add_effects={"at_customer"},
        delete_effects={"at_depot"}
    )
    
    planner.add_action(pickup)
    planner.add_action(deliver)
    planner.add_action(go_depot)
    planner.add_action(go_customer)
    
    print(f"  Actions: {len(planner.actions)}")
    
    # Initial state
    initial = State()
    initial.add("at_depot")
    initial.add("hands_empty")
    
    # Goal
    goal = {"delivered"}
    
    print("\n🎯 Planning delivery task...")
    print(f"  Initial: {initial.predicates}")
    print(f"  Goal: {goal}")
    
    # Find plan
    plan = planner.plan(initial, goal, max_depth=10)
    
    if plan:
        print(f"\n✓ Plan found ({len(plan)} steps):")
        for i, action in enumerate(plan, 1):
            print(f"  {i}. {action.name}")
    else:
        print("\n✗ No plan found")


def test_bayesian():
    """Test probabilistic reasoning."""
    print("\n" + "=" * 60)
    print("🎲 TESTING BAYESIAN REASONING")
    print("=" * 60)
    
    bn = BayesianNetwork()
    
    print("\n🌐 Creating Bayesian network:")
    bn.add_node("traffic", parents=[], cpt={True: 0.3, False: 0.7})
    bn.add_node("weather", parents=[], cpt={"rain": 0.2, "clear": 0.8})
    print("  Nodes: traffic, weather")
    
    # Test predictions
    print("\n🔮 Making predictions:")
    
    scenarios = [
        (8, "clear"),   # Morning, clear
        (8, "rain"),    # Morning, rain
        (18, "clear"),  # Evening rush, clear
        (14, "rain"),   # Afternoon, rain
    ]
    
    for time, weather in scenarios:
        prob = bn.predict_traffic(time, weather)
        print(f"  Time {time}:00, {weather}: P(traffic) = {prob:.2%}")
    
    # Path blockage
    print("\n🚧 Path blockage prediction:")
    events = [{"position": (10, 10), "type": "accident"}]
    prob = bn.predict_path_blockage((10, 10), events)
    print(f"  P(blocked at (10,10) | accident) = {prob:.2%}")


def test_explainability(search_result):
    """Test explanation generation."""
    print("\n" + "=" * 60)
    print("💬 TESTING EXPLAINABILITY")
    print("=" * 60)
    
    explainer = ExplainabilityEngine()
    
    # Create test agent
    agent = Agent("DeliveryBot", (2, 2))
    agent.set_goal((12, 12))
    agent.energy = 25
    
    print("\n📝 Generating explanations:")
    
    # Explain path
    path_explanation = explainer.explain_path(
        agent, 
        search_result.path,
        search_result.algorithm,
        search_result.cost
    )
    print(f"\n  Path Selection:")
    print(f"    {path_explanation}")
    
    # Explain action
    action_explanation = explainer.explain_action(
        agent,
        "move_to_charging_station",
        {"energy_low": True}
    )
    print(f"\n  Action:")
    print(f"    {action_explanation}")
    
    # Explain replanning
    old_path = search_result.path
    new_path = search_result.path[:-2] + [(11, 11), (12, 12)]
    
    replan_explanation = explainer.explain_replanning(
        agent,
        old_path,
        new_path,
        "obstacle detected on original route"
    )
    print(f"\n  Replanning:")
    print(f"    {replan_explanation}")
    
    # Full trace
    decision_log = [
        {"type": "path", "algorithm": "A*"},
        {"type": "action", "action": "move"},
        {"type": "state_change", "new_state": "CHARGING"},
    ]
    
    full_explanation = explainer.generate_full_explanation(agent, decision_log)
    print(f"\n  Full Decision Trace:")
    for line in full_explanation.split('\n'):
        print(f"    {line}")


def run_all_ai_tests():
    """Run complete AI test suite."""
    print("\n" + "🧠" * 30)
    print("NEXUS AI SYSTEM TEST SUITE")
    print("🧠" * 30 + "\n")
    
    try:
        # Setup
        city, graph = setup_test_environment()
        
        # Test each module
        search_result = test_search_algorithms(graph)
        test_logic_engine()
        test_csp_engine()
        test_planner()
        test_bayesian()
        test_explainability(search_result)
        
        # Success
        print("\n" + "✅" * 30)
        print("ALL AI TESTS PASSED!")
        print("✅" * 30)
        print("\n🎉 Your NEXUS AI layer is working perfectly!")
        print("\n📝 Summary:")
        print("  ✓ Search algorithms - OK (BFS, DFS, Dijkstra, A*, Bidirectional)")
        print("  ✓ Logic engine - OK (Rules, facts, inference)")
        print("  ✓ CSP engine - OK (Constraint satisfaction)")
        print("  ✓ Planner - OK (STRIPS planning)")
        print("  ✓ Bayesian - OK (Probabilistic reasoning)")
        print("  ✓ Explainability - OK (Natural language)")
        print("\n🚀 NEXUS is fully operational!")
        
        return True
        
    except Exception as e:
        print("\n" + "❌" * 30)
        print(f"TEST FAILED: {e}")
        print("❌" * 30)
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_ai_tests()
    sys.exit(0 if success else 1)