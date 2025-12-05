"""
NEXUS AI Module Package
Exports all AI functionality.
"""

from .search import (
    SearchResult,
    bfs, dfs, dijkstra, a_star, bidirectional_a_star,
    compare_algorithms, find_all_paths_bfs
)

from .logic_engine import (
    LogicEngine, Rule, Fact, RuleType,
    create_simple_rule, create_constraint_rule,
    AgentRules
)

from .csp_engine import (
    CSPEngine, Variable, Constraint,
    create_agent_scheduling_csp
)

from .planner import (
    Planner, State, Action,
    create_navigation_planner
)

from .bayesian import BayesianNetwork

from .explainability import ExplainabilityEngine

__all__ = [
    # Search
    "SearchResult", "bfs", "dfs", "dijkstra", "a_star", 
    "bidirectional_a_star", "compare_algorithms", "find_all_paths_bfs",
    
    # Logic
    "LogicEngine", "Rule", "Fact", "RuleType",
    "create_simple_rule", "create_constraint_rule", "AgentRules",
    
    # CSP
    "CSPEngine", "Variable", "Constraint", "create_agent_scheduling_csp",
    
    # Planning
    "Planner", "State", "Action", "create_navigation_planner",
    
    # Bayesian
    "BayesianNetwork",
    
    # Explainability
    "ExplainabilityEngine",
]