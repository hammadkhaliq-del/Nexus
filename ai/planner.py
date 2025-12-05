# ==============================================================================
# FILE 4: ai/planner.py - STRIPS-Style Planning
# ==============================================================================

"""
NEXUS AI Module: Planner
STRIPS-style hierarchical task planning for agents.
"""

from typing import List, Dict, Set, Optional, Callable
from dataclasses import dataclass, field
from copy import deepcopy


@dataclass
class State:
    """World state representation."""
    predicates: Set[str] = field(default_factory=set)
    
    def holds(self, predicate: str) -> bool:
        return predicate in self.predicates
    
    def add(self, predicate: str):
        self.predicates.add(predicate)
    
    def remove(self, predicate: str):
        self.predicates.discard(predicate)
    
    def copy(self):
        return State(self.predicates.copy())


@dataclass
class Action:
    """STRIPS action with preconditions and effects."""
    name: str
    preconditions: Set[str]
    add_effects: Set[str]
    delete_effects: Set[str]
    cost: float = 1.0
    
    def is_applicable(self, state: State) -> bool:
        """Check if action can be applied in state."""
        return all(state.holds(p) for p in self.preconditions)
    
    def apply(self, state: State) -> State:
        """Apply action to state, return new state."""
        new_state = state.copy()
        for effect in self.delete_effects:
            new_state.remove(effect)
        for effect in self.add_effects:
            new_state.add(effect)
        return new_state


class Planner:
    """
    STRIPS-style forward-search planner.
    """
    
    def __init__(self):
        self.actions: List[Action] = []
    
    def add_action(self, action: Action):
        """Register an action."""
        self.actions.append(action)
    
    def plan(self, initial_state: State, goal: Set[str],
            max_depth: int = 20) -> Optional[List[Action]]:
        """
        Find plan from initial state to goal.
        
        Args:
            initial_state: Starting state
            goal: Goal predicates
            max_depth: Maximum plan length
        
        Returns:
            List of actions or None
        """
        from collections import deque
        
        queue = deque([(initial_state, [])])
        visited = set()
        
        while queue:
            state, plan = queue.popleft()
            
            # Check if goal reached
            if all(state.holds(g) for g in goal):
                return plan
            
            if len(plan) >= max_depth:
                continue
            
            # Try all applicable actions
            for action in self.actions:
                if action.is_applicable(state):
                    new_state = action.apply(state)
                    state_key = frozenset(new_state.predicates)
                    
                    if state_key not in visited:
                        visited.add(state_key)
                        queue.append((new_state, plan + [action]))
        
        return None  # No plan found


# Example: Navigation Planning
def create_navigation_planner(graph) -> Planner:
    """Create planner for navigation tasks."""
    planner = Planner()
    
    # Actions: move(from, to)
    for node in graph.graph.nodes():
        for neighbor in graph.get_neighbors(node):
            action = Action(
                name=f"move_{node}_to_{neighbor}",
                preconditions={f"at_{node}"},
                add_effects={f"at_{neighbor}"},
                delete_effects={f"at_{node}"},
                cost=graph.get_edge_weight(node, neighbor) or 1.0
            )
            planner.add_action(action)
    
    return planner
