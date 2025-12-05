
# ==============================================================================
# FILE 3: ai/csp_engine.py - Constraint Satisfaction Problems
# ==============================================================================

"""
NEXUS AI Module: CSP Engine
Constraint Satisfaction for scheduling, resource allocation, conflict resolution.
"""

from typing import List, Dict, Any, Set, Optional, Callable
from dataclasses import dataclass
import itertools


@dataclass
class Variable:
    """CSP Variable with domain."""
    name: str
    domain: List[Any]
    value: Optional[Any] = None
    
    def is_assigned(self) -> bool:
        return self.value is not None
    
    def __hash__(self):
        return hash(self.name)


@dataclass
class Constraint:
    """CSP Constraint between variables."""
    name: str
    variables: List[str]
    predicate: Callable[[Dict[str, Any]], bool]
    
    def is_satisfied(self, assignment: Dict[str, Any]) -> bool:
        # Check if all variables are assigned
        if not all(var in assignment for var in self.variables):
            return True  # Unassigned variables don't violate
        return self.predicate(assignment)


class CSPEngine:
    """
    Constraint Satisfaction Problem solver.
    Supports backtracking with heuristics.
    """
    
    def __init__(self):
        self.variables: Dict[str, Variable] = {}
        self.constraints: List[Constraint] = []
        self.solutions: List[Dict[str, Any]] = []
    
    def add_variable(self, name: str, domain: List[Any]) -> None:
        """Add a variable with its domain."""
        self.variables[name] = Variable(name, domain)
    
    def add_constraint(self, name: str, variables: List[str],
                      predicate: Callable[[Dict[str, Any]], bool]) -> None:
        """Add a constraint."""
        self.constraints.append(Constraint(name, variables, predicate))
    
    def solve(self, find_all: bool = False) -> Optional[Dict[str, Any]]:
        """
        Solve CSP using backtracking.
        
        Args:
            find_all: Find all solutions (not just first)
        
        Returns:
            Solution assignment or None
        """
        self.solutions.clear()
        assignment = {}
        self._backtrack(assignment, find_all)
        
        if self.solutions:
            return self.solutions[0] if not find_all else self.solutions
        return None
    
    def _backtrack(self, assignment: Dict[str, Any], find_all: bool) -> bool:
        """Recursive backtracking."""
        if len(assignment) == len(self.variables):
            self.solutions.append(assignment.copy())
            return not find_all  # Stop if only want one solution
        
        # Select unassigned variable (MRV heuristic)
        var = self._select_unassigned_variable(assignment)
        
        # Try each value in domain
        for value in self._order_domain_values(var, assignment):
            assignment[var.name] = value
            
            if self._is_consistent(assignment):
                if self._backtrack(assignment, find_all):
                    return True
            
            del assignment[var.name]
        
        return False
    
    def _select_unassigned_variable(self, assignment: Dict[str, Any]) -> Variable:
        """MRV heuristic: choose variable with smallest domain."""
        unassigned = [v for v in self.variables.values() 
                     if v.name not in assignment]
        return min(unassigned, key=lambda v: len(v.domain))
    
    def _order_domain_values(self, var: Variable, 
                            assignment: Dict[str, Any]) -> List[Any]:
        """LCV heuristic: try least constraining values first."""
        return var.domain  # Simple version, can be improved
    
    def _is_consistent(self, assignment: Dict[str, Any]) -> bool:
        """Check if assignment satisfies all constraints."""
        return all(c.is_satisfied(assignment) for c in self.constraints)
    
    def get_all_solutions(self) -> List[Dict[str, Any]]:
        """Get all solutions found."""
        return self.solutions.copy()


# Example: Agent Scheduling
def create_agent_scheduling_csp(agents: List, time_slots: List[int],
                                locations: List[tuple]) -> CSPEngine:
    """Create CSP for scheduling agents to locations at times."""
    csp = CSPEngine()
    
    # Variables: agent -> (time, location)
    for agent in agents:
        domain = list(itertools.product(time_slots, locations))
        csp.add_variable(f"agent_{agent.id}", domain)
    
    # Constraint: No two agents at same location at same time
    def no_conflict(assignment):
        values = list(assignment.values())
        return len(values) == len(set(values))
    
    agent_vars = [f"agent_{a.id}" for a in agents]
    csp.add_constraint("no_conflict", agent_vars, no_conflict)
    
    return csp