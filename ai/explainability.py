# ==============================================================================
# FILE 6: ai/explainability.py - Decision Explanations
# ==============================================================================

"""
NEXUS AI Module: Explainability
Natural language explanations for AI decisions.
"""

from typing import List, Dict, Any, Optional


class ExplainabilityEngine:
    """
    Generates natural language explanations for AI decisions.
    """
    
    def __init__(self):
        self.explanation_templates = {
            "path_selection": "Agent chose path via {waypoints} because {reason}.",
            "action_taken": "Agent performed {action} because {reason}.",
            "constraint_violation": "Action blocked: {constraint} was violated.",
            "goal_selection": "Agent set goal {goal} because {reason}.",
        }
    
    def explain_path(self, agent, path: List[tuple], 
                    algorithm: str, cost: float) -> str:
        """Explain path selection."""
        if not path:
            return f"No valid path found for {agent.name}."
        
        waypoints = f"{path[0]} -> ... -> {path[-1]}"
        reason = f"it is the shortest path (cost: {cost:.2f}) found by {algorithm}"
        
        return self.explanation_templates["path_selection"].format(
            waypoints=waypoints, reason=reason
        )
    
    def explain_action(self, agent, action: str, context: Dict) -> str:
        """Explain why an action was taken."""
        reasons = []
        
        if agent.energy < 30:
            reasons.append("energy was low")
        
        if agent.goal and not agent.is_at_goal():
            reasons.append(f"moving toward goal at {agent.goal}")
        
        if not reasons:
            reasons.append("it was the next action in the plan")
        
        reason = " and ".join(reasons)
        
        return self.explanation_templates["action_taken"].format(
            action=action, reason=reason
        )
    
    def explain_constraint_failure(self, constraint_name: str,
                                   details: Dict) -> str:
        """Explain why a constraint failed."""
        return self.explanation_templates["constraint_violation"].format(
            constraint=constraint_name
        )
    
    def explain_replanning(self, agent, old_path: List,
                          new_path: List, reason: str) -> str:
        """Explain why replanning occurred."""
        return (f"{agent.name} replanned route because {reason}. "
                f"New path has {len(new_path)} waypoints "
                f"(was {len(old_path)}).")
    
    def generate_full_explanation(self, agent, decision_log: List[Dict]) -> str:
        """Generate comprehensive explanation from decision log."""
        lines = [f"Decision trace for {agent.name}:"]
        
        for i, entry in enumerate(decision_log, 1):
            decision_type = entry.get("type")
            
            if decision_type == "path":
                lines.append(f"  {i}. Selected path using {entry.get('algorithm')}")
            elif decision_type == "action":
                lines.append(f"  {i}. Performed {entry.get('action')}")
            elif decision_type == "state_change":
                lines.append(f"  {i}. Changed state to {entry.get('new_state')}")
        
        return "\n".join(lines)