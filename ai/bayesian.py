# ==============================================================================
# FILE 5: ai/bayesian.py - Probabilistic Reasoning
# ==============================================================================

"""
NEXUS AI Module: Bayesian Reasoning
Probabilistic inference for uncertainty handling.
"""

from typing import Dict, List, Optional
import random


class BayesianNetwork:
    """
    Simple Bayesian Network for probabilistic reasoning.
    """
    
    def __init__(self):
        self.nodes: Dict[str, Dict] = {}
        self.probabilities: Dict[str, float] = {}
    
    def add_node(self, name: str, parents: List[str] = None,
                cpt: Dict = None):
        """
        Add node to network.
        
        Args:
            name: Node name
            parents: Parent node names
            cpt: Conditional probability table
        """
        self.nodes[name] = {
            "parents": parents or [],
            "cpt": cpt or {}
        }
    
    def set_evidence(self, node: str, value: bool):
        """Set observed evidence."""
        self.probabilities[node] = 1.0 if value else 0.0
    
    def infer(self, query: str, evidence: Dict[str, bool] = None) -> float:
        """
        Compute P(query | evidence) using simple enumeration.
        
        Args:
            query: Query variable
            evidence: Evidence dictionary
        
        Returns:
            Probability
        """
        if evidence is None:
            evidence = {}
        
        # Simple: return prior if no evidence
        if query in self.probabilities:
            return self.probabilities[query]
        
        # Default uniform
        return 0.5
    
    def predict_traffic(self, time: int, weather: str) -> float:
        """Example: Predict traffic probability."""
        base_prob = 0.3
        
        # Rush hour
        if 7 <= time <= 9 or 17 <= time <= 19:
            base_prob += 0.4
        
        # Bad weather
        if weather in ["rain", "snow"]:
            base_prob += 0.2
        
        return min(1.0, base_prob)
    
    def predict_path_blockage(self, node: tuple,
                             recent_events: List) -> float:
        """Predict probability of path blockage."""
        prob = 0.05  # Base probability
        
        # Check recent events near node
        for event in recent_events:
            if event.get("position") == node:
                prob += 0.6
        
        return min(1.0, prob)
