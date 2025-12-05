"""
NEXUS AI Module: Logic Engine
Rule-based reasoning system for agent decision-making.
"""

from typing import List, Dict, Any, Callable, Optional, Set
from dataclasses import dataclass
from enum import Enum


class RuleType(Enum):
    """Types of logical rules."""
    IF_THEN = "if_then"
    IF_THEN_ELSE = "if_then_else"
    CONSTRAINT = "constraint"
    GOAL = "goal"


@dataclass
class Fact:
    """Represents a fact in the knowledge base."""
    predicate: str
    arguments: tuple
    confidence: float = 1.0
    
    def __hash__(self):
        return hash((self.predicate, self.arguments))
    
    def __eq__(self, other):
        return (self.predicate == other.predicate and 
                self.arguments == other.arguments)
    
    def __repr__(self):
        args_str = ", ".join(str(arg) for arg in self.arguments)
        return f"{self.predicate}({args_str}) [{self.confidence:.2f}]"


@dataclass
class Rule:
    """Represents a logical rule."""
    name: str
    rule_type: RuleType
    conditions: List[Callable[[Dict[str, Any]], bool]]
    actions: List[Callable[[Dict[str, Any]], Any]]
    priority: int = 0
    enabled: bool = True
    
    def __repr__(self):
        return f"<Rule '{self.name}' priority={self.priority} enabled={self.enabled}>"


class LogicEngine:
    """
    Rule-based reasoning engine for agent decision-making.
    Supports forward chaining, constraint checking, and goal reasoning.
    """
    
    def __init__(self):
        self.facts: Set[Fact] = set()
        self.rules: List[Rule] = []
        self.working_memory: Dict[str, Any] = {}
        self.inference_log: List[str] = []
        self.max_iterations = 100
    
    # -------------------- Fact Management --------------------
    
    def add_fact(self, predicate: str, *arguments, confidence: float = 1.0) -> None:
        """Add a fact to the knowledge base."""
        fact = Fact(predicate, arguments, confidence)
        self.facts.add(fact)
        self.log(f"Added fact: {fact}")
    
    def remove_fact(self, predicate: str, *arguments) -> bool:
        """Remove a fact from the knowledge base."""
        fact = Fact(predicate, arguments)
        if fact in self.facts:
            self.facts.remove(fact)
            self.log(f"Removed fact: {fact}")
            return True
        return False
    
    def has_fact(self, predicate: str, *arguments) -> bool:
        """Check if a fact exists."""
        fact = Fact(predicate, arguments)
        return fact in self.facts
    
    def get_facts(self, predicate: Optional[str] = None) -> List[Fact]:
        """Get all facts, optionally filtered by predicate."""
        if predicate is None:
            return list(self.facts)
        return [f for f in self.facts if f.predicate == predicate]
    
    def clear_facts(self) -> None:
        """Clear all facts."""
        self.facts.clear()
        self.log("Cleared all facts")
    
    # -------------------- Rule Management --------------------
    
    def add_rule(self, rule: Rule) -> None:
        """Add a rule to the engine."""
        self.rules.append(rule)
        # Keep rules sorted by priority (higher first)
        self.rules.sort(key=lambda r: r.priority, reverse=True)
        self.log(f"Added rule: {rule.name}")
    
    def remove_rule(self, name: str) -> bool:
        """Remove a rule by name."""
        for i, rule in enumerate(self.rules):
            if rule.name == name:
                del self.rules[i]
                self.log(f"Removed rule: {name}")
                return True
        return False
    
    def enable_rule(self, name: str) -> bool:
        """Enable a rule."""
        for rule in self.rules:
            if rule.name == name:
                rule.enabled = True
                return True
        return False
    
    def disable_rule(self, name: str) -> bool:
        """Disable a rule."""
        for rule in self.rules:
            if rule.name == name:
                rule.enabled = False
                return True
        return False
    
    def get_rule(self, name: str) -> Optional[Rule]:
        """Get a rule by name."""
        for rule in self.rules:
            if rule.name == name:
                return rule
        return None
    
    # -------------------- Inference --------------------
    
    def forward_chain(self, context: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Forward chaining inference - apply rules until no new inferences.
        """
        if context is None:
            context = {}
        
        # Merge with working memory
        full_context = {**self.working_memory, **context}
        
        actions_taken = []
        iteration = 0
        
        while iteration < self.max_iterations:
            iteration += 1
            fired_any = False
            
            for rule in self.rules:
                if not rule.enabled:
                    continue
                
                # Check all conditions
                conditions_met = all(
                    condition(full_context) for condition in rule.conditions
                )
                
                if conditions_met:
                    self.log(f"Rule '{rule.name}' fired")
                    fired_any = True
                    
                    # Execute actions
                    for action in rule.actions:
                        result = action(full_context)
                        actions_taken.append(f"{rule.name}: {result}")
            
            # Stop if no rules fired
            if not fired_any:
                break
        
        self.log(f"Forward chaining completed in {iteration} iterations")
        return actions_taken
    
    def evaluate_constraints(self, context: Optional[Dict[str, Any]] = None) -> Dict[str, bool]:
        """Evaluate all constraint rules."""
        if context is None:
            context = {}
        
        full_context = {**self.working_memory, **context}
        results = {}
        
        for rule in self.rules:
            if rule.rule_type != RuleType.CONSTRAINT or not rule.enabled:
                continue
            
            satisfied = all(
                condition(full_context) for condition in rule.conditions
            )
            results[rule.name] = satisfied
            
            if not satisfied:
                self.log(f"Constraint '{rule.name}' violated")
        
        return results
    
    def explain_decision(self, context: Dict[str, Any]) -> str:
        """Generate natural language explanation of decision."""
        full_context = {**self.working_memory, **context}
        explanation = ["Decision reasoning:"]
        
        for rule in self.rules:
            if not rule.enabled:
                continue
            
            conditions_met = all(
                condition(full_context) for condition in rule.conditions
            )
            
            if conditions_met:
                explanation.append(f"  - Rule '{rule.name}' applies (priority {rule.priority})")
        
        return "\n".join(explanation)
    
    # -------------------- Utility --------------------
    
    def update_memory(self, key: str, value: Any) -> None:
        """Update working memory."""
        self.working_memory[key] = value
        self.log(f"Updated memory: {key} = {value}")
    
    def get_memory(self, key: str, default: Any = None) -> Any:
        """Get value from working memory."""
        return self.working_memory.get(key, default)
    
    def clear_memory(self) -> None:
        """Clear working memory."""
        self.working_memory.clear()
        self.log("Cleared working memory")
    
    def reset(self) -> None:
        """Reset engine to initial state."""
        self.facts.clear()
        self.working_memory.clear()
        self.inference_log.clear()
        self.log("Logic engine reset")
    
    def log(self, message: str) -> None:
        """Add entry to inference log."""
        self.inference_log.append(message)
    
    def get_log(self, last_n: Optional[int] = None) -> List[str]:
        """Get inference log."""
        if last_n is None:
            return self.inference_log.copy()
        return self.inference_log[-last_n:]
    
    def __repr__(self):
        return (f"<LogicEngine facts={len(self.facts)} "
                f"rules={len(self.rules)} memory={len(self.working_memory)}>")


# -------------------- Helper Functions --------------------

def create_simple_rule(name: str, 
                       condition: Callable[[Dict[str, Any]], bool],
                       action: Callable[[Dict[str, Any]], Any],
                       priority: int = 0) -> Rule:
    """Helper to create a simple if-then rule."""
    return Rule(
        name=name,
        rule_type=RuleType.IF_THEN,
        conditions=[condition],
        actions=[action],
        priority=priority
    )


def create_constraint_rule(name: str,
                           condition: Callable[[Dict[str, Any]], bool],
                           priority: int = 0) -> Rule:
    """Helper to create a constraint rule."""
    return Rule(
        name=name,
        rule_type=RuleType.CONSTRAINT,
        conditions=[condition],
        actions=[],
        priority=priority
    )


# -------------------- Example Rule Templates --------------------

class AgentRules:
    """Common rule templates for agent behavior."""
    
    @staticmethod
    def low_energy_recharge(threshold: float = 30.0) -> Rule:
        """Rule: If energy low, go recharge."""
        def condition(ctx):
            agent = ctx.get("agent")
            return agent and agent.energy < threshold
        
        def action(ctx):
            agent = ctx.get("agent")
            # FIX: We now import AgentState properly instead of guessing
            from core.agent import AgentState
            agent.state = AgentState.CHARGING
            return f"Agent {agent.name} switching to CHARGING state"
        
        return create_simple_rule(
            "low_energy_recharge",
            condition,
            action,
            priority=10
        )
    
    @staticmethod
    def obstacle_detected_replan() -> Rule:
        """Rule: If obstacle detected ahead, trigger replanning."""
        def condition(ctx):
            agent = ctx.get("agent")
            obstacles = ctx.get("detected_obstacles", [])
            path = agent.path if agent else []
            
            if not agent or not path or not obstacles:
                return False
            
            # Check if next position is blocked
            if agent.path_index < len(path):
                next_pos = path[agent.path_index]
                return next_pos in obstacles
            return False
        
        def action(ctx):
            agent = ctx.get("agent")
            return f"Agent {agent.name} needs replanning - obstacle ahead"
        
        return create_simple_rule(
            "obstacle_detected_replan",
            condition,
            action,
            priority=8
        )
    
    @staticmethod
    def goal_reached_idle() -> Rule:
        """Rule: If at goal, switch to idle."""
        def condition(ctx):
            agent = ctx.get("agent")
            return agent and agent.is_at_goal()
        
        def action(ctx):
            agent = ctx.get("agent")
            from core.agent import AgentState
            agent.state = AgentState.IDLE
            return f"Agent {agent.name} reached goal, switching to IDLE"
        
        return create_simple_rule(
            "goal_reached_idle",
            condition,
            action,
            priority=5
        )