"""
NEXUS Core Module: Graph Builder
Converts city grid into NetworkX graph for pathfinding and AI operations.
"""

import networkx as nx
import math
from typing import Tuple, List, Optional, Dict
from .city import City

EdgeKey = Tuple[Tuple[int, int], Tuple[int, int]]


class CityGraph:
    """
    Manages NetworkX graph representation of the city.
    Supports dynamic updates, weights, and pathfinding operations.
    """

    def __init__(self, city: Optional[City] = None):
        self.city = city
        self.graph = nx.Graph()
        self.blocked_nodes: set[Tuple[int, int]] = set()
        self.weights: Dict[EdgeKey, float] = {}
        self._base_graph: Optional[nx.Graph] = None

        if city:
            self.build_graph()

    def build_graph(self, city: Optional[City] = None, diagonal: bool = False) -> nx.Graph:
        """
        Build graph from city grid.

        Args:
            city: City instance (uses self.city if not provided)
            diagonal: Allow diagonal movement

        Returns:
            NetworkX graph
        """
        if city:
            self.city = city
        if not self.city:
            raise ValueError("No city provided to build graph from")

        self.graph.clear()
        self.blocked_nodes.clear()
        self.weights.clear()

        # Add walkable nodes
        for row in range(self.city.height):
            for col in range(self.city.width):
                if self.city.is_walkable(row, col):
                    self.graph.add_node((row, col), pos=(row, col))

        # Add edges
        for node in self.graph.nodes():
            row, col = node
            neighbors = self.city.walkable_neighbors(row, col, diagonal=diagonal)
            for neighbor in neighbors:
                if neighbor in self.graph.nodes():
                    weight = self._calculate_distance(node, neighbor)
                    self.graph.add_edge(node, neighbor, weight=weight)
                    self.weights[(node, neighbor)] = weight
                    self.weights[(neighbor, node)] = weight

        # Save base graph for reset
        self._base_graph = self.graph.copy()

        print(f"✓ Graph built: {self.graph.number_of_nodes()} nodes, "
              f"{self.graph.number_of_edges()} edges")
        return self.graph

    def reset_to_base(self) -> None:
        """Reset graph to original base state."""
        if self._base_graph is None:
            raise RuntimeError("No base graph loaded")
        self.graph = self._base_graph.copy()
        self.blocked_nodes.clear()
        self.weights = {(u, v): d['weight'] for u, v, d in self.graph.edges(data=True)}

    def _calculate_distance(self, node1: Tuple[int, int], node2: Tuple[int, int]) -> float:
        """Euclidean distance between two nodes."""
        r1, c1 = node1
        r2, c2 = node2
        return math.sqrt((r2 - r1) ** 2 + (c2 - c1) ** 2)

    def get_neighbors(self, node: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Return neighbor nodes in the graph."""
        if node not in self.graph:
            return []
        return list(self.graph.neighbors(node))

    def update_edge_weight(self, node1: Tuple[int, int], node2: Tuple[int, int], weight: float) -> bool:
        """Update weight of an edge."""
        if not self.graph.has_edge(node1, node2):
            return False
        self.graph[node1][node2]['weight'] = weight
        self.weights[(node1, node2)] = weight
        self.weights[(node2, node1)] = weight
        return True

    def get_edge_weight(self, node1: Tuple[int, int], node2: Tuple[int, int]) -> Optional[float]:
        """Get weight of an edge."""
        if not self.graph.has_edge(node1, node2):
            return None
        return self.graph[node1][node2].get('weight', 1.0)

    def block_node(self, node: Tuple[int, int]) -> bool:
        """Block a node (road closure, hazard, etc.)."""
        if node not in self.graph:
            return False
        self.blocked_nodes.add(node)
        edges = list(self.graph.edges(node))
        for edge in edges:
            self.graph.remove_edge(*edge)
        return True

    def unblock_node(self, node: Tuple[int, int], diagonal: bool = False) -> bool:
        """Unblock a previously blocked node."""
        if node not in self.blocked_nodes:
            return False
        self.blocked_nodes.remove(node)
        row, col = node
        neighbors = self.city.walkable_neighbors(row, col, diagonal=diagonal)
        for neighbor in neighbors:
            if neighbor in self.graph.nodes() and neighbor not in self.blocked_nodes:
                weight = self._calculate_distance(node, neighbor)
                self.graph.add_edge(node, neighbor, weight=weight)
                self.weights[(node, neighbor)] = weight
                self.weights[(neighbor, node)] = weight
        return True

    def is_blocked(self, node: Tuple[int, int]) -> bool:
        """Check if a node is blocked."""
        return node in self.blocked_nodes

    def heuristic(self, node1: Tuple[int, int], node2: Tuple[int, int], method: str = "euclidean") -> float:
        """Heuristic distance for A*, etc."""
        r1, c1 = node1
        r2, c2 = node2
        if method == "manhattan":
            return abs(r2 - r1) + abs(c2 - c1)
        elif method == "chebyshev":
            return max(abs(r2 - r1), abs(c2 - c1))
        else:
            return math.sqrt((r2 - r1) ** 2 + (c2 - c1) ** 2)

    def get_subgraph(self, center: Tuple[int, int], radius: int) -> nx.Graph:
        """Extract subgraph around center node."""
        cr, cc = center
        nodes = [node for node in self.graph.nodes() if abs(node[0]-cr)<=radius and abs(node[1]-cc)<=radius]
        return self.graph.subgraph(nodes).copy()

    def get_stats(self) -> Dict:
        """Return graph statistics."""
        stats = {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "blocked_nodes": len(self.blocked_nodes),
            "average_degree": round(sum(dict(self.graph.degree()).values())/self.graph.number_of_nodes(), 2) if self.graph.number_of_nodes() > 0 else 0,
            "is_connected": nx.is_connected(self.graph) if self.graph.number_of_nodes() > 0 else False
        }
        return stats

    def add_traffic(self, node1: Tuple[int, int], node2: Tuple[int, int], multiplier: float = 2.0) -> bool:
        """Increase edge weight to simulate traffic."""
        current = self.get_edge_weight(node1, node2)
        if current is None:
            return False
        return self.update_edge_weight(node1, node2, current * multiplier)

    def clear_traffic(self, node1: Tuple[int, int], node2: Tuple[int, int]) -> bool:
        """Reset edge to original distance weight."""
        weight = self._calculate_distance(node1, node2)
        return self.update_edge_weight(node1, node2, weight)

    def __repr__(self) -> str:
        return f"<CityGraph nodes={self.graph.number_of_nodes()} edges={self.graph.number_of_edges()}>"
