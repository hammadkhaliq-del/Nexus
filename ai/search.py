"""
NEXUS AI Module: Search Algorithms
Classical pathfinding: BFS, DFS, Dijkstra, A*, Bidirectional A*
All algorithms use (row, col) coordinate system.
"""

from typing import List, Tuple, Optional, Dict, Set, Callable
from collections import deque
import heapq
import math

# ==============================================================================
# 1. SEARCH ENGINE CLASS (REQUIRED FOR NEXUS PRIME)
# ==============================================================================
class SearchEngine:
    """
    Static class wrapper to handle pathfinding requests for Agents.
    """
    @staticmethod
    def a_star(graph, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Streamlined A* implementation for Agents.
        Returns just the path list (or empty list if failed).
        """
        # We can reuse the standalone logic below, but we need to adapt the return type
        # The standalone 'a_star' returns a SearchResult object.
        # This method needs to return a simple List[Tuple].
        
        result = a_star(graph, start, goal, heuristic="euclidean")
        return result.path if result.success else []

# ==============================================================================
# 2. STANDALONE ALGORITHMS (EXISTING LIBRARY)
# ==============================================================================

class SearchResult:
    """Container for search results with metadata."""
    
    def __init__(self, path: List[Tuple[int, int]], cost: float, 
                 nodes_explored: int, algorithm: str):
        self.path = path
        self.cost = cost
        self.nodes_explored = nodes_explored
        self.algorithm = algorithm
        self.success = len(path) > 0
    
    def __repr__(self) -> str:
        status = "✓" if self.success else "✗"
        return (f"<SearchResult {status} {self.algorithm}: "
                f"path_length={len(self.path)}, cost={self.cost:.2f}, "
                f"explored={self.nodes_explored}>")


def bfs(graph, start: Tuple[int, int], goal: Tuple[int, int]) -> SearchResult:
    """
    Breadth-First Search - guarantees shortest path in unweighted graphs.
    """
    if start not in graph.graph.nodes() or goal not in graph.graph.nodes():
        return SearchResult([], float('inf'), 0, "BFS")
    
    if start == goal:
        return SearchResult([start], 0.0, 0, "BFS")
    
    queue = deque([start])
    came_from = {start: None}
    nodes_explored = 0
    
    while queue:
        current = queue.popleft()
        nodes_explored += 1
        
        if current == goal:
            # Reconstruct path
            path = []
            node = goal
            while node is not None:
                path.append(node)
                node = came_from[node]
            path.reverse()
            
            cost = len(path) - 1
            return SearchResult(path, float(cost), nodes_explored, "BFS")
        
        for neighbor in graph.get_neighbors(current):
            if neighbor not in came_from:
                came_from[neighbor] = current
                queue.append(neighbor)
    
    return SearchResult([], float('inf'), nodes_explored, "BFS")


def dfs(graph, start: Tuple[int, int], goal: Tuple[int, int], 
        max_depth: int = 1000) -> SearchResult:
    """
    Depth-First Search - explores deeply but doesn't guarantee shortest path.
    """
    if start not in graph.graph.nodes() or goal not in graph.graph.nodes():
        return SearchResult([], float('inf'), 0, "DFS")
    
    if start == goal:
        return SearchResult([start], 0.0, 0, "DFS")
    
    stack = [(start, [start], 0)]
    visited = set()
    nodes_explored = 0
    
    while stack:
        current, path, depth = stack.pop()
        
        if depth > max_depth:
            continue
        
        if current in visited:
            continue
        
        visited.add(current)
        nodes_explored += 1
        
        if current == goal:
            cost = len(path) - 1
            return SearchResult(path, float(cost), nodes_explored, "DFS")
        
        for neighbor in graph.get_neighbors(current):
            if neighbor not in visited:
                stack.append((neighbor, path + [neighbor], depth + 1))
    
    return SearchResult([], float('inf'), nodes_explored, "DFS")


def dijkstra(graph, start: Tuple[int, int], goal: Tuple[int, int]) -> SearchResult:
    """
    Dijkstra's Algorithm - finds shortest path in weighted graphs.
    """
    if start not in graph.graph.nodes() or goal not in graph.graph.nodes():
        return SearchResult([], float('inf'), 0, "Dijkstra")
    
    if start == goal:
        return SearchResult([start], 0.0, 0, "Dijkstra")
    
    # Priority queue: (cost, node)
    pq = [(0.0, start)]
    came_from = {start: None}
    cost_so_far = {start: 0.0}
    nodes_explored = 0
    
    while pq:
        current_cost, current = heapq.heappop(pq)
        nodes_explored += 1
        
        if current == goal:
            # Reconstruct path
            path = []
            node = goal
            while node is not None:
                path.append(node)
                node = came_from[node]
            path.reverse()
            
            return SearchResult(path, cost_so_far[goal], nodes_explored, "Dijkstra")
        
        # Skip if we've found a better path already
        if current_cost > cost_so_far.get(current, float('inf')):
            continue
        
        for neighbor in graph.get_neighbors(current):
            edge_weight = graph.get_edge_weight(current, neighbor)
            if edge_weight is None:
                continue
            
            new_cost = cost_so_far[current] + edge_weight
            
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                came_from[neighbor] = current
                heapq.heappush(pq, (new_cost, neighbor))
    
    return SearchResult([], float('inf'), nodes_explored, "Dijkstra")


def a_star(graph, start: Tuple[int, int], goal: Tuple[int, int],
           heuristic: str = "euclidean") -> SearchResult:
    """
    A* Search - optimal pathfinding with heuristic guidance.
    """
    if start not in graph.graph.nodes() or goal not in graph.graph.nodes():
        return SearchResult([], float('inf'), 0, "A*")
    
    if start == goal:
        return SearchResult([start], 0.0, 0, "A*")
    
    # Priority queue: (f_score, node)
    pq = [(0.0, start)]
    came_from = {start: None}
    g_score = {start: 0.0}
    nodes_explored = 0
    
    while pq:
        _, current = heapq.heappop(pq)
        nodes_explored += 1
        
        if current == goal:
            # Reconstruct path
            path = []
            node = goal
            while node is not None:
                path.append(node)
                node = came_from[node]
            path.reverse()
            
            return SearchResult(path, g_score[goal], nodes_explored, "A*")
        
        for neighbor in graph.get_neighbors(current):
            edge_weight = graph.get_edge_weight(current, neighbor)
            if edge_weight is None:
                continue
            
            tentative_g = g_score[current] + edge_weight
            
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                g_score[neighbor] = tentative_g
                h = graph.heuristic(neighbor, goal, method=heuristic)
                f_score = tentative_g + h
                came_from[neighbor] = current
                heapq.heappush(pq, (f_score, neighbor))
    
    return SearchResult([], float('inf'), nodes_explored, "A*")


def bidirectional_a_star(graph, start: Tuple[int, int], goal: Tuple[int, int],
                         heuristic: str = "euclidean") -> SearchResult:
    """
    Bidirectional A* - searches from both start and goal simultaneously.
    """
    if start not in graph.graph.nodes() or goal not in graph.graph.nodes():
        return SearchResult([], float('inf'), 0, "Bidirectional-A*")
    
    if start == goal:
        return SearchResult([start], 0.0, 0, "Bidirectional-A*")
    
    # Forward search from start
    pq_forward = [(0.0, start)]
    came_from_forward = {start: None}
    g_forward = {start: 0.0}
    
    # Backward search from goal
    pq_backward = [(0.0, goal)]
    came_from_backward = {goal: None}
    g_backward = {goal: 0.0}
    
    best_path_cost = float('inf')
    meeting_node = None
    nodes_explored = 0
    
    while pq_forward and pq_backward:
        # Forward step
        if pq_forward:
            _, current_f = heapq.heappop(pq_forward)
            nodes_explored += 1
            
            # Check if we've met the backward search
            if current_f in g_backward:
                total_cost = g_forward[current_f] + g_backward[current_f]
                if total_cost < best_path_cost:
                    best_path_cost = total_cost
                    meeting_node = current_f
            
            for neighbor in graph.get_neighbors(current_f):
                edge_weight = graph.get_edge_weight(current_f, neighbor)
                if edge_weight is None:
                    continue
                
                tentative_g = g_forward[current_f] + edge_weight
                
                if neighbor not in g_forward or tentative_g < g_forward[neighbor]:
                    g_forward[neighbor] = tentative_g
                    h = graph.heuristic(neighbor, goal, method=heuristic)
                    f_score = tentative_g + h
                    came_from_forward[neighbor] = current_f
                    heapq.heappush(pq_forward, (f_score, neighbor))
        
        # Backward step
        if pq_backward:
            _, current_b = heapq.heappop(pq_backward)
            nodes_explored += 1
            
            # Check if we've met the forward search
            if current_b in g_forward:
                total_cost = g_forward[current_b] + g_backward[current_b]
                if total_cost < best_path_cost:
                    best_path_cost = total_cost
                    meeting_node = current_b
            
            for neighbor in graph.get_neighbors(current_b):
                edge_weight = graph.get_edge_weight(current_b, neighbor)
                if edge_weight is None:
                    continue
                
                tentative_g = g_backward[current_b] + edge_weight
                
                if neighbor not in g_backward or tentative_g < g_backward[neighbor]:
                    g_backward[neighbor] = tentative_g
                    h = graph.heuristic(neighbor, start, method=heuristic)
                    f_score = tentative_g + h
                    came_from_backward[neighbor] = current_b
                    heapq.heappush(pq_backward, (f_score, neighbor))
        
        # Early termination if path found
        if meeting_node and best_path_cost < float('inf'):
            # Check if we should continue searching
            min_forward = pq_forward[0][0] if pq_forward else float('inf')
            min_backward = pq_backward[0][0] if pq_backward else float('inf')
            
            if min_forward + min_backward >= best_path_cost:
                break
    
    if meeting_node is None:
        return SearchResult([], float('inf'), nodes_explored, "Bidirectional-A*")
    
    # Reconstruct path from both directions
    path_forward = []
    node = meeting_node
    while node is not None:
        path_forward.append(node)
        node = came_from_forward[node]
    path_forward.reverse()
    
    path_backward = []
    node = came_from_backward[meeting_node]
    while node is not None:
        path_backward.append(node)
        node = came_from_backward[node]
    
    path = path_forward + path_backward
    
    return SearchResult(path, best_path_cost, nodes_explored, "Bidirectional-A*")


def compare_algorithms(graph, start: Tuple[int, int], goal: Tuple[int, int]) -> Dict[str, SearchResult]:
    """
    Run all search algorithms and compare results.
    """
    results = {}
    results["BFS"] = bfs(graph, start, goal)
    results["DFS"] = dfs(graph, start, goal)
    results["Dijkstra"] = dijkstra(graph, start, goal)
    results["A*"] = a_star(graph, start, goal)
    results["Bidirectional-A*"] = bidirectional_a_star(graph, start, goal)
    return results


def find_all_paths_bfs(graph, start: Tuple[int, int], max_distance: int = 10) -> Dict[Tuple[int, int], int]:
    """
    Find all reachable nodes within max_distance from start.
    """
    if start not in graph.graph.nodes():
        return {}
    
    distances = {start: 0}
    queue = deque([start])
    
    while queue:
        current = queue.popleft()
        current_dist = distances[current]
        
        if current_dist >= max_distance:
            continue
        
        for neighbor in graph.get_neighbors(current):
            if neighbor not in distances:
                distances[neighbor] = current_dist + 1
                queue.append(neighbor)
    
    return distances