"""
Challenge 4: Emergency Routing Under Changing Conditions — A* Search
=====================================================================
Algorithm: A* Search with admissible & consistent heuristic (Manhattan distance)

Why A*?
  - A* guarantees finding the SHORTEST path when the heuristic is admissible
    (never overestimates) and consistent (satisfies triangle inequality).
  - Manhattan distance on a grid is both admissible and consistent.
  - When a road is blocked mid-journey, we simply re-run A* from the current
    position on the updated graph — the shared graph already reflects the change.

Alternative Considered: Dijkstra's Algorithm
  - Dijkstra also finds shortest paths but explores more nodes (no heuristic
    guidance). A* is strictly faster with a good heuristic. Both guarantee
    optimality, but A* is the informed search technique from the course.

Alternative Considered: BFS
  - Rejected because BFS ignores edge costs. Roads have different costs
    (standard=1.0, residential=0.8, plus risk multipliers). BFS would not
    find the truly shortest (minimum cost) path.

Constraints:
  - Must reach ALL civilians in sequence, not just the nearest
  - Real-time re-routing when environment changes (road blocked)
  - Must GUARANTEE shortest currently available path
"""

import heapq
import math


# ── A* Search Implementation ─────────────────────────────────────────


def _heuristic(pos, goal):
    """
    Manhattan distance heuristic.
    Admissible: never overestimates (minimum cost per step ≥ 0.8, but we
    use Manhattan which assumes cost = 1 per step, so h(n) ≤ actual cost
    when minimum edge cost is 0.8... Actually we need to scale.
    We use 0.8 * Manhattan to ensure admissibility since min edge cost = 0.8.
    Consistent: |h(a) - h(b)| ≤ cost(a,b) for adjacent nodes.
    """
    return 0.8 * (abs(pos[0] - goal[0]) + abs(pos[1] - goal[1]))


def a_star_search(city_graph, start, goal):
    """
    A* pathfinding on the shared city graph.
    Uses actual travel costs (including risk multipliers from Challenge 5).

    Returns:
        path: list of positions from start to goal, or None if no path exists.
        cost: total path cost, or infinity if unreachable.
        nodes_explored: number of nodes expanded (for logging).
    """
    if start == goal:
        return [start], 0, 1

    # Priority queue: (f_score, counter, position)
    counter = 0
    open_set = [(0 + _heuristic(start, goal), counter, start)]
    came_from = {}
    g_score = {start: 0}
    closed_set = set()
    nodes_explored = 0

    while open_set:
        f, _, current = heapq.heappop(open_set)

        if current in closed_set:
            continue

        closed_set.add(current)
        nodes_explored += 1

        if current == goal:
            # Reconstruct path
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path, g_score[goal], nodes_explored

        # Expand neighbors via actual road connections
        for neighbor, base_cost in city_graph.get_road_neighbors(current):
            if neighbor in closed_set:
                continue

            # Use travel cost with risk multiplier
            move_cost = city_graph.get_travel_cost(current, neighbor)
            tentative_g = g_score[current] + move_cost

            if tentative_g < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + _heuristic(neighbor, goal)
                counter += 1
                heapq.heappush(open_set, (f_score, counter, neighbor))

    return None, float('inf'), nodes_explored


# ── Multi-Civilian Emergency Router ──────────────────────────────────


class EmergencyRouter:
    """
    Routes a medical team through the city to reach all trapped civilians
    in sequence. Re-calculates route in real-time when roads are blocked.
    """

    def __init__(self, city_graph):
        self.city_graph = city_graph
        self.current_position = None
        self.civilians_remaining = []
        self.civilians_rescued = []
        self.total_path = []       # complete path taken
        self.total_cost = 0
        self.route_log = []        # log of routing decisions

    def initialize(self, start_pos, civilian_positions):
        """
        Set up the routing mission.
        start_pos: starting position of the medical team.
        civilian_positions: list of (x,y) positions of trapped civilians.
        """
        self.current_position = start_pos
        self.civilians_remaining = list(civilian_positions)
        self.civilians_rescued = []
        self.total_path = [start_pos]
        self.total_cost = 0
        self.route_log = []

        self._log(f"Mission started at {start_pos}. "
                  f"Civilians to rescue: {len(civilian_positions)}")

    def _log(self, message):
        self.route_log.append(message)
        self.city_graph._notify('routing_event', {'message': message})

    def _find_nearest_civilian(self):
        """
        Use A* to find the nearest reachable civilian (greedy ordering).
        Note: This is a greedy heuristic. For true optimal sequence,
        would need TSP-solving (computationally expensive for real-time).
        Greedy works well for real-time emergency response where
        immediate rescue matters more than theoretical optimality.
        """
        best_civilian = None
        best_cost = float('inf')
        best_path = None
        alternatives = []  # Track alternatives for logging

        for civ_pos in self.civilians_remaining:
            path, cost, _ = a_star_search(
                self.city_graph, self.current_position, civ_pos
            )
            if path:
                alternatives.append((civ_pos, cost, len(path)))
                if cost < best_cost:
                    best_cost = cost
                    best_civilian = civ_pos
                    best_path = path

        # Log decision reasoning for transparency
        if best_civilian and alternatives:
            alternatives.sort(key=lambda x: x[1])  # Sort by cost
            chosen = alternatives[0]
            self._log(f"Greedy choice: civilian at {chosen[0]} "
                     f"(cost={chosen[1]:.1f}, {chosen[2]} steps). "
                     f"Alternatives: {len(alternatives)-1} others evaluated")

        return best_civilian, best_path, best_cost

    def route_to_next_civilian(self):
        """
        Calculate and execute route to the next civilian.
        Returns the path taken, or None if no civilian is reachable.
        """
        if not self.civilians_remaining:
            self._log("All civilians rescued!")
            return None

        target, path, cost = self._find_nearest_civilian()

        if target is None:
            self._log("ERROR: No reachable civilians remaining!")
            return None

        self._log(f"Routing to civilian at {target} (cost: {cost:.2f})")

        # Execute the path
        self.total_path.extend(path[1:])  # skip current position (already there)
        self.total_cost += cost
        self.current_position = target
        self.civilians_remaining.remove(target)
        self.civilians_rescued.append(target)

        self._log(f"Civilian at {target} rescued! "
                  f"Remaining: {len(self.civilians_remaining)}")

        return path

    def reroute_after_block(self, blocked_pos1, blocked_pos2):
        """
        Called when a road is blocked mid-mission.
        The shared graph is already updated (Observer pattern).
        We just log it and the next route_to_next_civilian() will use A*
        on the updated graph automatically.
        """
        self._log(
            f"ALERT: Road {blocked_pos1}->{blocked_pos2} blocked! "
            f"Re-routing from current position {self.current_position}..."
        )

    def get_current_route_to_next(self):
        """
        Get the planned route from current position to the nearest civilian.
        Used for visualization — recalculated each time to reflect graph changes.
        """
        if not self.civilians_remaining:
            return None, None

        target, path, cost = self._find_nearest_civilian()
        return path, target

    def get_status(self):
        """Return current mission status."""
        return {
            'current_position': list(self.current_position) if self.current_position else None,
            'civilians_remaining': [list(c) for c in self.civilians_remaining],
            'civilians_rescued': [list(c) for c in self.civilians_rescued],
            'total_cost': round(self.total_cost, 2),
            'total_steps': len(self.total_path),
            'route_log': list(self.route_log),
            'mission_complete': len(self.civilians_remaining) == 0
        }


# ── Public API ────────────────────────────────────────────────────────


def create_emergency_router(city_graph, start_pos=None, civilian_positions=None):
    """
    Create and initialize an emergency router.
    If positions not provided, uses ambulance depot and random residential nodes.
    """
    router = EmergencyRouter(city_graph)

    if start_pos is None:
        depots = city_graph.get_nodes_by_type('AmbulanceDepot')
        start_pos = depots[0] if depots else (0, 0)

    if civilian_positions is None:
        import random
        residential = city_graph.get_nodes_by_type('Residential')
        num_civilians = min(5, len(residential))
        civilian_positions = random.sample(residential, num_civilians)

    router.initialize(start_pos, civilian_positions)
    return router
