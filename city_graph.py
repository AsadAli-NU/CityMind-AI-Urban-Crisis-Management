"""
CityMind - Shared City Graph (Single Source of Truth)
=====================================================
Observer-pattern graph ensuring all modules share the same state.
Any change (road block, risk update) is instantly visible everywhere.
"""

import math
from collections import deque


class CityGraph:
    """
    Grid-based city graph. Nodes = locations, Edges = roads.
    Implements the Observer pattern so every module sees changes immediately.
    """

    LOCATION_TYPES = [
        'Residential', 'Hospital', 'School',
        'Industrial', 'PowerPlant', 'AmbulanceDepot'
    ]

    @staticmethod
    def calculate_edge_cost(node1_type, node2_type):
        """
        Calculate road cost based on location types.
        Residential roads cost 0.8, others cost 1.0.
        """
        if node1_type == 'Residential' or node2_type == 'Residential':
            return 0.8
        return 1.0

    def __init__(self, width=10, height=10):
        self.width = width
        self.height = height
        self.nodes = {}       # (x, y) -> dict with type, population_density, risk_index, accessible
        self.edges = {}       # frozenset({(x1,y1),(x2,y2)}) -> dict with cost, blocked
        self._adj = {}        # adjacency cache: pos -> [(neighbor, cost), ...]
        self.observers = []   # list of callback functions
        self.event_log = []   # records every event for the UI

        # Initialize empty grid
        for x in range(width):
            for y in range(height):
                self.nodes[(x, y)] = {
                    'type': 'Empty',
                    'population_density': 0,
                    'risk_index': 0.0,
                    'accessible': True
                }

    # ── Observer Pattern ──────────────────────────────────────────────

    def add_observer(self, callback):
        """Register a callback: callback(event_type, data)"""
        self.observers.append(callback)

    def _notify(self, event_type, data):
        """Push change to all observers instantly."""
        self.event_log.append({'event': event_type, **data})
        for cb in self.observers:
            try:
                cb(event_type, data)
            except Exception:
                pass

    # ── Node Operations ───────────────────────────────────────────────

    def set_node(self, pos, node_type, population_density=0, risk_index=0.0):
        """Assign a location type and properties to a grid cell."""
        self.nodes[pos] = {
            'type': node_type,
            'population_density': population_density,
            'risk_index': risk_index,
            'accessible': True
        }
        self._notify('node_update', {'pos': pos, 'node': dict(self.nodes[pos])})

    def update_risk(self, pos, new_risk):
        """Update risk index for a node – used by Challenge 5."""
        if pos in self.nodes:
            self.nodes[pos]['risk_index'] = new_risk
            self._notify('risk_update', {'pos': pos, 'risk': new_risk})

    def get_node(self, pos):
        return self.nodes.get(pos)

    def get_nodes_by_type(self, node_type):
        """Return all positions of a given type."""
        return [p for p, n in self.nodes.items() if n['type'] == node_type]

    # ── Edge Operations ───────────────────────────────────────────────

    def _edge_key(self, pos1, pos2):
        return frozenset({pos1, pos2})

    def _rebuild_adj(self):
        """Rebuild adjacency cache from edges (called on any edge change)."""
        self._adj = {}
        for key, edge in self.edges.items():
            if edge['blocked']:
                continue
            p1, p2 = list(key)
            self._adj.setdefault(p1, []).append((p2, edge['cost']))
            self._adj.setdefault(p2, []).append((p1, edge['cost']))

    def add_edge(self, pos1, pos2, cost=1.0):
        """Build a road between two adjacent nodes."""
        key = self._edge_key(pos1, pos2)
        self.edges[key] = {'cost': cost, 'blocked': False}
        # Rebuild adjacency cache for consistency (handles all edge states)
        self._rebuild_adj()
        self._notify('edge_add', {'pos1': pos1, 'pos2': pos2, 'cost': cost})

    def remove_edge(self, pos1, pos2):
        key = self._edge_key(pos1, pos2)
        if key in self.edges:
            del self.edges[key]
            self._rebuild_adj()
            self._notify('edge_remove', {'pos1': pos1, 'pos2': pos2})

    def block_road(self, pos1, pos2):
        """Block a road (flooding / accident). Propagates to ALL modules."""
        key = self._edge_key(pos1, pos2)
        if key in self.edges:
            self.edges[key]['blocked'] = True
            self._rebuild_adj()
            self._notify('road_blocked', {
                'pos1': pos1, 'pos2': pos2,
                'message': f"Road {pos1}->{pos2} BLOCKED (flooding)"
            })

    def unblock_road(self, pos1, pos2):
        key = self._edge_key(pos1, pos2)
        if key in self.edges:
            self.edges[key]['blocked'] = False
            self._rebuild_adj()
            self._notify('road_unblocked', {'pos1': pos1, 'pos2': pos2})

    def get_edge(self, pos1, pos2):
        return self.edges.get(self._edge_key(pos1, pos2))

    def get_active_edges(self):
        """Return only non-blocked edges."""
        return {k: v for k, v in self.edges.items() if not v['blocked']}

    # ── Graph Queries ─────────────────────────────────────────────────

    def get_grid_neighbors(self, x, y):
        """4-connected grid neighbors (up/down/left/right)."""
        result = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.width and 0 <= ny < self.height:
                result.append((nx, ny))
        return result

    def get_road_neighbors(self, pos):
        """Neighbors reachable via non-blocked roads. O(1) via adjacency cache."""
        return self._adj.get(pos, [])

    def get_travel_cost(self, pos1, pos2):
        """Get effective travel cost including risk multiplier."""
        edge = self.get_edge(pos1, pos2)
        if edge is None or edge['blocked']:
            return float('inf')

        base_cost = edge['cost']
        # Risk multiplier from Challenge 5
        dest_node = self.nodes.get(pos2, {})
        risk = dest_node.get('risk_index', 0.0)
        # High risk (>0.7) adds 50% cost, Medium (>0.3) adds 20%
        if risk > 0.7:
            multiplier = 1.5
        elif risk > 0.3:
            multiplier = 1.2
        else:
            multiplier = 1.0

        return base_cost * multiplier

    def bfs_distance(self, start, end):
        """BFS shortest hop distance on the grid (ignoring roads)."""
        if start == end:
            return 0
        visited = {start}
        queue = deque([(start, 0)])
        while queue:
            pos, dist = queue.popleft()
            for n in self.get_grid_neighbors(*pos):
                if n == end:
                    return dist + 1
                if n not in visited:
                    visited.add(n)
                    queue.append((n, dist + 1))
        return float('inf')

    def bfs_nearest(self, start, target_type):
        """BFS to find shortest hop distance to nearest node of target_type."""
        visited = {start}
        queue = deque([(start, 0)])
        while queue:
            pos, dist = queue.popleft()
            if pos != start and self.nodes.get(pos, {}).get('type') == target_type:
                return dist
            for n in self.get_grid_neighbors(*pos):
                if n not in visited:
                    visited.add(n)
                    queue.append((n, dist + 1))
        return float('inf')

    @staticmethod
    def manhattan_distance(pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    @staticmethod
    def euclidean_distance(pos1, pos2):
        return math.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)

    # ── Serialization ─────────────────────────────────────────────────

    def to_dict(self):
        """Serialize entire graph state for the frontend."""
        nodes_list = []
        for (x, y), data in self.nodes.items():
            nodes_list.append({
                'x': x, 'y': y,
                'type': data['type'],
                'population_density': data['population_density'],
                'risk_index': data['risk_index'],
                'accessible': data['accessible']
            })

        edges_list = []
        for key, data in self.edges.items():
            positions = list(key)
            edges_list.append({
                'from': list(positions[0]),
                'to': list(positions[1]),
                'cost': data['cost'],
                'blocked': data['blocked']
            })

        return {
            'width': self.width,
            'height': self.height,
            'nodes': nodes_list,
            'edges': edges_list
        }

    def clear_layout(self):
        """Reset all nodes to Empty."""
        for pos in self.nodes:
            self.nodes[pos] = {
                'type': 'Empty',
                'population_density': 0,
                'risk_index': 0.0,
                'accessible': True
            }
        self.edges.clear()
        self._adj = {}
        self.event_log.clear()

    def clear_edges(self):
        """Remove all roads."""
        self.edges.clear()
        self._adj = {}
