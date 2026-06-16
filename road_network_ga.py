"""
Challenge 2: Road Network Optimization — Genetic Algorithm (GA)
================================================================
Algorithm: Genetic Algorithm with tournament selection, crossover, mutation

Why GA?
  - The search space is exponential: 2^E possible edge subsets (E = number of
    possible roads). Exhaustive search is infeasible for even a 10×10 grid.
  - GA explores diverse solutions via population-based search and evolves
    toward near-optimal networks satisfying the redundancy constraint.

Alternative Considered: Minimum Spanning Tree (Kruskal/Prim)
  - Rejected because MST produces exactly ONE path between any pair of nodes.
    The project requires at least 2 INDEPENDENT paths between Hospital and
    Ambulance Depot. MST cannot satisfy this constraint by definition.

Constraints:
  - All locations must be connected
  - At least 2 edge-disjoint paths between Primary Hospital and Ambulance Depot
  - Minimize total road cost
  - Road costs: standard = 1.0, through residential = 0.8
"""

import random
import copy
from collections import deque


# ── GA Configuration ──────────────────────────────────────────────────

POPULATION_SIZE = 40
GENERATIONS = 60
CROSSOVER_RATE = 0.85
MUTATION_RATE = 0.15
TOURNAMENT_SIZE = 5
ELITISM_COUNT = 4


# ── Helper Functions ──────────────────────────────────────────────────


def _get_possible_edges(city_graph):
    """Generate all possible edges between adjacent grid cells."""
    from core.city_graph import CityGraph
    edges = []
    for x in range(city_graph.width):
        for y in range(city_graph.height):
            for nx, ny in city_graph.get_grid_neighbors(x, y):
                if (nx, ny) > (x, y):  # avoid duplicates
                    pos1, pos2 = (x, y), (nx, ny)
                    # Use centralized cost calculation
                    t1 = city_graph.nodes[pos1]['type']
                    t2 = city_graph.nodes[pos2]['type']
                    cost = CityGraph.calculate_edge_cost(t1, t2)
                    edges.append((pos1, pos2, cost))
    return edges


def _build_adjacency(edges, active_bits):
    """Build adjacency list from active edges."""
    adj = {}
    for i, (p1, p2, cost) in enumerate(edges):
        if active_bits[i]:
            adj.setdefault(p1, []).append(p2)
            adj.setdefault(p2, []).append(p1)
    return adj


def _is_connected(adj, all_nodes):
    """BFS to check if all nodes are reachable."""
    if not all_nodes:
        return True
    # Check for isolated nodes (nodes with no edges at all)
    for node in all_nodes:
        if node not in adj or not adj[node]:
            return False
    start = next(iter(all_nodes))
    visited = {start}
    queue = deque([start])
    while queue:
        node = queue.popleft()
        for neighbor in adj.get(node, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return len(visited) == len(all_nodes)


def _count_edge_disjoint_paths(edges, active_bits, source, target):
    """
    Find number of edge-disjoint paths using iterative max-flow (Ford-Fulkerson).
    For undirected edge {u,v}, we add two directed edges each with capacity 1.
    The residual backward edge gets capacity 0 initially to prevent double-counting.
    """
    # Build capacity graph for undirected edges
    # For edge {u,v}: capacity u→v = 1, v→u = 1 (both directions usable)
    # Residual backward edges start at 0
    capacity = {}
    for i, (p1, p2, _) in enumerate(edges):
        if active_bits[i]:
            # Both directions have capacity 1 (undirected edge can be traversed either way)
            capacity.setdefault(p1, {})[p2] = capacity.get(p1, {}).get(p2, 0) + 1
            capacity.setdefault(p2, {})[p1] = capacity.get(p2, {}).get(p1, 0) + 1

    def bfs_find_path(cap, s, t):
        visited = {s}
        queue = deque([(s, [s])])
        while queue:
            node, path = queue.popleft()
            for neighbor, c in cap.get(node, {}).items():
                if neighbor not in visited and c > 0:
                    if neighbor == t:
                        return path + [t]
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        return None

    flow = 0
    while True:
        path = bfs_find_path(capacity, source, target)
        if path is None:
            break
        flow += 1
        for j in range(len(path) - 1):
            u, v = path[j], path[j + 1]
            capacity[u][v] -= 1
            capacity.setdefault(v, {})[u] = capacity.get(v, {}).get(u, 0) + 1
    return flow


# ── Genetic Algorithm ─────────────────────────────────────────────────


class RoadNetworkGA:
    """
    GA to find optimal road network.
    Chromosome: binary string, one bit per possible edge (1 = build, 0 = skip).
    """

    def __init__(self, city_graph):
        self.city_graph = city_graph
        self.possible_edges = _get_possible_edges(city_graph)
        self.num_edges = len(self.possible_edges)
        self.all_nodes = set(city_graph.nodes.keys())

        # Find primary hospital and ambulance depot
        hospitals = city_graph.get_nodes_by_type('Hospital')
        depots = city_graph.get_nodes_by_type('AmbulanceDepot')
        self.hospital = hospitals[0] if hospitals else None
        self.depot = depots[0] if depots else None

    def _random_chromosome(self):
        """Generate random chromosome biased toward including ~60% edges."""
        return [random.random() < 0.6 for _ in range(self.num_edges)]

    def _fitness(self, chromosome):
        """
        Fitness function:
          - Heavily penalize disconnected networks
          - Penalize lack of 2 edge-disjoint paths (Hospital ↔ Depot)
          - Minimize total road cost
        """
        adj = _build_adjacency(self.possible_edges, chromosome)

        # Penalty for disconnection
        if not _is_connected(adj, self.all_nodes):
            return -10000

        # Calculate total cost
        total_cost = sum(
            self.possible_edges[i][2]
            for i in range(self.num_edges)
            if chromosome[i]
        )

        # Check redundancy: ≥ 2 edge-disjoint paths
        redundancy_penalty = 0
        if self.hospital and self.depot:
            disjoint = _count_edge_disjoint_paths(
                self.possible_edges, chromosome, self.hospital, self.depot
            )
            if disjoint < 2:
                redundancy_penalty = 5000

        # Fitness = negative cost (lower cost = higher fitness)
        return -(total_cost + redundancy_penalty)

    def _tournament_select(self, population, fitnesses):
        """Tournament selection."""
        contestants = random.sample(
            list(zip(population, fitnesses)), min(TOURNAMENT_SIZE, len(population))
        )
        return max(contestants, key=lambda x: x[1])[0]

    def _crossover(self, parent1, parent2):
        """Two-point crossover."""
        if random.random() > CROSSOVER_RATE:
            return list(parent1), list(parent2)
        pt1 = random.randint(0, self.num_edges - 1)
        pt2 = random.randint(pt1, self.num_edges - 1)
        child1 = parent1[:pt1] + parent2[pt1:pt2] + parent1[pt2:]
        child2 = parent2[:pt1] + parent1[pt1:pt2] + parent2[pt2:]
        return child1, child2

    def _mutate(self, chromosome):
        """Bit-flip mutation."""
        result = list(chromosome)
        for i in range(len(result)):
            if random.random() < MUTATION_RATE:
                result[i] = not result[i]
        return result

    def evolve(self):
        """
        Run the GA for GENERATIONS iterations.
        Returns the best chromosome (edge selection).
        """
        # Initialize population
        population = [self._random_chromosome() for _ in range(POPULATION_SIZE)]
        best_chromosome = None
        best_fitness = float('-inf')

        for gen in range(GENERATIONS):
            fitnesses = [self._fitness(ch) for ch in population]

            # Track best
            gen_best_idx = max(range(len(fitnesses)), key=lambda i: fitnesses[i])
            if fitnesses[gen_best_idx] > best_fitness:
                best_fitness = fitnesses[gen_best_idx]
                best_chromosome = list(population[gen_best_idx])

            # Elitism: carry top individuals (only connected ones)
            sorted_pop = sorted(zip(population, fitnesses), key=lambda x: -x[1])
            new_population = []
            for ch, fit in sorted_pop:
                if len(new_population) >= ELITISM_COUNT:
                    break
                # Only add to elitism if connected (fitness > -10000)
                if fit > -10000:
                    new_population.append(list(ch))

            # Fill rest via selection + crossover + mutation
            while len(new_population) < POPULATION_SIZE:
                p1 = self._tournament_select(population, fitnesses)
                p2 = self._tournament_select(population, fitnesses)
                c1, c2 = self._crossover(p1, p2)
                new_population.append(self._mutate(c1))
                if len(new_population) < POPULATION_SIZE:
                    new_population.append(self._mutate(c2))

            population = new_population

        return best_chromosome


# ── Public API ────────────────────────────────────────────────────────


def build_road_network(city_graph):
    """
    Run GA to find optimal road network and apply it to the shared CityGraph.
    Returns dict with stats about the solution.
    """
    ga = RoadNetworkGA(city_graph)
    best = ga.evolve()

    # Clear existing edges and apply best solution
    city_graph.clear_edges()

    total_cost = 0
    edge_count = 0
    for i, (p1, p2, cost) in enumerate(ga.possible_edges):
        if best[i]:
            city_graph.add_edge(p1, p2, cost)
            total_cost += cost
            edge_count += 1

    # Verify redundancy
    disjoint_paths = _count_edge_disjoint_paths(
        ga.possible_edges, best, ga.hospital, ga.depot
    )

    result = {
        'total_cost': round(total_cost, 2),
        'edge_count': edge_count,
        'disjoint_paths_hospital_depot': disjoint_paths,
        'redundancy_satisfied': disjoint_paths >= 2
    }

    city_graph._notify('road_network_complete', {
        'message': f'Road network built via GA: {edge_count} roads, cost={total_cost:.1f}, '
                   f'redundant paths={disjoint_paths}',
        **result
    })

    return result
