"""
Challenge 3: Ambulance Placement — Genetic Algorithm (GA)
==========================================================
Algorithm: Genetic Algorithm with minimax fitness objective

Why GA?
  - The number of possible placements is C(N, 3) where N = grid size.
    For 100 cells, that's 161,700 combinations — too large for brute force,
    but the landscape has many local optima, so simple hill climbing fails.
  - GA's population-based search with crossover escapes local optima effectively.

Alternative Considered: Hill Climbing with random restarts
  - Rejected because the minimax objective creates a rugged fitness landscape
    with many plateaus. GA's population diversity handles this better.

Objective: Minimize the WORST-CASE response time (minimax).
  - Place 3 ambulances so the maximum distance from any residential node
    to its nearest ambulance is minimized.
  - Must be re-evaluated when risk weights change (from Challenge 5).
"""

import random
import heapq
from collections import defaultdict


# ── GA Configuration ──────────────────────────────────────────────────

POPULATION_SIZE = 50
GENERATIONS = 40
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.2
TOURNAMENT_SIZE = 5
ELITISM_COUNT = 4
NUM_AMBULANCES = 3


# ── Dijkstra for Accurate Distance ───────────────────────────────────


def _dijkstra_from(city_graph, source):
    """
    Dijkstra's algorithm from a source node using actual road network.
    Returns dict of {node: shortest_distance}.
    Uses travel cost which includes risk multipliers from Challenge 5.
    """
    dist = {source: 0}
    heap = [(0, source)]
    visited = set()

    while heap:
        d, node = heapq.heappop(heap)
        if node in visited:
            continue
        visited.add(node)

        for neighbor, base_cost in city_graph.get_road_neighbors(node):
            # Use effective cost (includes risk multiplier)
            cost = city_graph.get_travel_cost(node, neighbor)
            new_dist = d + cost
            if new_dist < dist.get(neighbor, float('inf')):
                dist[neighbor] = new_dist
                heapq.heappush(heap, (new_dist, neighbor))

    return dist


# ── Genetic Algorithm ─────────────────────────────────────────────────


class AmbulancePlacementGA:
    """
    GA to optimally place ambulances.
    Chromosome: list of NUM_AMBULANCES node positions [(x1,y1), (x2,y2), (x3,y3)].
    Fitness: -(maximum distance from any residential node to nearest ambulance).
    """

    def __init__(self, city_graph, num_ambulances=NUM_AMBULANCES):
        self.city_graph = city_graph
        self.num_ambulances = num_ambulances

        # Valid placement positions (any non-empty node with road access)
        self.valid_positions = [
            pos for pos, node in city_graph.nodes.items()
            if node['type'] != 'Empty' and city_graph.get_road_neighbors(pos)
        ]

        # Residential nodes (the ones we need to cover)
        self.residential = [
            pos for pos, node in city_graph.nodes.items()
            if node['type'] == 'Residential'
        ]

    def _random_chromosome(self):
        """Generate random ambulance placement."""
        return random.sample(self.valid_positions, min(self.num_ambulances, len(self.valid_positions)))

    def _fitness(self, chromosome):
        """
        Minimax fitness: minimize the worst-case response time.
        Fitness = -(max distance from any residential node to nearest ambulance).
        Higher is better (less negative = closer coverage).
        """
        if not chromosome or not self.residential:
            return -float('inf')

        # Compute Dijkstra distances from each ambulance position
        ambulance_dists = []
        for amb_pos in chromosome:
            dists = _dijkstra_from(self.city_graph, amb_pos)
            ambulance_dists.append(dists)

        # For each residential node, find distance to nearest ambulance
        max_min_distance = 0
        total_distance = 0

        for res_pos in self.residential:
            min_dist = float('inf')
            for dists in ambulance_dists:
                d = dists.get(res_pos, float('inf'))
                if d < min_dist:
                    min_dist = d
            if min_dist == float('inf'):
                return -10000  # Unreachable residential area
            max_min_distance = max(max_min_distance, min_dist)
            total_distance += min_dist

        # Primary: minimize worst case. Secondary: minimize average.
        avg_distance = total_distance / len(self.residential) if self.residential else 0
        return -(max_min_distance * 100 + avg_distance)

    def _tournament_select(self, population, fitnesses):
        contestants = random.sample(
            list(zip(population, fitnesses)), min(TOURNAMENT_SIZE, len(population))
        )
        return max(contestants, key=lambda x: x[1])[0]

    def _crossover(self, parent1, parent2):
        """Uniform crossover: each ambulance position taken from either parent."""
        if random.random() > CROSSOVER_RATE:
            return list(parent1), list(parent2)

        child1, child2 = [], []
        for i in range(self.num_ambulances):
            if random.random() < 0.5:
                child1.append(parent1[i])
                child2.append(parent2[i])
            else:
                child1.append(parent2[i])
                child2.append(parent1[i])

        # Remove duplicates
        child1 = self._fix_duplicates(child1)
        child2 = self._fix_duplicates(child2)
        return child1, child2

    def _fix_duplicates(self, chromosome):
        """Replace duplicate positions with random valid ones."""
        seen = set()
        result = []
        for pos in chromosome:
            if pos in seen:
                available = [p for p in self.valid_positions if p not in seen]
                if available:
                    pos = random.choice(available)
                else:
                    continue
            seen.add(pos)
            result.append(pos)
        return result

    def _mutate(self, chromosome):
        """Move one random ambulance to a new random position."""
        result = list(chromosome)
        if random.random() < MUTATION_RATE:
            idx = random.randint(0, len(result) - 1)
            used = set(result)
            available = [p for p in self.valid_positions if p not in used]
            if available:
                result[idx] = random.choice(available)
        return result

    def _local_hill_climbing(self, initial_positions, max_iterations=100):
        """
        Simple 1-opt local search: try moving each ambulance to nearby positions.
        Keep move if fitness improves.
        """
        current = list(initial_positions)
        current_fitness = self._fitness(current)

        improved = True
        iteration = 0

        while improved and iteration < max_iterations:
            improved = False
            iteration += 1

            # Try moving each ambulance
            for amb_idx in range(len(current)):
                current_pos = current[amb_idx]

                # Try swapping with positions in nearby grid locations
                nearby_positions = []
                x, y = current_pos
                for dx in [-2, -1, 0, 1, 2]:
                    for dy in [-2, -1, 0, 1, 2]:
                        new_x, new_y = x + dx, y + dy
                        if 0 <= new_x < self.city_graph.width and 0 <= new_y < self.city_graph.height:
                            neighbor = (new_x, new_y)
                            if neighbor not in current:  # Don't place 2 ambulances at same spot
                                nearby_positions.append(neighbor)

                # Try each nearby position
                for new_pos in nearby_positions:
                    # Create new configuration
                    test_config = list(current)
                    test_config[amb_idx] = new_pos

                    test_fitness = self._fitness(test_config)
                    if test_fitness > current_fitness:  # Improvement found
                        current = test_config
                        current_fitness = test_fitness
                        improved = True
                        break  # Found improvement, restart search

                if improved:
                    break

        return current

    def evolve(self):
        """Run GA with local hill-climbing refinement."""
        if len(self.valid_positions) < self.num_ambulances:
            return self.valid_positions[:self.num_ambulances]

        population = [self._random_chromosome() for _ in range(POPULATION_SIZE)]
        best_chromosome = None
        best_fitness = float('-inf')

        for gen in range(GENERATIONS):
            fitnesses = [self._fitness(ch) for ch in population]

            gen_best_idx = max(range(len(fitnesses)), key=lambda i: fitnesses[i])
            if fitnesses[gen_best_idx] > best_fitness:
                best_fitness = fitnesses[gen_best_idx]
                best_chromosome = list(population[gen_best_idx])

            sorted_pop = sorted(zip(population, fitnesses), key=lambda x: -x[1])
            new_population = [list(ch) for ch, _ in sorted_pop[:ELITISM_COUNT]]

            while len(new_population) < POPULATION_SIZE:
                p1 = self._tournament_select(population, fitnesses)
                p2 = self._tournament_select(population, fitnesses)
                c1, c2 = self._crossover(p1, p2)
                new_population.append(self._mutate(c1))
                if len(new_population) < POPULATION_SIZE:
                    new_population.append(self._mutate(c2))

            population = new_population

        # NEW: Local hill climbing refinement
        best_chromosome = self._local_hill_climbing(best_chromosome)

        return best_chromosome


# ── Public API ────────────────────────────────────────────────────────


def place_ambulances(city_graph, num_ambulances=NUM_AMBULANCES):
    """
    Run GA to find optimal ambulance positions and return results.
    Returns dict with positions and coverage stats.
    """
    ga = AmbulancePlacementGA(city_graph, num_ambulances)
    positions = ga.evolve()

    if not positions:
        return {'positions': [], 'max_response_distance': float('inf')}

    # Calculate coverage statistics
    ambulance_dists = [_dijkstra_from(city_graph, p) for p in positions]
    residential = city_graph.get_nodes_by_type('Residential')

    max_response = 0
    total_response = 0
    coverage = {}  # ambulance_idx -> list of covered residential nodes

    for i in range(len(positions)):
        coverage[i] = []

    for res_pos in residential:
        min_dist = float('inf')
        nearest_amb = 0
        for i, dists in enumerate(ambulance_dists):
            d = dists.get(res_pos, float('inf'))
            if d < min_dist:
                min_dist = d
                nearest_amb = i
        max_response = max(max_response, min_dist)
        total_response += min_dist
        coverage[nearest_amb].append(res_pos)

    avg_response = total_response / len(residential) if residential else 0

    result = {
        'positions': [list(p) for p in positions],
        'max_response_distance': round(max_response, 2),
        'avg_response_distance': round(avg_response, 2),
        'coverage_per_ambulance': {i: len(v) for i, v in coverage.items()},
        'num_ambulances': len(positions)
    }

    # Note: Logging is handled by the caller (simulation) to ensure correct step timing
    city_graph._notify('ambulance_placement_complete', {
        'positions': result['positions'],
        'max_response_distance': result['max_response_distance'],
        'avg_response_distance': result['avg_response_distance'],
        'num_ambulances': result['num_ambulances']
    })

    return result
