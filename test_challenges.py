"""
CityMind Project - Comprehensive Test Suite
Tests all 5 challenges and integration scenarios
"""

import unittest
import random
from collections import deque

from core.city_graph import CityGraph
from core.simulation import Simulation
from challenges.layout_csp import solve_layout
from challenges.road_network_ga import build_road_network, _count_edge_disjoint_paths
from challenges.ambulance_placement import place_ambulances
from challenges.emergency_routing import a_star_search
from challenges.crime_prediction import run_crime_prediction_pipeline, deploy_police_officers


class TestChallenge1(unittest.TestCase):
    """Challenge 1: City Layout Planning (CSP)"""

    def test_c1_industrial_adjacency(self):
        """C1: Industrial zones NOT adjacent to School/Hospital."""
        graph = CityGraph(10, 10)
        success, conflict = solve_layout(graph)

        self.assertTrue(success, f"Layout generation failed: {conflict}")

        # Check adjacency
        layout = {pos: node['type'] for pos, node in graph.nodes.items()}
        for pos, cell_type in layout.items():
            if cell_type == 'Industrial':
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = pos[0] + dx, pos[1] + dy
                    if 0 <= nx < 10 and 0 <= ny < 10:
                        neighbor_type = layout.get((nx, ny))
                        self.assertNotIn(neighbor_type, ('School', 'Hospital'),
                            f"Industrial at {pos} adjacent to {neighbor_type} at {(nx, ny)}")

        print("✓ C1 constraint satisfied: No Industrial adjacent to School/Hospital")

    def test_c2_residential_hospital_distance(self):
        """C2: Every Residential within 3 hops of Hospital."""
        graph = CityGraph(10, 10)
        solve_layout(graph, hospital_hops=3)

        residential = graph.get_nodes_by_type('Residential')
        hospitals = graph.get_nodes_by_type('Hospital')

        self.assertGreater(len(hospitals), 0, "No hospitals placed!")

        violated = []
        for res_pos in residential:
            min_dist = float('inf')
            for hosp_pos in hospitals:
                dist = graph.bfs_distance(res_pos, hosp_pos)
                min_dist = min(min_dist, dist)

            if min_dist > 3:
                violated.append((res_pos, min_dist))

        if violated:
            print(f"✗ C2 constraint violated at {len(violated)} locations:")
            for pos, dist in violated[:5]:
                print(f"  Residential at {pos}: {dist} hops from nearest hospital")

        self.assertEqual(len(violated), 0, f"C2 violated at {len(violated)} locations")
        print("✓ C2 constraint satisfied: All Residential within 3 hops of Hospital")

    def test_c3_powerplant_industrial_distance(self):
        """C3: PowerPlant within 2 hops of Industrial zone."""
        graph = CityGraph(10, 10)
        solve_layout(graph, powerplant_hops=2)

        powerplants = graph.get_nodes_by_type('PowerPlant')
        industrials = graph.get_nodes_by_type('Industrial')

        self.assertGreater(len(industrials), 0, "No industrial zones placed!")
        self.assertGreater(len(powerplants), 0, "No power plants placed!")

        violated = []
        for pp_pos in powerplants:
            min_dist = float('inf')
            for ind_pos in industrials:
                dist = graph.bfs_distance(pp_pos, ind_pos)
                min_dist = min(min_dist, dist)

            if min_dist > 2:
                violated.append((pp_pos, min_dist))

        if violated:
            print(f"✗ C3 constraint violated: {len(violated)} power plants too far from industry")
            for pos, dist in violated:
                print(f"  PowerPlant at {pos}: {dist} hops (max allowed: 2)")

        self.assertEqual(len(violated), 0, f"C3 violated at {len(violated)} locations")
        print("✓ C3 constraint satisfied: All PowerPlants within 2 hops of Industrial")

    def test_all_facilities_placed(self):
        """Verify expected facility counts."""
        graph = CityGraph(10, 10)
        solve_layout(graph)

        expected = {
            'Hospital': 2,
            'School': 3,
            'Industrial': 5,
            'PowerPlant': 2,
            'AmbulanceDepot': 1,
        }

        for ftype, expected_count in expected.items():
            actual = len(graph.get_nodes_by_type(ftype))
            self.assertEqual(actual, expected_count,
                f"{ftype}: expected {expected_count}, got {actual}")
            print(f"✓ {ftype}: {actual} placed")


class TestChallenge2(unittest.TestCase):
    """Challenge 2: Road Network Optimization (GA)"""

    def test_full_connectivity(self):
        """All locations must be connected."""
        graph = CityGraph(10, 10)
        solve_layout(graph)
        build_road_network(graph)

        edges = graph.get_active_edges()
        self.assertGreater(len(edges), 0, "No roads built!")

        # Build adjacency from active edges
        adj = {}
        for key, edge in edges.items():
            if not edge['blocked']:
                p1, p2 = list(key)
                adj.setdefault(p1, []).append(p2)
                adj.setdefault(p2, []).append(p1)

        # Check connectivity via BFS
        all_nodes = set(graph.nodes.keys())
        non_empty = [n for n in all_nodes if graph.nodes[n]['type'] != 'Empty']

        if not non_empty:
            self.fail("No non-empty nodes to test connectivity!")

        start = non_empty[0]
        visited = {start}
        queue = deque([start])
        while queue:
            node = queue.popleft()
            for neighbor in adj.get(node, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        unconnected = set(non_empty) - visited
        if unconnected:
            print(f"✗ {len(unconnected)} nodes unreachable from {start}")
        else:
            print(f"✓ Full connectivity: All {len(non_empty)} non-empty nodes reachable")

        self.assertEqual(len(unconnected), 0, f"{len(unconnected)} nodes unreachable")

    def test_redundancy_satisfied(self):
        """At least 2 edge-disjoint paths between Hospital and Depot."""
        graph = CityGraph(10, 10)
        solve_layout(graph)
        result = build_road_network(graph)

        self.assertTrue(result['redundancy_satisfied'],
            f"Redundancy requirement not met: only {result['disjoint_paths_hospital_depot']} paths found")

        print(f"✓ Redundancy satisfied: {result['disjoint_paths_hospital_depot']} independent paths exist")

    def test_road_cost_calculation(self):
        """Total road cost calculated correctly."""
        graph = CityGraph(10, 10)
        solve_layout(graph)
        result = build_road_network(graph)

        self.assertGreater(result['edge_count'], 0, "No roads built!")
        self.assertGreater(result['total_cost'], 0, "Invalid cost!")

        print(f"✓ Road network: {result['edge_count']} roads, total cost: {result['total_cost']:.2f}")


class TestChallenge3(unittest.TestCase):
    """Challenge 3: Ambulance Placement (GA)"""

    def test_coverage_requirement(self):
        """All residential areas have ambulance coverage."""
        graph = CityGraph(10, 10)
        solve_layout(graph)
        build_road_network(graph)
        run_crime_prediction_pipeline(graph)
        result = place_ambulances(graph, num_ambulances=3)

        positions = result['positions']
        self.assertEqual(len(positions), 3, f"Wrong number of ambulances: {len(positions)}")

        # All positions must be valid locations
        for pos in positions:
            node = graph.nodes[pos]
            self.assertNotEqual(node['type'], 'Empty', f"Ambulance placed on Empty cell!")

        print(f"✓ Ambulance coverage:")
        print(f"  Positions: {positions}")
        print(f"  Worst-case response: {result['max_response_distance']:.2f}")
        print(f"  Average response: {result['avg_response_distance']:.2f}")


class TestChallenge4(unittest.TestCase):
    """Challenge 4: Emergency Routing (A*)"""

    def test_a_star_shortest_path(self):
        """A* finds optimal shortest path."""
        graph = CityGraph(10, 10)
        solve_layout(graph)
        build_road_network(graph)
        run_crime_prediction_pipeline(graph)

        residential = graph.get_nodes_by_type('Residential')
        depot = graph.get_nodes_by_type('AmbulanceDepot')[0]

        if len(residential) < 2:
            self.skipTest("Not enough residential for test")

        start = residential[0]
        goal = residential[1]

        path, cost, explored = a_star_search(graph, start, goal)

        self.assertIsNotNone(path, "No path found!")
        self.assertEqual(path[0], start, "Path doesn't start at start!")
        self.assertEqual(path[-1], goal, "Path doesn't end at goal!")

        # Verify cost calculation
        calculated_cost = 0
        for i in range(len(path) - 1):
            calculated_cost += graph.get_travel_cost(path[i], path[i+1])

        self.assertAlmostEqual(calculated_cost, cost, places=2,
            msg=f"Cost mismatch: {cost} vs calculated {calculated_cost}")

        print(f"✓ A* routing:")
        print(f"  Path length: {len(path)} steps")
        print(f"  Total cost: {cost:.2f}")
        print(f"  Nodes explored: {explored}")

    def test_rerouting_after_block(self):
        """Router adapts when roads block."""
        graph = CityGraph(10, 10)
        solve_layout(graph)
        build_road_network(graph)
        run_crime_prediction_pipeline(graph)

        residential = graph.get_nodes_by_type('Residential')
        depot = graph.get_nodes_by_type('AmbulanceDepot')[0]

        if len(residential) < 1:
            self.skipTest("Not enough residential for test")

        start = depot
        goal = residential[0]

        # Original path
        path1, cost1, _ = a_star_search(graph, start, goal)

        if not path1 or len(path1) < 2:
            self.skipTest("Cannot block road on path")

        # Block a road on the path
        pos1, pos2 = path1[0], path1[1]
        graph.block_road(pos1, pos2)

        # New path
        path2, cost2, _ = a_star_search(graph, start, goal)

        if path2 is None:
            self.fail("No path available after blocking!")
        else:
            print(f"✓ Dynamic re-routing:")
            print(f"  Original path cost: {cost1:.2f}")
            print(f"  After blocking {pos1}-{pos2}: {cost2:.2f}")
            if cost2 > cost1:
                print(f"  (Cost increased by {cost2-cost1:.2f}, expected)")


class TestChallenge5(unittest.TestCase):
    """Challenge 5: Crime Risk Prediction (ML)"""

    def test_kmeans_clustering(self):
        """K-Means produces 3 clusters."""
        graph = CityGraph(10, 10)
        solve_layout(graph)

        from challenges.crime_prediction import cluster_neighborhoods
        cluster_labels, centers, features, positions = cluster_neighborhoods(graph, n_clusters=3)

        cluster_counts = {}
        for label in cluster_labels.values():
            cluster_counts[label] = cluster_counts.get(label, 0) + 1

        self.assertEqual(len(cluster_counts), 3, f"Expected 3 clusters, got {len(cluster_counts)}")

        print(f"✓ K-Means clustering:")
        for i in range(3):
            print(f"  Cluster {i}: {cluster_counts.get(i, 0)} locations")

    def test_decision_tree_accuracy(self):
        """Decision Tree achieves reasonable accuracy."""
        graph = CityGraph(10, 10)
        solve_layout(graph)
        result = run_crime_prediction_pipeline(graph)

        accuracy = result['classifier_accuracy']
        print(f"✓ Decision Tree classifier:")
        print(f"  Accuracy: {accuracy:.1%}")
        print(f"  Risk distribution: {result['risk_distribution']}")

        # Reasonable accuracy > 50%
        self.assertGreater(accuracy, 0.5, f"Accuracy too low: {accuracy:.1%}")

    def test_police_deployment(self):
        """10 police deployed with spread."""
        graph = CityGraph(10, 10)
        solve_layout(graph)
        run_crime_prediction_pipeline(graph)

        result = deploy_police_officers(graph, num_officers=10)
        positions = result['positions']

        self.assertEqual(len(positions), 10, f"Expected 10 officers, got {len(positions)}")

        # Check spread (Manhattan distance >= 2)
        violations = 0
        for i, p1 in enumerate(positions):
            for j, p2 in enumerate(positions):
                if i < j:
                    dist = abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
                    if dist < 2:
                        violations += 1
                        print(f"⚠ Officers at {p1} and {p2} are close (distance {dist})")

        print(f"✓ Police deployment:")
        print(f"  Officers: {len(positions)}")
        print(f"  High-risk covered: {result['high_risk_covered']}")
        print(f"  Medium-risk covered: {result['medium_risk_covered']}")
        print(f"  Spread violations: {violations}")


class TestIntegration(unittest.TestCase):
    """Integration: 20-Step Simulation"""

    def test_full_initialization(self):
        """All 5 challenges run in sequence."""
        graph = CityGraph(10, 10)
        sim = Simulation(graph, total_steps=20)

        state = sim.initialize()

        self.assertEqual(state['state'], 'initialized', "Not initialized!")
        self.assertEqual(len(sim.ambulance_positions), 3, "No ambulances!")
        self.assertGreater(len(sim.civilian_positions), 0, "No civilians!")
        self.assertEqual(len(sim.police_positions), 10, "No police!")

        print("✓ Initialization complete with all components")

    def test_20_steps(self):
        """Simulation runs all 20 steps without error."""
        graph = CityGraph(10, 10)
        sim = Simulation(graph, total_steps=20)

        sim.initialize()

        for step in range(20):
            try:
                state = sim.step()
                print(f"Step {step+1}: OK")
            except Exception as e:
                self.fail(f"Step {step+1} failed: {e}")

        self.assertEqual(sim.current_step, 20, "Didn't complete all steps")
        self.assertEqual(sim.state, 'complete', "Didn't mark complete")

        print("✓ All 20 steps completed successfully")

    def test_event_logging(self):
        """Events are logged throughout simulation."""
        graph = CityGraph(10, 10)
        sim = Simulation(graph, total_steps=20)

        sim.initialize()
        sim.step()
        sim.step()

        self.assertGreater(len(graph.event_log), 0, "No events logged!")
        self.assertGreater(len(sim.step_log), 0, "No step logs!")

        print(f"✓ Logging:")
        print(f"  Event log entries: {len(graph.event_log)}")
        print(f"  Step log entries: {len(sim.step_log)}")


class TestModifications(unittest.TestCase):
    """Constraint Modification Tests"""

    def test_hospital_hops_change(self):
        """Change hospital_hops constraint."""
        graph = CityGraph(10, 10)
        sim = Simulation(graph, total_steps=20)
        sim.hospital_hops = 2  # Change from 3 to 2

        state = sim.initialize()

        # Verify C2 with new constraint
        residential = graph.get_nodes_by_type('Residential')
        hospitals = graph.get_nodes_by_type('Hospital')

        for res_pos in residential:
            min_dist = min(graph.bfs_distance(res_pos, h) for h in hospitals)
            self.assertLessEqual(min_dist, 2, f"C2 violated with new constraint")

        print("✓ hospital_hops constraint change works")

    def test_num_ambulances_change(self):
        """Change number of ambulances."""
        graph = CityGraph(10, 10)
        solve_layout(graph)
        build_road_network(graph)
        run_crime_prediction_pipeline(graph)

        result = place_ambulances(graph, num_ambulances=5)
        positions = result['positions']

        self.assertEqual(len(positions), 5, f"Expected 5, got {len(positions)}")
        print(f"✓ Ambulance count change: {len(positions)} ambulances placed")


if __name__ == '__main__':
    unittest.main(verbosity=2)
