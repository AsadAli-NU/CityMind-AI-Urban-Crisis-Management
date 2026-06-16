"""
CityMind — 20-Step Simulation Engine
======================================
Orchestrates all 5 challenge modules through the shared city graph.
Each simulation step may trigger random events (flooding), causing
real-time adaptation across all modules.
"""

import random
from challenges.layout_csp import solve_layout
from challenges.road_network_ga import build_road_network
from challenges.ambulance_placement import place_ambulances
from challenges.emergency_routing import create_emergency_router
from challenges.crime_prediction import run_crime_prediction_pipeline, deploy_police_officers


class Simulation:
    """
    20-step simulation that ties all 5 challenges together.

    Flow:
      1. Challenge 1 (CSP) generates city layout
      2. Challenge 2 (GA) builds road network
      3. Challenge 5 (ML) predicts crime risk → updates graph weights
      4. Challenge 3 (GA) places ambulances based on risk-weighted graph
      5. Each step: random flooding → Challenge 4 (A*) re-routes → 
         Challenge 3 may re-evaluate ambulance positions
    """

    def __init__(self, city_graph, total_steps=20):
        self.graph = city_graph
        self.total_steps = total_steps
        self.current_step = 0
        self.state = 'idle'  # idle, initialized, running, paused, complete
        self.router = None
        self.ambulance_positions = []
        self.police_positions = []
        self.civilian_positions = []
        self.blocked_roads = []
        self.step_log = []
        self.results = {
            'layout': None,
            'road_network': None,
            'crime_prediction': None,
            'police_deployment': None,
            'ambulance_placement': None,
            'routing': None
        }
        # Constraint parameters for live modification
        self.hospital_hops = 3
        self.powerplant_hops = 2
        # Flooding probability per step
        self.flood_probability = 0.3

    def _log(self, message):
        entry = {'step': self.current_step, 'message': message}
        self.step_log.append(entry)
        self.graph._notify('simulation_log', entry)

    # ── Phase 1: Initialize all modules ───────────────────────────────

    def initialize(self):
        """Run Challenges 1, 2, 5, 3 in sequence to set up the city."""
        self.state = 'initializing'
        self.graph.clear_layout()
        self.step_log = []
        self.blocked_roads = []
        self.current_step = 0

        # Challenge 1: City Layout (CSP)
        self._log("═══ CHALLENGE 1: Generating city layout via CSP ═══")
        self._log(f"   Constraints: hospital_hops={self.hospital_hops}, powerplant_hops={self.powerplant_hops}")
        success, conflict = solve_layout(self.graph, hospital_hops=self.hospital_hops, powerplant_hops=self.powerplant_hops)
        if success:
            self._log("✓ City layout generated successfully (all constraints satisfied)")
        else:
            self._log(f"⚠ Layout generated with conflicts: {conflict}")
        self.results['layout'] = {'success': success, 'conflict': conflict, 'hospital_hops': self.hospital_hops, 'powerplant_hops': self.powerplant_hops}

        # Challenge 2: Road Network (GA)
        self._log("═══ CHALLENGE 2: Building road network via Genetic Algorithm ═══")
        road_result = build_road_network(self.graph)
        self._log(f"✓ Road network: {road_result['edge_count']} roads, "
                  f"cost={road_result['total_cost']}, "
                  f"redundant paths={road_result['disjoint_paths_hospital_depot']}")
        self.results['road_network'] = road_result

        # Challenge 5: Crime Risk Prediction (K-Means + Decision Tree)
        self._log("═══ CHALLENGE 5: Running crime prediction pipeline ═══")
        crime_result = run_crime_prediction_pipeline(self.graph)
        self._log(f"✓ Crime prediction: accuracy={crime_result['classifier_accuracy']:.1%}, "
                  f"risk distribution={crime_result['risk_distribution']}")
        self.results['crime_prediction'] = crime_result

        # Challenge 5b: Police Officer Deployment (10 officers)
        self._log("═══ CHALLENGE 5b: Deploying 10 police officers ═══")
        police_result = deploy_police_officers(self.graph, num_officers=10)
        self.police_positions = [tuple(p) for p in police_result['positions']]
        self._log(f"✓ Police deployed: {police_result['num_officers']} officers "
                  f"(High-risk coverage: {police_result['high_risk_covered']})")
        self.results['police_deployment'] = police_result

        # Challenge 3: Ambulance Placement (GA)
        self._log("═══ CHALLENGE 3: Placing ambulances via Genetic Algorithm ═══")
        amb_result = place_ambulances(self.graph)
        self.ambulance_positions = [tuple(p) for p in amb_result['positions']]
        self._log(f"✓ Ambulances placed at {self.ambulance_positions}, "
                  f"worst-case response={amb_result['max_response_distance']}")
        self.results['ambulance_placement'] = amb_result

        # Challenge 4: Initialize Emergency Router (A*)
        self._log("═══ CHALLENGE 4: Initializing emergency router (A*) ═══")
        residential = self.graph.get_nodes_by_type('Residential')
        num_civilians = min(5, len(residential))
        self.civilian_positions = random.sample(residential, num_civilians)

        depots = self.graph.get_nodes_by_type('AmbulanceDepot')
        start = depots[0] if depots else (0, 0)

        self.router = create_emergency_router(
            self.graph, start, self.civilian_positions
        )
        self._log(f"✓ Emergency router ready. {num_civilians} civilians to rescue. "
                  f"Starting from {start}")

        self.state = 'initialized'
        self._log("═══ INITIALIZATION COMPLETE — Ready to simulate ═══")
        return self.get_state()

    # ── Simulation Step ───────────────────────────────────────────────

    def step(self):
        """
        Execute one simulation step:
          1. Maybe trigger a random flooding event
          2. Route to next civilian (A*)
          3. Maybe re-evaluate ambulance positions
        """
        if self.state not in ('initialized', 'running'):
            return self.get_state()

        self.state = 'running'
        self.current_step += 1
        self._log(f"──── STEP {self.current_step}/{self.total_steps} ────")

        # Random flooding event
        if random.random() < self.flood_probability and self.current_step > 1:
            self._trigger_flooding()

        # Route to next civilian via A*
        if self.router and not self.router.get_status()['mission_complete']:
            path = self.router.route_to_next_civilian()
            if path:
                self._log(f"A* routed team via {len(path)} nodes")
            else:
                self._log("No more civilians to rescue or all unreachable")
        else:
            self._log("All civilians already rescued")

        # Crime wave event (changes risk levels) - high probability to guarantee integration demo
        if random.random() < 0.80 and self.current_step > 1 and self.current_step % 5 == 0:
            self._trigger_crime_wave()

        # Re-evaluate ambulance placement when risk weights may have shifted
        # (steps 5, 10, 15) - but NOT on the final step to avoid pointless repositioning
        if self.current_step % 5 == 0 and self.current_step < self.total_steps:
            self._log("Re-evaluating ambulance positions due to risk changes...")
            amb_result = place_ambulances(self.graph)
            self.ambulance_positions = [tuple(p) for p in amb_result['positions']]
            self._log(f"Ambulances repositioned to {self.ambulance_positions}")
            self.results['ambulance_placement'] = amb_result

        # Check if simulation is complete - log it BEFORE re-evaluation to avoid duplicates
        if self.current_step >= self.total_steps:
            self.state = 'complete'
            self._log("═══ SIMULATION COMPLETE ═══")

        return self.get_state()

    def _trigger_flooding(self):
        """Randomly block a road to simulate flooding."""
        active_edges = self.graph.get_active_edges()
        if not active_edges:
            return

        # Pick a random active edge
        edge_key = random.choice(list(active_edges.keys()))
        positions = list(edge_key)
        pos1, pos2 = positions[0], positions[1]

        self.graph.block_road(pos1, pos2)
        self.blocked_roads.append((pos1, pos2))
        self._log(f"🌊 FLOODING: Road {pos1} → {pos2} blocked!")

        # Notify router to re-route
        if self.router:
            self.router.reroute_after_block(pos1, pos2)

    def _trigger_crime_wave(self):
        """Simulate random crime wave that increases risk in areas."""
        all_nodes = list(self.graph.nodes.keys())
        if not all_nodes:
            return

        # Pick random area and increase risk
        affected_area = random.choice(all_nodes)
        node_type = self.graph.nodes[affected_area].get('type', '')

        # Don't affect critical infrastructure
        if node_type in ('Hospital', 'School', 'AmbulanceDepot'):
            return

        self._log(f"🚨 CRIME WAVE in {affected_area}!")

        # Increase risk in affected and nearby areas
        risk_changed = False
        for pos in self.graph.nodes:
            dist = abs(pos[0] - affected_area[0]) + abs(pos[1] - affected_area[1])
            if dist <= 2:
                old_risk = self.graph.nodes[pos].get('risk_index', 0)
                # Increase risk by 0.1-0.2, cap at 0.9 (High)
                increase = random.uniform(0.1, 0.2)
                new_risk = min(0.9, old_risk + increase)
                if new_risk != old_risk:
                    self.graph.update_risk(pos, new_risk)
                    risk_changed = True

        if risk_changed:
            self._log("   Risk levels updated in affected areas")

    # ── State Retrieval ───────────────────────────────────────────────

    def get_state(self):
        """Return complete simulation state for the frontend."""
        router_status = self.router.get_status() if self.router else None

        return {
            'state': self.state,
            'current_step': self.current_step,
            'total_steps': self.total_steps,
            'graph': self.graph.to_dict(),
            'ambulance_positions': [list(p) for p in self.ambulance_positions],
            'police_positions': [list(p) for p in self.police_positions],
            'civilian_positions': [list(c) for c in self.civilian_positions],
            'blocked_roads': [
                [list(p1), list(p2)] for p1, p2 in self.blocked_roads
            ],
            'router_status': router_status,
            'results': {
                k: v for k, v in self.results.items()
                if v is not None and k != 'crime_prediction'
            },
            'crime_risk_distribution': (
                self.results['crime_prediction']['risk_distribution']
                if self.results.get('crime_prediction') else None
            ),
            'feature_importances': (
                self.results['crime_prediction']['feature_importances']
                if self.results.get('crime_prediction') else None
            ),
            'hospital_hops': self.hospital_hops,
            'powerplant_hops': self.powerplant_hops,
            'step_log': self.step_log[-50:],  # Last 50 entries
            'event_log': self.graph.event_log[-30:]  # Last 30 events
        }

    def run_all(self):
        """Run the complete simulation (all steps)."""
        if self.state == 'idle':
            self.initialize()
        while self.current_step < self.total_steps:
            self.step()
        return self.get_state()
