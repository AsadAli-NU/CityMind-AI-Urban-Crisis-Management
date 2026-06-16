"""
Challenge 1: City Layout Planning — Constraint Satisfaction Problem (CSP)
=========================================================================
Algorithm: Backtracking Search + Arc Consistency (AC-3) + MRV Heuristic

Why CSP?
  - The problem assigns location types to grid cells under hard constraints.
  - Variables = grid cells to fill, Domains = location types, Constraints = urban rules.
  - CSP with backtracking guarantees finding a valid solution if one exists.

Alternative Considered: Hill Climbing / Local Search
  - Rejected because it cannot guarantee constraint satisfaction and cannot
    reliably identify WHICH constraint causes infeasibility.

Constraints:
  C1: Industrial zones NOT adjacent (4-connected) to Schools or Hospitals
  C2: Every Residential area within ≤ 3 road hops of at least one Hospital
  C3: Power Plants within ≤ 2 road hops of at least one Industrial zone
  C4: If no valid layout → identify conflicting rule + propose min-conflict solution
"""

import random
from collections import deque


# ── Configuration ─────────────────────────────────────────────────────

FACILITY_COUNTS = {
    'Hospital': 2,
    'School': 3,
    'Industrial': 5,
    'PowerPlant': 2,
    'AmbulanceDepot': 1,
}

# ── Helper Functions ──────────────────────────────────────────────────


def _grid_neighbors(x, y, w, h):
    """Return 4-connected neighbors within grid bounds."""
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < w and 0 <= ny < h:
            yield (nx, ny)


def _bfs_min_distance(grid, start, target_type, w, h):
    """BFS shortest hop-distance from start to nearest cell of target_type."""
    visited = {start}
    queue = deque([(start, 0)])
    while queue:
        pos, dist = queue.popleft()
        if pos != start and grid.get(pos) == target_type:
            return dist
        for n in _grid_neighbors(*pos, w, h):
            if n not in visited:
                visited.add(n)
                queue.append((n, dist + 1))
    return float('inf')


# ── CSP Solver ────────────────────────────────────────────────────────


class LayoutCSP:
    """
    CSP solver for city layout planning.

    Approach:
      1. Place facility types one-by-one using Backtracking with MRV ordering.
      2. After facilities are placed, fill remaining cells as Residential.
      3. Verify global distance constraints (C2, C3).
      4. If infeasible, identify the violating constraint and run Min-Conflicts repair.
    """

    def __init__(self, width, height, facility_counts=None, hospital_hops=3, powerplant_hops=2):
        self.width = width
        self.height = height
        self.counts = facility_counts or dict(FACILITY_COUNTS)
        self.hospital_hops = hospital_hops  # C2: Residential within this many hops of Hospital
        self.powerplant_hops = powerplant_hops  # C3: PowerPlant within this many hops of Industrial
        self.grid = {}             # (x,y) -> location type
        self.conflict_info = None  # stores which constraint failed

    def _all_cells(self):
        return [(x, y) for x in range(self.width) for y in range(self.height)]

    def _available_cells(self):
        return [c for c in self._all_cells() if c not in self.grid]

    # ── AC-3 Arc Consistency ─────────────────────────────────────────

    def _ac3(self, domains):
        """
        AC-3 (Arc Consistency 3) pre-solve consistency check.

        This runs BEFORE backtracking to detect early infeasibility.
        It prunes domains by removing values that violate C1 (adjacency).
        If any domain becomes empty, we know the problem has no solution
        before wasting time on backtracking.

        Note: This is a pre-check only. The reduced domains are not passed
        to backtracking due to implementation complexity. Backtracking uses
        _check_adjacency() for forward checking instead.

        Returns True if consistent (no empty domains), False otherwise.
        """
        # Initialize queue with all arcs (constraints between variables)
        queue = deque()

        # Add arcs: each cell with its neighbors for adjacency constraint (C1)
        cells = self._all_cells()
        for cell in cells:
            for neighbor in _grid_neighbors(*cell, self.width, self.height):
                queue.append((cell, neighbor))  # arc: cell -> neighbor

        while queue:
            xi, xj = queue.popleft()

            if self._revise(domains, xi, xj):
                if not domains[xi]:  # Domain became empty
                    return False

                # Add all neighbors of xi back to queue (except xj)
                for xk in _grid_neighbors(*xi, self.width, self.height):
                    if xk != xj:
                        queue.append((xk, xi))

        return True

    def _revise(self, domains, xi, xj):
        """
        Revise domain of xi based on constraint with xj.
        Remove values from xi's domain that have no support in xj's domain.
        Returns True if domain was revised (changed).
        """
        revised = False
        to_remove = []

        for value in domains[xi]:
            # Check if there's any value in xj's domain that's consistent with this value
            has_support = False
            for other_value in domains[xj]:
                if self._is_consistent_pair(xi, value, xj, other_value):
                    has_support = True
                    break

            if not has_support:
                to_remove.append(value)
                revised = True

        for value in to_remove:
            domains[xi].remove(value)

        return revised

    def _is_consistent_pair(self, pos1, type1, pos2, type2):
        """Check if two adjacent cell types satisfy C1 constraint."""
        # C1: Industrial not adjacent to School/Hospital
        if type1 == 'Industrial' and type2 in ('School', 'Hospital'):
            return False
        if type2 == 'Industrial' and type1 in ('School', 'Hospital'):
            return False
        return True

    def _initialize_domains(self):
        """Initialize domains for all cells."""
        domains = {}
        all_types = list(self.counts.keys()) + ['Residential']  # All possible types

        for cell in self._all_cells():
            domains[cell] = all_types.copy()

        return domains

    # ── Constraint Checks ─────────────────────────────────────────────

    def _check_adjacency(self, pos, cell_type):
        """C1: Industrial not adjacent to School/Hospital (and vice-versa)."""
        for n in _grid_neighbors(*pos, self.width, self.height):
            n_type = self.grid.get(n)
            if n_type is None:
                continue
            if cell_type == 'Industrial' and n_type in ('School', 'Hospital'):
                return False
            if cell_type in ('School', 'Hospital') and n_type == 'Industrial':
                return False
        return True

    def _check_power_plant_proximity(self, pos):
        """C3: PowerPlant must be within powerplant_hops of at least one Industrial zone."""
        dist = _bfs_min_distance(self.grid, pos, 'Industrial', self.width, self.height)
        return dist <= self.powerplant_hops

    def _check_residential_hospital(self, pos):
        """C2: Residential within hospital_hops of at least one Hospital."""
        dist = _bfs_min_distance(self.grid, pos, 'Hospital', self.width, self.height)
        return dist <= self.hospital_hops

    def _verify_global_constraints(self):
        """
        After full layout, verify all distance-based constraints.
        Returns (valid, conflict_description).
        """
        # C2: Every Residential within hospital_hops of Hospital
        for pos, ptype in self.grid.items():
            if ptype == 'Residential':
                dist = _bfs_min_distance(self.grid, pos, 'Hospital', self.width, self.height)
                if dist > self.hospital_hops:
                    return False, {
                        'rule': 'C2',
                        'description': f'Residential must be within {self.hospital_hops} hops of Hospital',
                        'violating_cell': pos,
                        'nearest_hospital_hops': dist,
                        'configured_hops': self.hospital_hops
                    }

        # C3: PowerPlant within powerplant_hops of Industrial
        for pos, ptype in self.grid.items():
            if ptype == 'PowerPlant':
                dist = _bfs_min_distance(self.grid, pos, 'Industrial', self.width, self.height)
                if dist > self.powerplant_hops:
                    return False, {
                        'rule': 'C3',
                        'description': f'Power Plant must be within {self.powerplant_hops} hops of Industrial',
                        'violating_cell': pos,
                        'nearest_industrial_hops': dist,
                        'configured_hops': self.powerplant_hops
                    }

        return True, None

    # ── MRV Variable Ordering ─────────────────────────────────────────

    def _count_valid_positions(self, cell_type):
        """Count how many available cells can legally hold this type (MRV)."""
        count = 0
        for c in self._available_cells():
            if self._check_adjacency(c, cell_type):
                count += 1
        return count

    # ── Backtracking Search ───────────────────────────────────────────

    def _backtrack_place(self, facilities_to_place):
        """
        Recursive backtracking to place all facility types.
        facilities_to_place: list of (type, remaining_count) tuples.
        """
        if not facilities_to_place:
            return True

        # MRV: pick the type with fewest valid positions
        facilities_to_place.sort(key=lambda ft: self._count_valid_positions(ft[0]))

        ftype, remaining = facilities_to_place[0]
        rest = facilities_to_place[1:]

        available = self._available_cells()
        random.shuffle(available)

        # Sort candidates by strategic value
        if ftype == 'Hospital':
            # Spread hospitals apart for maximum coverage
            placed_hospitals = [p for p, t in self.grid.items() if t == 'Hospital']
            if placed_hospitals:
                available.sort(
                    key=lambda c: -min(abs(c[0]-h[0]) + abs(c[1]-h[1]) for h in placed_hospitals)
                )
        elif ftype == 'PowerPlant':
            # Prefer positions near Industrial zones
            industrials = [p for p, t in self.grid.items() if t == 'Industrial']
            if industrials:
                available.sort(
                    key=lambda c: min(abs(c[0]-i[0]) + abs(c[1]-i[1]) for i in industrials)
                )

        for cell in available:
            if not self._check_adjacency(cell, ftype):
                continue

            # Forward checking for PowerPlant
            if ftype == 'PowerPlant':
                self.grid[cell] = ftype
                if not self._check_power_plant_proximity(cell):
                    del self.grid[cell]
                    continue
            else:
                self.grid[cell] = ftype

            if remaining > 1:
                new_list = [(ftype, remaining - 1)] + rest
            else:
                new_list = list(rest)

            if self._backtrack_place(new_list):
                return True

            del self.grid[cell]

        return False

    def _propose_solution(self, conflict):
        """Suggest how to fix the conflict (Requirement C4)."""
        if conflict['rule'] == 'C2':
            return {
                'suggestion': 'Add more hospitals or reduce hospital_hops constraint',
                'reason': f"Residential at {conflict['violating_cell']} is {conflict.get('nearest_hospital_hops', 'unknown')} "
                          f"hops away, exceeds limit of {conflict.get('configured_hops', self.hospital_hops)}",
                'options': [
                    f'Reduce hospital_hops constraint (current: {self.hospital_hops})',
                    'Add more hospitals to layout',
                    'Move hospitals closer to outlying residential areas'
                ]
            }
        elif conflict['rule'] == 'C3':
            return {
                'suggestion': 'Adjust powerplant placement or reduce powerplant_hops constraint',
                'reason': f"PowerPlant at {conflict['violating_cell']} is {conflict.get('nearest_industrial_hops', 'unknown')} "
                          f"hops away, exceeds limit of {conflict.get('configured_hops', self.powerplant_hops)}",
                'options': [
                    f'Reduce powerplant_hops constraint (current: {self.powerplant_hops})',
                    'Add more industrial zones',
                    'Move power plants closer to industrial zones'
                ]
            }
        elif conflict['rule'] == 'C1':
            return {
                'suggestion': 'Separate Industrial zones from Schools/Hospitals',
                'reason': 'Industrial zone is adjacent to School or Hospital',
                'options': [
                    'Move Industrial zones away from Schools/Hospitals',
                    'Add buffer zones between Industrial and residential areas',
                    'Relocate Schools/Hospitals to safer areas'
                ]
            }
        return {'suggestion': 'Manual review required', 'reason': 'Unknown constraint violation'}

    # ── Min-Conflicts Repair ──────────────────────────────────────────

    def _min_conflicts_repair(self, max_steps=500):
        """
        Local search repair when global constraints are violated.
        Iteratively moves violating facilities to reduce conflicts.
        """
        for step in range(max_steps):
            valid, conflict = self._verify_global_constraints()
            if valid:
                return True

            if conflict['rule'] == 'C2':
                # A Residential cell is too far from Hospital.
                # Find the best place to add a hospital to cover this residential
                viol_pos = conflict['violating_cell']
                configured_hops = conflict.get('configured_hops', self.hospital_hops)

                # Strategy: Find the cell within hospital_hops of the violating residential
                # that can accommodate a hospital (not adjacent to industrial/school)
                best_hospital_pos = None
                best_score = float('inf')

                # Search within hospital_hops radius for best hospital placement
                for dx in range(-configured_hops, configured_hops + 1):
                    for dy in range(-configured_hops, configured_hops + 1):
                        if abs(dx) + abs(dy) > configured_hops:
                            continue
                        candidate = (viol_pos[0] + dx, viol_pos[1] + dy)
                        if 0 <= candidate[0] < self.width and 0 <= candidate[1] < self.height:
                            # Check if candidate can be a hospital
                            if self.grid.get(candidate) in ('Residential', 'Empty', None):
                                if self._check_adjacency(candidate, 'Hospital'):
                                    # Score: prefer converting Residential, then Empty
                                    score = 0 if self.grid.get(candidate) == 'Residential' else 1
                                    if score < best_score:
                                        best_score = score
                                        best_hospital_pos = candidate

                if best_hospital_pos:
                    self.grid[best_hospital_pos] = 'Hospital'
                    # If we converted a residential, we need to ensure the violating cell
                    # is still residential (it should be)
                    if self.grid.get(viol_pos) != 'Residential':
                        self.grid[viol_pos] = 'Residential'
                    # Verify the fix worked for this cell
                    if self._check_residential_hospital(viol_pos):
                        continue  # Move to next conflict
                else:
                    # Fallback: Relax hospital constraint slightly
                    # Try to find any valid position within hospital_hops + 1
                    for dx in range(-configured_hops-1, configured_hops + 2):
                        for dy in range(-configured_hops-1, configured_hops + 2):
                            candidate = (viol_pos[0] + dx, viol_pos[1] + dy)
                            if 0 <= candidate[0] < self.width and 0 <= candidate[1] < self.height:
                                if self.grid.get(candidate) in ('Residential', 'Empty', None):
                                    if self._check_adjacency(candidate, 'Hospital'):
                                        self.grid[candidate] = 'Hospital'
                                        break
                        else:
                            continue
                        break

            elif conflict['rule'] == 'C3':
                # PowerPlant too far from Industrial. Move it closer.
                viol_pos = conflict['violating_cell']
                configured_hops = conflict.get('configured_hops', self.powerplant_hops)
                industrials = [p for p, t in self.grid.items() if t == 'Industrial']

                if industrials:
                    # Find closest industrial zone
                    closest_ind = min(industrials,
                                      key=lambda i: abs(i[0]-viol_pos[0]) + abs(i[1]-viol_pos[1]))

                    # Find best position within powerplant_hops of this industrial
                    best_pos = None
                    best_dist = float('inf')

                    for dx in range(-configured_hops, configured_hops + 1):
                        for dy in range(-configured_hops, configured_hops + 1):
                            if abs(dx) + abs(dy) > configured_hops:
                                continue
                            candidate = (closest_ind[0] + dx, closest_ind[1] + dy)
                            if 0 <= candidate[0] < self.width and 0 <= candidate[1] < self.height:
                                if self.grid.get(candidate) in ('Residential', 'Empty', None):
                                    if self._check_adjacency(candidate, 'PowerPlant'):
                                        dist = abs(candidate[0]-closest_ind[0]) + abs(candidate[1]-closest_ind[1])
                                        if dist < best_dist:
                                            best_dist = dist
                                            best_pos = candidate

                    if best_pos:
                        self.grid[best_pos] = 'PowerPlant'
                        self.grid[viol_pos] = 'Residential'
                    else:
                        # Fallback: any valid position
                        for n in _grid_neighbors(*closest_ind, self.width, self.height):
                            if self.grid.get(n) in ('Residential', 'Empty', None):
                                self.grid[n] = 'PowerPlant'
                                self.grid[viol_pos] = 'Residential'
                                break

        return False

    # ── Main Solve Method ─────────────────────────────────────────────

    def _ensure_residential_coverage(self):
        """
        After placing facilities, fill remaining cells as Residential
        and ensure C2 constraint is satisfied by adding hospitals if needed.
        """
        # First pass: fill all empty cells as Residential
        for cell in self._all_cells():
            if cell not in self.grid:
                self.grid[cell] = 'Residential'

        # Second pass: verify and fix C2 violations proactively
        hospitals = [p for p, t in self.grid.items() if t == 'Hospital']

        for cell in self._all_cells():
            if self.grid.get(cell) == 'Residential':
                # Check if within hospital_hops of nearest hospital (single BFS call)
                min_dist = _bfs_min_distance(self.grid, cell, 'Hospital', self.width, self.height)

                if min_dist > self.hospital_hops:
                    # This residential is too far - convert a nearby cell to hospital
                    best_hospital_pos = None
                    best_dist = float('inf')

                    # Find best position within hospital_hops
                    for dx in range(-self.hospital_hops, self.hospital_hops + 1):
                        for dy in range(-self.hospital_hops, self.hospital_hops + 1):
                            if abs(dx) + abs(dy) > self.hospital_hops:
                                continue
                            candidate = (cell[0] + dx, cell[1] + dy)
                            if 0 <= candidate[0] < self.width and 0 <= candidate[1] < self.height:
                                if self.grid.get(candidate) in ('Residential', 'Empty', None):
                                    if self._check_adjacency(candidate, 'Hospital'):
                                        dist_to_violator = abs(candidate[0]-cell[0]) + abs(candidate[1]-cell[1])
                                        if dist_to_violator < best_dist:
                                            best_dist = dist_to_violator
                                            best_hospital_pos = candidate

                    if best_hospital_pos:
                        self.grid[best_hospital_pos] = 'Hospital'
                        hospitals.append(best_hospital_pos)

    def solve(self, max_attempts=50):
        """
        Run CSP solver with backtracking and constraint checking.
        Uses AC-3 as a pre-solve feasibility check, then backtracking with
        forward checking (_check_adjacency) for actual solution finding.
        Returns (success, grid_dict, conflict_info_or_None).
        """
        # AC-3 pre-check: detect infeasibility early before backtracking
        domains = self._initialize_domains()
        ac3_success = self._ac3(domains)  # Pre-check only, domains not used in backtracking
        if not ac3_success:
            # AC-3 detected inconsistency early - problem has no solution
            self.conflict_info = {
                'rule': 'AC3',
                'description': 'Arc consistency detected impossible configuration'
            }

        # Placement order: most constrained first (MRV at type level)
        facility_list = []
        for ftype in ['Hospital', 'Industrial', 'PowerPlant', 'School', 'AmbulanceDepot']:
            count = self.counts.get(ftype, 0)
            if count > 0:
                facility_list.append((ftype, count))

        for attempt in range(max_attempts):
            self.grid = {}

            if self._backtrack_place(list(facility_list)):
                # Fill remaining cells with improved residential coverage
                self._ensure_residential_coverage()

                # Verify global constraints
                valid, conflict = self._verify_global_constraints()
                if valid:
                    return True, dict(self.grid), None
                else:
                    # Attempt min-conflicts repair with more steps
                    if self._min_conflicts_repair(max_steps=1000):
                        return True, dict(self.grid), None
                    self.conflict_info = conflict
                    # Continue to next attempt if repair failed

        # All attempts failed — return best effort with conflict info
        if not self.grid:
            self.grid = {}
            for cell in self._all_cells():
                self.grid[cell] = 'Residential'
            self.conflict_info = {
                'rule': 'ALL',
                'description': 'Could not place facilities satisfying all constraints'
            }

        # Add proposed solution to conflict info
        if self.conflict_info:
            self.conflict_info['proposed_solution'] = self._propose_solution(self.conflict_info)

        return False, dict(self.grid), self.conflict_info


# ── Public API ────────────────────────────────────────────────────────


def solve_layout(city_graph, facility_counts=None, hospital_hops=3, powerplant_hops=2):
    """
    Solve the city layout CSP and apply results to the shared CityGraph.
    Args:
        city_graph: The shared CityGraph instance
        facility_counts: Optional dict of facility counts
        hospital_hops: Max hops from Residential to Hospital (default 3)
        powerplant_hops: Max hops from PowerPlant to Industrial (default 2)
    Returns:
        (success, conflict_info)
    """
    counts = facility_counts or dict(FACILITY_COUNTS)
    solver = LayoutCSP(city_graph.width, city_graph.height, counts, hospital_hops, powerplant_hops)
    success, layout, conflict = solver.solve()

    pop_density_map = {
        'Residential': random.randint(50, 200),
        'Hospital': 20,
        'School': 30,
        'Industrial': 10,
        'PowerPlant': 5,
        'AmbulanceDepot': 10,
        'Empty': 0,
    }

    for (x, y), loc_type in layout.items():
        density = pop_density_map.get(loc_type, 0)
        if loc_type == 'Residential':
            density = random.randint(50, 200)
        city_graph.set_node((x, y), loc_type, population_density=density)

    city_graph._notify('layout_complete', {
        'success': success,
        'message': 'City layout generated via CSP (Backtracking + MRV + AC-3)' if success
                   else f'Layout partial — conflict: {conflict}'
    })

    return success, conflict
