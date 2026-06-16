"""
CityMind — Flask Web Application
==================================
REST API serving the simulation engine and real-time city visualization.
"""

from flask import Flask, render_template, request
from core.city_graph import CityGraph
from core.simulation import Simulation
import json
import math

app = Flask(__name__)


def clean_infinity(obj):
    """
    Recursively clean data structure by converting Infinity, -Infinity, and NaN to None.
    This is necessary because JSON spec doesn't support these float values.
    """
    if isinstance(obj, float):
        if math.isinf(obj) or math.isnan(obj):
            return None
        return obj
    elif isinstance(obj, dict):
        return {k: clean_infinity(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_infinity(item) for item in obj]
    elif isinstance(obj, tuple):
        return [clean_infinity(item) for item in obj]
    return obj


def safe_jsonify(data):
    """Return JSON response with Infinity/NaN values cleaned."""
    cleaned_data = clean_infinity(data)
    return app.response_class(
        response=json.dumps(cleaned_data),
        status=200,
        mimetype='application/json'
    )

# ── Global State ──────────────────────────────────────────────────────
city_graph = CityGraph(width=10, height=10)
simulation = Simulation(city_graph, total_steps=20)


# ── Routes ────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/state')
def get_state():
    """Get current simulation state."""
    return safe_jsonify(simulation.get_state())


@app.route('/api/initialize', methods=['POST'])
def initialize():
    """Initialize the simulation (run Challenges 1, 2, 5, 3)."""
    global city_graph, simulation
    city_graph = CityGraph(width=10, height=10)
    simulation = Simulation(city_graph, total_steps=20)
    state = simulation.initialize()
    return safe_jsonify(state)


@app.route('/api/step', methods=['POST'])
def step():
    """Execute one simulation step."""
    state = simulation.step()
    return safe_jsonify(state)


@app.route('/api/run_all', methods=['POST'])
def run_all():
    """Run all remaining simulation steps."""
    while simulation.current_step < simulation.total_steps:
        simulation.step()
    return safe_jsonify(simulation.get_state())


@app.route('/api/reset', methods=['POST'])
def reset():
    """Reset everything."""
    global city_graph, simulation
    city_graph = CityGraph(width=10, height=10)
    simulation = Simulation(city_graph, total_steps=20)
    return safe_jsonify(simulation.get_state())


@app.route('/api/block_road', methods=['POST'])
def block_road():
    """Manually block a road (for live modification demo)."""
    data = request.json
    pos1 = tuple(data['pos1'])
    pos2 = tuple(data['pos2'])
    city_graph.block_road(pos1, pos2)
    if simulation.router:
        simulation.router.reroute_after_block(pos1, pos2)
    return safe_jsonify({'status': 'blocked', 'pos1': list(pos1), 'pos2': list(pos2)})


@app.route('/api/update_constraint', methods=['POST'])
def update_constraint():
    """
    Live modification challenge: change a constraint and re-run.
    Supports changing hospital_hops, powerplant_hops, num_ambulances.
    """
    global city_graph, simulation
    data = request.json
    constraint = data.get('constraint')
    value = data.get('value')

    city_graph = CityGraph(width=10, height=10)
    simulation = Simulation(city_graph, total_steps=20)

    # Apply constraint changes before initialization
    if constraint == 'hospital_hops':
        simulation.hospital_hops = int(value)
        state = simulation.initialize()
    elif constraint == 'powerplant_hops':
        simulation.powerplant_hops = int(value)
        state = simulation.initialize()
    elif constraint == 'num_ambulances':
        state = simulation.initialize()
        from challenges.ambulance_placement import place_ambulances
        result = place_ambulances(city_graph, num_ambulances=int(value))
        simulation.ambulance_positions = [tuple(p) for p in result['positions']]
        simulation.results['ambulance_placement'] = result
    else:
        state = simulation.initialize()

    return safe_jsonify(simulation.get_state())


if __name__ == '__main__':
    print("\n" + "="*60)
    print("  CityMind — Urban Intelligence System")
    print("  Open http://127.0.0.1:5000 in your browser")
    print("="*60 + "\n")
    app.run(debug=True, port=5000)
