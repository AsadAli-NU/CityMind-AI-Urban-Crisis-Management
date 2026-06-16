# CityMind - AI Urban Crisis Management System

CityMind is an Artificial Intelligence semester project that simulates an intelligent urban crisis management system. The project is designed to help manage city planning, road networks, ambulance placement, emergency routing, and crime risk prediction using multiple AI techniques.

The system uses a shared city graph where different modules interact with the same city data. Changes such as blocked roads, risk updates, and ambulance repositioning are reflected across the simulation.

## Project Overview

CityMind models a city as a grid-based graph. Each location in the city can represent a residential area, hospital, school, industrial zone, power plant, ambulance depot, police unit, or civilian location.

The project combines different AI algorithms to solve five major challenges:

1. City layout planning
2. Road network optimization
3. Ambulance placement
4. Emergency routing
5. Crime risk prediction and police deployment

## Key Features

* Interactive city simulation interface
* Grid-based city visualization
* Road network overlay
* Ambulance coverage overlay
* Crime risk heatmap
* Police coverage overlay
* Live event log
* 20-step simulation flow
* Live modification of selected constraints
* Dynamic emergency routing when roads are blocked
* Risk-based ambulance repositioning

## AI Techniques Used

### 1. Constraint Satisfaction Problem

Used for city layout planning. The system places facilities such as hospitals, schools, residential areas, industrial zones, power plants, and ambulance depots while following placement constraints.

### 2. Genetic Algorithm

Used for road network optimization and ambulance placement. The system searches for efficient road structures and suitable ambulance locations.

### 3. A* Search

Used for emergency routing. The system finds the shortest available path for reaching civilians, while adapting when roads become blocked.

### 4. K-Means Clustering

Used for grouping city areas based on features such as population density and industrial proximity.

### 5. Decision Tree Classification

Used for predicting crime risk levels and supporting police deployment decisions.

## Technology Stack

* Python
* Flask
* HTML5
* CSS
* JavaScript
* HTML5 Canvas
* scikit-learn

## Project Structure

```text
CityMind/
├── app.py
├── core/
│   ├── city_graph.py
│   └── simulation.py
├── challenges/
│   ├── layout_csp.py
│   ├── road_network_ga.py
│   ├── ambulance_placement.py
│   ├── emergency_routing.py
│   └── crime_prediction.py
├── static/
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── main.js
├── templates/
│   └── index.html
├── tests/
│   └── test_challenges.py
└── README.md
```

## Modules

### City Layout Planning

The layout module uses CSP concepts to place city facilities while checking constraints such as hospital distance, industrial zone separation, and power plant placement.

### Road Network Optimization

The road module uses a Genetic Algorithm to build a connected road network while considering cost and redundancy requirements.

### Ambulance Placement

The ambulance module places ambulances in locations that reduce worst-case response time and improve emergency coverage.

### Emergency Routing

The routing module uses A* Search to find the shortest available route to civilians. If a road becomes blocked, the system recalculates the route.

### Crime Prediction

The crime prediction module uses K-Means clustering and Decision Tree classification to predict risk levels across the city. These risk levels are used to influence routing and deployment decisions.

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/AsadAli-NU/CityMind-AI-Urban-Crisis-Management.git
cd CityMind-AI-Urban-Crisis-Management
```

### 2. Install required packages

```bash
pip install flask scikit-learn
```

### 3. Run the Flask application

```bash
python app.py
```

### 4. Open the project in your browser

```text
http://127.0.0.1:5000
```

## Usage

After running the application, the web interface allows you to:

* Initialize the city
* Run the simulation step by step
* Run all simulation steps
* Reset the simulation
* Toggle road network, ambulance coverage, crime risk heatmap, civilians, and police coverage
* View challenge results
* Track system decisions through the live event log
* Modify selected constraints during runtime

## Team Members

* Asad Ali
* Rehan Ahmed
* Muhammad Harris

## Learning Outcomes

This project helped us understand how different AI techniques can be integrated into one system. It strengthened our understanding of search algorithms, constraint satisfaction, optimization, machine learning, simulation design, Flask-based development, and teamwork.

## Disclaimer

This project was developed as an academic semester project for learning purposes. It is a simulation and is not intended for real-world emergency management use.
