# config.py — central settings for SightPath

GRAPH_TYPE  = "walk"      # pedestrian walkable network
OSM_TIMEOUT = 60

# weights for A* composite heuristic (distance, safety, accessibility, complexity)
WEIGHTS = {
    "distance":      0.25,
    "safety":        0.40,
    "accessibility": 0.25,
    "complexity":    0.10,
}

# NSGA-II parameters
NSGA2_PARAMS = {
    "pop_size":       80,
    "n_generations":  120,
    "crossover_prob": 0.85,
    "mutation_prob":  0.15,
    "seed":           42,
}

# directories
OUTPUT_DIR = "outputs"
DATA_DIR   = "data"
