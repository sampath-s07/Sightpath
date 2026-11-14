"""
dijkstra.py  —  baseline single-objective shortest-path

This is the classic Dijkstra's algorithm used as a baseline to compare against
our multi-objective methods. We run it three times — once per objective — so
we can at least see what the "best possible" single-objective route looks like.

Nothing fancy here, mostly wrapping networkx's implementation and
recording some extra metadata (time taken, explored nodes, etc.)
"""

import time
import networkx as nx


def run(G: nx.MultiDiGraph, source: int, target: int, weight: str = "distance_m") -> dict:
    """
    Run Dijkstra from source to target using the given edge weight.

    Parameters
    ----------
    G       : the scored road network
    source  : start node id
    target  : end node id
    weight  : edge attribute to minimise (distance_m | safety_score | accessibility | composite_cost)

    Returns
    -------
    dict with keys:
        path        — list of node ids
        total_cost  — sum of `weight` along path
        distance_m  — total physical distance
        safety      — average safety score along path
        accessibility — average accessibility
        complexity  — average complexity
        time_ms     — runtime in milliseconds
        algorithm   — "dijkstra"
    """
    start_t = time.perf_counter()

    # for safety and accessibility, higher is better, so we invert them
    # so we can still minimise the cost
    invert = weight in ("safety_score", "accessibility")

    if invert:
        # build a temporary dict of inverted weights
        for u, v, k, d in G.edges(keys=True, data=True):
            G[u][v][k]["_inv_" + weight] = 1.0 - d.get(weight, 0.5)
        actual_weight = "_inv_" + weight
    else:
        actual_weight = weight

    try:
        path = nx.dijkstra_path(G, source, target, weight=actual_weight)
        cost = nx.dijkstra_path_length(G, source, target, weight=actual_weight)
    except nx.NetworkXNoPath:
        return {"path": [], "error": "No path found", "algorithm": "dijkstra"}
    except nx.NodeNotFound as e:
        return {"path": [], "error": str(e), "algorithm": "dijkstra"}

    elapsed = (time.perf_counter() - start_t) * 1000.0

    # collect stats along the found path
    stats = _path_stats(G, path)
    stats["total_cost"]  = cost if not invert else (1.0 - cost)
    stats["path"]        = path
    stats["time_ms"]     = round(elapsed, 3)
    stats["algorithm"]   = "dijkstra"
    stats["weight_used"] = weight
    return stats


def run_all_objectives(G: nx.MultiDiGraph, source: int, target: int) -> dict:
    """
    Run Dijkstra for each individual objective.
    Returns a dict keyed by objective name.
    """
    objectives = ["distance_m", "composite_cost", "safety_score", "accessibility"]
    results = {}
    for obj in objectives:
        results[obj] = run(G, source, target, weight=obj)
    return results


def _path_stats(G: nx.MultiDiGraph, path: list) -> dict:
    """Compute aggregate metrics for a given node-sequence path."""
    if len(path) < 2:
        return {
            "distance_m": 0, "safety": 0, "accessibility": 0, "complexity": 0,
            "n_edges": 0,
        }

    total_dist = 0.0
    safety_vals, acc_vals, comp_vals = [], [], []

    for u, v in zip(path[:-1], path[1:]):
        # pick the cheapest parallel edge (MultiDiGraph can have several)
        edges = G[u][v]
        best_key = min(edges, key=lambda k: edges[k].get("distance_m", float("inf")))
        d = edges[best_key]

        total_dist  += d.get("distance_m",    0.0)
        safety_vals.append(d.get("safety_score",  0.5))
        acc_vals.append(d.get("accessibility", 0.5))
        comp_vals.append(d.get("complexity",   0.5))

    return {
        "distance_m":    round(total_dist, 2),
        "safety":        round(sum(safety_vals) / len(safety_vals), 4),
        "accessibility": round(sum(acc_vals)    / len(acc_vals),    4),
        "complexity":    round(sum(comp_vals)   / len(comp_vals),   4),
        "n_edges":       len(path) - 1,
    }

