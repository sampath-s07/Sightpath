"""
astar.py  —  A* with composite multi-objective heuristic

Standard A* but the edge cost combines all four objectives into one
weighted scalar (the composite_cost attribute we assign in feature_scorer).

This is better than plain Dijkstra because:
  1. the heuristic guides the search toward the destination faster
  2. the composite cost already bakes in safety + accessibility

The heuristic is Euclidean distance in graph coordinate space,
scaled by the same factor we use to normalise distance_m.

For graphs that come from OSM projection (UTM), node x/y are in metres,
so the heuristic is admissible as long as the scaling is correct.
"""

import heapq
import math
import time
import networkx as nx


def _euclidean(G: nx.MultiDiGraph, u: int, v: int) -> float:
    """Straight-line distance between two nodes using their (x, y) coords."""
    xu, yu = G.nodes[u].get("x", 0), G.nodes[u].get("y", 0)
    xv, yv = G.nodes[v].get("x", 0), G.nodes[v].get("y", 0)
    return math.sqrt((xu - xv) ** 2 + (yu - yv) ** 2)


def _heuristic(G: nx.MultiDiGraph, u: int, goal: int, scale: float = 500.0) -> float:
    """
    Heuristic for A* — straight-line distance / scale.
    Scale = rough max edge distance; keeps heuristic values comparable
    to the composite_cost edge weights (which are also in [0,1] range).
    """
    return _euclidean(G, u, goal) / scale * 0.25   # weight 0.25 same as distance in composite


def run(G: nx.MultiDiGraph, source: int, target: int, weight: str = "composite_cost") -> dict:
    """
    A* search from source to target.

    Parameters
    ----------
    G      : scored graph
    source : start node
    target : end node
    weight : edge cost attribute to use (default = composite_cost)

    Returns
    -------
    Same dict schema as dijkstra.run()
    """
    start_t = time.perf_counter()

    # priority queue: (f_score, g_score, node, path_so_far)
    # storing full path in queue is memory-heavy but simple to implement
    open_heap = []
    h0 = _heuristic(G, source, target)
    heapq.heappush(open_heap, (h0, 0.0, source, [source]))

    visited   = {}   # node -> best g_score seen
    nodes_explored = 0

    while open_heap:
        f, g, current, path = heapq.heappop(open_heap)

        if current in visited and visited[current] <= g:
            continue
        visited[current] = g
        nodes_explored  += 1

        if current == target:
            elapsed = (time.perf_counter() - start_t) * 1000.0
            stats = _path_stats(G, path)
            stats["path"]           = path
            stats["total_cost"]     = round(g, 6)
            stats["time_ms"]        = round(elapsed, 3)
            stats["algorithm"]      = "astar"
            stats["nodes_explored"] = nodes_explored
            return stats

        if current not in G:
            continue

        for nbr, edges in G[current].items():
            # pick the minimum-cost parallel edge
            best_cost = min(
                e.get(weight, e.get("composite_cost", 1.0))
                for e in edges.values()
            )
            new_g = g + best_cost
            h     = _heuristic(G, nbr, target)
            new_f = new_g + h

            if nbr not in visited or visited[nbr] > new_g:
                heapq.heappush(open_heap, (new_f, new_g, nbr, path + [nbr]))

    # if we exit the loop without finding target
    elapsed = (time.perf_counter() - start_t) * 1000.0
    return {
        "path": [], "error": "No path found",
        "algorithm": "astar", "time_ms": round(elapsed, 3),
    }


def _path_stats(G: nx.MultiDiGraph, path: list) -> dict:
    """Same helper as in dijkstra.py — aggregate stats along a path."""
    if len(path) < 2:
        return {"distance_m": 0, "safety": 0, "accessibility": 0, "complexity": 0, "n_edges": 0}

    total_dist = 0.0
    safety_vals, acc_vals, comp_vals = [], [], []

    for u, v in zip(path[:-1], path[1:]):
        edges = G[u][v]
        best_key = min(edges, key=lambda k: edges[k].get("composite_cost", 1.0))
        d = edges[best_key]
        total_dist += d.get("distance_m", 0.0)
        safety_vals.append(d.get("safety_score",  0.5))
        acc_vals.append(   d.get("accessibility", 0.5))
        comp_vals.append(  d.get("complexity",    0.5))

    return {
        "distance_m":    round(total_dist, 2),
        "safety":        round(sum(safety_vals) / len(safety_vals), 4),
        "accessibility": round(sum(acc_vals)    / len(acc_vals),    4),
        "complexity":    round(sum(comp_vals)   / len(comp_vals),   4),
        "n_edges":       len(path) - 1,
    }

