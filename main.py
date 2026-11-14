"""
main.py  —  SightPath pipeline runner

Usage:
    python main.py

Loads real Chennai road network from data/chennai.osm,
runs all three algorithms, evaluates and plots results.
"""

import os
import sys
import random

sys.path.insert(0, os.path.dirname(__file__))

import networkx as nx

from config import NSGA2_PARAMS, OUTPUT_DIR
from src.graph_builder  import get_graph
from src.feature_scorer import score_graph, normalise_graph_weights
from src.algorithms     import dijkstra, astar, nsga2_optimizer as nsga2
from src.evaluator      import (build_comparison_table, print_comparison,
                                hypervolume, spread)
from src.visualizer     import (plot_graph, plot_routes,
                                plot_pareto_front, plot_metrics_bar,
                                plot_accessibility_heatmap, plot_convergence,
                                plot_radar_chart, plot_feature_distributions)


def pick_endpoints(G: nx.MultiDiGraph, seed: int = 7) -> tuple:
    """Pick a source and target node with a valid path of reasonable length."""
    rng   = random.Random(seed)
    nodes = list(G.nodes())
    well_connected = [n for n in nodes if G.degree(n) >= 3] or nodes

    for _ in range(50):
        src, tgt = rng.sample(well_connected, 2)
        try:
            path = nx.shortest_path(G, src, tgt, weight="distance_m")
            if len(path) >= 5:
                return src, tgt
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue

    return nodes[0], nodes[-1]


def main(seed: int = 42):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n" + "=" * 60)
    print("  SightPath — Multi-Objective Route Optimisation")
    print("  Dataset: OpenStreetMap — Chennai, India")
    print("=" * 60)

    # ── 1. Load real OSM graph ────────────────────────────────────
    print("\n[Step 1] Loading Chennai road network ...")
    G = get_graph()

    # ── 2. Score edges ────────────────────────────────────────────
    print("\n[Step 2] Scoring edges with multi-objective features ...")
    G = score_graph(G, seed=seed)
    G = normalise_graph_weights(G)

    # ── 3. Pick endpoints ─────────────────────────────────────────
    print("\n[Step 3] Selecting source and target nodes ...")
    source, target = pick_endpoints(G, seed=seed)
    print(f"         Source: {source}  →  Target: {target}")

    # ── 4. Run algorithms ─────────────────────────────────────────
    print("\n[Step 4] Running algorithms ...")

    print("  → Dijkstra (all objectives) ...")
    dijkstra_results = dijkstra.run_all_objectives(G, source, target)

    print("  → A* with composite heuristic ...")
    astar_result = astar.run(G, source, target)

    print("  → NSGA-II ...")
    nsga2_result = nsga2.run(
        G, source, target,
        pop_size       = NSGA2_PARAMS["pop_size"],
        n_generations  = NSGA2_PARAMS["n_generations"],
        crossover_prob = NSGA2_PARAMS["crossover_prob"],
        mutation_prob  = NSGA2_PARAMS["mutation_prob"],
        seed           = NSGA2_PARAMS["seed"],
    )

    # ── 5. Evaluate ───────────────────────────────────────────────
    print("\n[Step 5] Evaluating results ...")
    df = build_comparison_table(dijkstra_results, astar_result, nsga2_result)
    print_comparison(df, nsga2_result)

    csv_path = os.path.join(OUTPUT_DIR, "comparison.csv")
    df.to_csv(csv_path, index=False)
    print(f"[main] Comparison saved → {csv_path}")

    pf = nsga2_result.get("pareto_front", [])
    if pf:
        print(f"[main] Hypervolume: {hypervolume(pf)} | Spread: {spread(pf)}")

    # ── 6. Visualise ──────────────────────────────────────────────
    print("\n[Step 6] Generating plots ...")

    plot_graph(G)
    plot_accessibility_heatmap(G)
    plot_feature_distributions(G)

    routes = {}
    if dijkstra_results.get("distance_m", {}).get("path"):
        routes["dijkstra"] = dijkstra_results["distance_m"]["path"]
    if astar_result.get("path"):
        routes["astar"] = astar_result["path"]
    if pf:
        best = nsga2.best_compromise(pf)
        if best:
            routes["nsga2"] = best.path

    if routes:
        plot_routes(G, routes, source, target)

    if pf:
        best_ind = nsga2.best_compromise(pf)
        plot_pareto_front(pf, highlight=best_ind)

    if not df.empty:
        plot_metrics_bar(df)
        plot_radar_chart(df)

    hv_hist = nsga2_result.get("hv_history", [])
    sp_hist = nsga2_result.get("spread_history", [])
    if hv_hist:
        plot_convergence(hv_hist, sp_hist)

    print("\n[main] All done! Check the 'outputs/' folder for results.\n")
    return df, nsga2_result


if __name__ == "__main__":
    main()
