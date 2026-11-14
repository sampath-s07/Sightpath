"""
evaluator.py  —  compare algorithm results and compute evaluation metrics

After running all three algorithms we want to know:
  - Which one found the safest route?
  - Which is most accessible?
  - How much does the Pareto front improve over single-objective baselines?
  - How fast is each algorithm?

We also compute the Hypervolume indicator for the NSGA-II Pareto front,
which is the standard metric in multi-objective optimisation literature.
"""

import math
import numpy as np
import pandas as pd
from typing import List, Optional


# ── Single-route metrics ──────────────────────────────────────────────────────

def route_score(result: dict) -> Optional[dict]:
    """
    Summarise a single route result (from dijkstra or astar) into a
    flat metric dict. Returns None if the result has an error.
    """
    if not result.get("path") or result.get("error"):
        return None

    return {
        "algorithm":     result.get("algorithm", "?"),
        "distance_m":    result.get("distance_m", 0),
        "safety":        result.get("safety",        0),
        "accessibility": result.get("accessibility", 0),
        "complexity":    result.get("complexity",    0),
        "n_edges":       result.get("n_edges",       0),
        "time_ms":       result.get("time_ms",       0),
    }


def summarise_pareto(pareto_front: list) -> Optional[dict]:
    """
    Aggregate stats across the entire NSGA-II Pareto front:
    best / worst / mean for each objective.
    """
    if not pareto_front:
        return None

    objs = np.array([ind.objectives for ind in pareto_front])

    # objective labels (all minimised)
    labels = ["distance_km", "avg_risk", "avg_inaccessibility", "avg_complexity"]
    summary = {"algorithm": "nsga2", "pareto_size": len(pareto_front)}

    for i, label in enumerate(labels):
        col = objs[:, i]
        summary[f"{label}_min"]  = round(float(col.min()),  4)
        summary[f"{label}_mean"] = round(float(col.mean()), 4)
        summary[f"{label}_max"]  = round(float(col.max()),  4)

    return summary


# ── Hypervolume ───────────────────────────────────────────────────────────────

def hypervolume(pareto_front: list, reference_point: Optional[List[float]] = None) -> float:
    """
    Compute 2D hypervolume of the Pareto front using objectives 0 and 1
    (distance and risk). The HV is the area dominated by the front
    relative to a reference point.

    We use just 2D for simplicity; extending to 4D would need a full HV algorithm
    (e.g., WFG) which is significantly more complex to implement from scratch.

    reference_point defaults to [max_dist + 0.1, max_risk + 0.1].
    """
    if not pareto_front:
        return 0.0

    # extract first two objectives
    points = [(ind.objectives[0], ind.objectives[1]) for ind in pareto_front
              if ind.objectives[0] < 1e8]

    if not points:
        return 0.0

    if reference_point is None:
        ref_x = max(p[0] for p in points) + 0.1
        ref_y = max(p[1] for p in points) + 0.1
        reference_point = [ref_x, ref_y]

    # sort by x ascending
    points.sort(key=lambda p: p[0])

    hv    = 0.0
    prev_y = reference_point[1]

    for x, y in points:
        if y < prev_y:
            hv    += (reference_point[0] - x) * (prev_y - y)
            prev_y = y

    return round(hv, 6)


def spread(pareto_front: list) -> float:
    """
    Spread (diversity) of the Pareto front in objective space.
    Computed as the average distance between consecutive solutions
    after sorting by first objective. Higher = more diverse front.
    """
    if len(pareto_front) < 2:
        return 0.0

    pts = sorted(
        [(ind.objectives[0], ind.objectives[1]) for ind in pareto_front
         if ind.objectives[0] < 1e8],
        key=lambda p: p[0]
    )
    if len(pts) < 2:
        return 0.0

    dists = [
        math.sqrt((pts[i+1][0] - pts[i][0])**2 + (pts[i+1][1] - pts[i][1])**2)
        for i in range(len(pts) - 1)
    ]
    return round(sum(dists) / len(dists), 6)


# ── Comparison table ──────────────────────────────────────────────────────────

def build_comparison_table(dijkstra_results: dict,
                            astar_result:     dict,
                            nsga2_result:     dict) -> pd.DataFrame:
    """
    Build a tidy comparison DataFrame suitable for printing or plotting.
    dijkstra_results is a dict of {objective_name: result_dict}.
    """
    rows = []

    # dijkstra runs
    for obj_name, res in dijkstra_results.items():
        s = route_score(res)
        if s:
            s["label"] = f"Dijkstra ({obj_name})"
            rows.append(s)

    # A*
    s = route_score(astar_result)
    if s:
        s["label"] = "A* Composite"
        rows.append(s)

    # NSGA-II best compromise route
    pf = nsga2_result.get("pareto_front", [])
    if pf:
        from src.algorithms.nsga2_optimizer import best_compromise
        best = best_compromise(pf)
        if best and best.path:
            # reconstruct stats — objectives are [dist_km, risk, inacc, comp]
            obj = best.objectives
            rows.append({
                "label":         "NSGA-II (best compromise)",
                "algorithm":     "nsga2",
                "distance_m":    round(obj[0] * 1000, 1),   # km → m
                "safety":        round(1.0 - obj[1], 4),
                "accessibility": round(1.0 - obj[2], 4),
                "complexity":    round(obj[3], 4),
                "n_edges":       len(best.path) - 1,
                "time_ms":       nsga2_result.get("time_ms", 0),
            })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # reorder columns
    cols = ["label", "distance_m", "safety", "accessibility",
            "complexity", "n_edges", "time_ms", "algorithm"]
    df = df[[c for c in cols if c in df.columns]]
    return df


def print_comparison(df: pd.DataFrame, nsga2_result: dict) -> None:
    """Pretty-print the comparison table and extra NSGA-II metrics."""
    if df.empty:
        print("[evaluator] No results to compare.")
        return

    print("\n" + "=" * 70)
    print("  ALGORITHM COMPARISON")
    print("=" * 70)
    print(df.to_string(index=False))

    pf = nsga2_result.get("pareto_front", [])
    if pf:
        hv = hypervolume(pf)
        sp = spread(pf)
        print(f"\n  NSGA-II Pareto front size : {len(pf)}")
        print(f"  Hypervolume (2D)          : {hv}")
        print(f"  Spread (diversity)        : {sp}")
        print(f"  Generations run           : {nsga2_result.get('generations', '?')}")
        print(f"  Total time                : {nsga2_result.get('time_ms', '?')} ms")
    print("=" * 70 + "\n")

