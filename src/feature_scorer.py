"""
feature_scorer.py

Assigns multi-objective edge weights to the graph.
Handles both real OSM data and synthetic graphs.
OSM attributes can be lists, strings, ints, or None — this module
handles all cases safely using _get() helper.
"""

import random
import numpy as np
import networkx as nx


GOOD_SURFACES = {"asphalt", "concrete", "paved", "paving_stones", "sett"}
BAD_SURFACES  = {"gravel", "dirt", "grass", "sand", "cobblestone", "unpaved"}

HIGHWAY_SAFETY = {
    "footway":       0.95,
    "path":          0.80,
    "pedestrian":    0.95,
    "living_street": 0.85,
    "residential":   0.75,
    "service":       0.70,
    "tertiary":      0.60,
    "secondary":     0.45,
    "primary":       0.30,
    "trunk":         0.15,
    "motorway":      0.05,
}


# ── safe getter ───────────────────────────────────────────────────────────────

def _get(data: dict, key: str, default):
    """
    Safely get a value from edge data dict.
    Handles: None, list, string, int, float — always returns a scalar.
    """
    val = data.get(key, default)
    if val is None:
        return default
    if isinstance(val, list):
        val = val[0] if len(val) > 0 else default
    if val is None or val == "":
        return default
    return val


def _safe_float(val, default: float = 0.0) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _safe_int(val, default: int = 1) -> int:
    try:
        return int(float(str(val).split(".")[0]))
    except (TypeError, ValueError):
        return default


def _safe_str(val, default: str = "") -> str:
    try:
        return str(val).lower().strip()
    except Exception:
        return default


# ── feature extraction ────────────────────────────────────────────────────────

def _score_edge(data: dict, rng: random.Random) -> dict:
    """
    Score a single edge. Works for both OSM and synthetic edge data.
    Returns dict with distance_m, safety_score, accessibility, complexity.
    """

    # ── distance ──────────────────────────────────────────────────────────────
    dist = _safe_float(_get(data, "length", _get(data, "distance_m", 50.0)), 50.0)
    if dist <= 0:
        dist = 50.0

    # ── safety ────────────────────────────────────────────────────────────────
    hw      = _safe_str(_get(data, "highway", "residential"))
    safety  = HIGHWAY_SAFETY.get(hw, 0.60)

    sidewalk = _safe_str(_get(data, "sidewalk", ""))
    if sidewalk in ("yes", "both", "left", "right"):
        safety = min(1.0, safety + 0.10)

    safety = float(np.clip(safety + rng.uniform(-0.05, 0.05), 0.05, 1.0))

    # ── accessibility ─────────────────────────────────────────────────────────
    acc = 0.5

    surface = _safe_str(_get(data, "surface", ""))
    if surface in GOOD_SURFACES:
        acc += 0.25
    elif surface in BAD_SURFACES:
        acc -= 0.25

    if _safe_str(_get(data, "tactile_paving", "")) == "yes":
        acc += 0.20

    wheelchair = _safe_str(_get(data, "wheelchair", ""))
    if wheelchair == "yes":
        acc += 0.15
    elif wheelchair == "no":
        acc -= 0.30

    kerb = _safe_str(_get(data, "kerb", ""))
    if kerb in ("lowered", "flush"):
        acc += 0.10
    elif kerb == "raised":
        acc -= 0.15

    # for edges with no OSM tags at all, use distance-based heuristic
    has_osm_tags = any(k in data for k in ("highway", "surface", "sidewalk", "wheelchair"))
    if not has_osm_tags:
        road_level = min(dist / 300.0, 1.0)
        safety = float(np.clip(0.85 - road_level * 0.55 + rng.uniform(-0.1, 0.1), 0.1, 1.0))
        acc    = float(np.clip(0.70 - road_level * 0.40 + rng.uniform(-0.1, 0.1), 0.0, 1.0))

    acc = float(np.clip(acc + rng.uniform(-0.05, 0.05), 0.0, 1.0))

    # ── complexity ────────────────────────────────────────────────────────────
    lanes    = _safe_int(_get(data, "lanes", 1), 1)
    oneway   = _safe_str(_get(data, "oneway", "no")) == "yes"
    maxspeed = _safe_float(_get(data, "maxspeed", 30) or 30, 30.0)

    complexity = lanes * 0.2 + (0.2 if oneway else 0.0) + maxspeed / 200.0
    complexity = float(np.clip(complexity + rng.uniform(0, 0.1), 0.0, 1.0))

    return {
        "distance_m":    round(dist,       2),
        "safety_score":  round(safety,     4),
        "accessibility": round(acc,        4),
        "complexity":    round(complexity, 4),
    }


# ── main functions ────────────────────────────────────────────────────────────

def score_graph(G: nx.MultiDiGraph, seed: int = 0) -> nx.MultiDiGraph:
    """
    Iterate over every edge in G and add the four feature attributes in-place.
    Returns the same graph (modified).
    """
    rng   = random.Random(seed)
    count = 0

    for u, v, key, data in G.edges(keys=True, data=True):
        try:
            feats = _score_edge(data, rng)
        except Exception as e:
            # ultimate fallback — should never happen but just in case
            print(f"[feature_scorer] Warning: edge ({u},{v}) scoring failed ({e}), using defaults")
            feats = {
                "distance_m":    50.0,
                "safety_score":  0.6,
                "accessibility": 0.5,
                "complexity":    0.3,
            }

        for k, val in feats.items():
            G[u][v][key][k] = val

        G[u][v][key]["composite_cost"] = (
            0.25 * (feats["distance_m"] / 500.0) +
            0.40 * (1.0 - feats["safety_score"]) +
            0.25 * (1.0 - feats["accessibility"]) +
            0.10 * feats["complexity"]
        )
        count += 1

    print(f"[feature_scorer] Scored {count} edges successfully")
    return G


def normalise_graph_weights(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """Normalise distance_m to [0,1] scale across the graph."""
    distances = []
    for _, _, d in G.edges(data=True):
        v = d.get("distance_m")
        if v is not None:
            try:
                distances.append(float(v))
            except (TypeError, ValueError):
                pass

    if not distances:
        return G

    max_d = max(distances)
    if max_d == 0:
        return G

    for u, v, key, data in G.edges(keys=True, data=True):
        try:
            G[u][v][key]["distance_norm"] = data.get("distance_m", 0) / max_d
        except Exception:
            G[u][v][key]["distance_norm"] = 0.0

    return G
