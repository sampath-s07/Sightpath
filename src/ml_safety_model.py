"""
ml_safety_model.py  —  ML-based safety & accessibility scorer for SightPath

Trains a Random Forest classifier/regressor on OSM edge features to predict
safety and accessibility scores. This replaces the hand-coded lookup table
in feature_scorer.py with a data-driven ML model.

The model is trained on the same OSM data we already have — it uses the
hand-coded scores as 'ground truth' labels for the initial training, then
gets refined by user feedback over time (preference learning).

Why Random Forest?
  - Works well with mixed feature types (numbers, categories)
  - Handles missing values gracefully
  - Easy to interpret (feature importances)
  - No need for feature scaling
  - Fast to train on small graphs
"""

import os
import pickle
import random
import numpy as np
import networkx as nx

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder

MODEL_DIR  = "data"
SAFETY_MODEL_PATH = os.path.join(MODEL_DIR, "ml_safety_model.pkl")
ACCESS_MODEL_PATH = os.path.join(MODEL_DIR, "ml_access_model.pkl")

# Maps highway type to numeric level (for ML encoding)
HIGHWAY_LEVELS = {
    "motorway": 1, "trunk": 2, "primary": 3, "secondary": 4,
    "tertiary": 5, "residential": 6, "service": 7,
    "living_street": 8, "path": 9, "footway": 10, "pedestrian": 10,
    "cycleway": 8, "unclassified": 5,
}

SURFACE_LEVELS = {
    "asphalt": 10, "concrete": 10, "paved": 9, "paving_stones": 8,
    "sett": 7, "compacted": 6, "fine_gravel": 5,
    "gravel": 3, "unpaved": 2, "dirt": 2, "grass": 1, "sand": 1, "cobblestone": 4,
}


def _extract_features(edge_data: dict) -> list:
    """
    Convert raw OSM edge attributes into a numeric feature vector for ML.

    Features:
      0  highway_level        (1=motorway .. 10=footway)
      1  surface_level        (1=worst .. 10=best)
      2  has_sidewalk         (0/1)
      3  has_tactile_paving   (0/1)
      4  wheelchair_yes       (0/1)
      5  wheelchair_no        (0/1)
      6  kerb_lowered         (0/1)
      7  kerb_raised          (0/1)
      8  lanes                (count)
      9  is_oneway            (0/1)
      10 maxspeed             (km/h, capped at 120)
      11 distance_m           (metres, capped at 500)
    """
    def _get(key, default):
        v = edge_data.get(key, default)
        if isinstance(v, list):
            v = v[0] if v else default
        return v if v is not None else default

    def _str(key, default=""):
        return str(_get(key, default)).lower().strip()

    def _flt(key, default=0.0):
        try:
            return float(_get(key, default))
        except (TypeError, ValueError):
            return float(default)

    def _int(key, default=1):
        try:
            return int(float(str(_get(key, default)).split(".")[0]))
        except (TypeError, ValueError):
            return int(default)

    hw      = _str("highway", "residential")
    surface = _str("surface", "")
    sidewalk = _str("sidewalk", "")
    tactile  = _str("tactile_paving", "")
    wheelchair = _str("wheelchair", "")
    kerb    = _str("kerb", "")
    lanes   = _int("lanes", 1)
    oneway  = 1 if _str("oneway", "no") == "yes" else 0
    maxspeed = min(_flt("maxspeed", 30), 120.0)
    dist    = min(_flt("length", _flt("distance_m", 50.0)), 500.0)

    hw_level  = HIGHWAY_LEVELS.get(hw, 5)
    sur_level = SURFACE_LEVELS.get(surface, 5) if surface else 5

    has_sidewalk = 1 if sidewalk in ("yes", "both", "left", "right") else 0
    has_tactile  = 1 if tactile == "yes" else 0
    wc_yes = 1 if wheelchair == "yes" else 0
    wc_no  = 1 if wheelchair == "no"  else 0
    kerb_low    = 1 if kerb in ("lowered", "flush") else 0
    kerb_raised = 1 if kerb == "raised" else 0

    return [
        hw_level, sur_level, has_sidewalk, has_tactile,
        wc_yes, wc_no, kerb_low, kerb_raised,
        lanes, oneway, maxspeed, dist,
    ]


FEATURE_NAMES = [
    "highway_level", "surface_level", "has_sidewalk", "has_tactile_paving",
    "wheelchair_yes", "wheelchair_no", "kerb_lowered", "kerb_raised",
    "lanes", "is_oneway", "maxspeed", "distance_m",
]


def build_training_data(G: nx.MultiDiGraph):
    """
    Build training (X, y_safety, y_access) from the already-scored graph.
    The hand-coded scores from feature_scorer.py serve as initial labels.
    """
    X, y_safety, y_access = [], [], []

    for u, v, key, data in G.edges(keys=True, data=True):
        # only use edges that have already been scored
        if "safety_score" not in data:
            continue
        try:
            features = _extract_features(data)
            X.append(features)
            y_safety.append(data["safety_score"])
            y_access.append(data["accessibility"])
        except Exception:
            continue

    return np.array(X), np.array(y_safety), np.array(y_access)


def train_models(G: nx.MultiDiGraph, verbose: bool = True):
    """
    Train Random Forest models for safety and accessibility prediction.
    Saves models to disk and returns (safety_model, access_model, metrics).
    """
    os.makedirs(MODEL_DIR, exist_ok=True)

    X, y_safety, y_access = build_training_data(G)
    if len(X) < 10:
        raise ValueError("Not enough edge data to train ML models.")

    if verbose:
        print(f"[ml_safety_model] Training on {len(X)} edge samples...")

    X_tr, X_te, ys_tr, ys_te = train_test_split(X, y_safety, test_size=0.2, random_state=42)
    _,    _,    ya_tr, ya_te  = train_test_split(X, y_access, test_size=0.2, random_state=42)

    safety_model = RandomForestRegressor(
        n_estimators=120, max_depth=8, min_samples_leaf=3,
        random_state=42, n_jobs=-1
    )
    safety_model.fit(X_tr, ys_tr)

    access_model = RandomForestRegressor(
        n_estimators=120, max_depth=8, min_samples_leaf=3,
        random_state=42, n_jobs=-1
    )
    access_model.fit(X_tr, ya_tr)

    # Evaluation
    ys_pred = safety_model.predict(X_te)
    ya_pred = access_model.predict(X_te)

    metrics = {
        "safety_mae":    round(mean_absolute_error(ys_te, ys_pred), 4),
        "safety_r2":     round(r2_score(ys_te, ys_pred), 4),
        "access_mae":    round(mean_absolute_error(ya_te, ya_pred), 4),
        "access_r2":     round(r2_score(ya_te, ya_pred), 4),
        "n_samples":     len(X),
        "feature_importances_safety": dict(zip(FEATURE_NAMES, safety_model.feature_importances_.round(4).tolist())),
    }

    if verbose:
        print(f"[ml_safety_model] Safety  — MAE: {metrics['safety_mae']}, R²: {metrics['safety_r2']}")
        print(f"[ml_safety_model] Access  — MAE: {metrics['access_mae']}, R²: {metrics['access_r2']}")

    # save to disk
    with open(SAFETY_MODEL_PATH, "wb") as f:
        pickle.dump(safety_model, f)
    with open(ACCESS_MODEL_PATH, "wb") as f:
        pickle.dump(access_model, f)

    if verbose:
        print(f"[ml_safety_model] Models saved to {MODEL_DIR}/")

    return safety_model, access_model, metrics


def load_models():
    """Load pre-trained models from disk. Returns (safety_model, access_model) or (None, None)."""
    if not (os.path.exists(SAFETY_MODEL_PATH) and os.path.exists(ACCESS_MODEL_PATH)):
        return None, None
    with open(SAFETY_MODEL_PATH, "rb") as f:
        safety_model = pickle.load(f)
    with open(ACCESS_MODEL_PATH, "rb") as f:
        access_model = pickle.load(f)
    return safety_model, access_model


def apply_ml_scores(G: nx.MultiDiGraph, safety_model, access_model) -> nx.MultiDiGraph:
    """
    Re-score all edges using ML-predicted safety and accessibility.
    Updates the graph in-place. Returns G.
    """
    count = 0
    for u, v, key, data in G.edges(keys=True, data=True):
        try:
            features = np.array(_extract_features(data)).reshape(1, -1)
            ml_safety = float(np.clip(safety_model.predict(features)[0], 0.0, 1.0))
            ml_access = float(np.clip(access_model.predict(features)[0], 0.0, 1.0))

            G[u][v][key]["safety_score"]  = round(ml_safety, 4)
            G[u][v][key]["accessibility"] = round(ml_access, 4)

            # update composite_cost with ML scores
            dist = G[u][v][key].get("distance_m", 50.0)
            comp = G[u][v][key].get("complexity", 0.3)
            G[u][v][key]["composite_cost"] = (
                0.25 * (dist / 500.0) +
                0.40 * (1.0 - ml_safety) +
                0.25 * (1.0 - ml_access) +
                0.10 * comp
            )
            count += 1
        except Exception:
            continue

    print(f"[ml_safety_model] ML scores applied to {count} edges.")
    return G
