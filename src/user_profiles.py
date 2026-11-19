"""
user_profiles.py  —  User profiles and preference learning for SightPath

Defines preset profiles for different user types (Blind, Wheelchair, Elderly, General).
Also handles preference learning — tracking which Pareto solutions a user picks
and gradually adapting the weights to match their real preferences.

The preference learning is simple but effective:
  - Each time a user picks a route from the Pareto front, we record which
    objectives that route was best at.
  - We use exponential moving average to update the user's weight vector.
  - After ~5–10 picks, the system recommends routes that match their real pattern.
"""

import os
import json
import math
import numpy as np
from typing import Dict, Tuple, Optional, List

PREFS_FILE = "data/user_preferences.json"


# ── User profiles ─────────────────────────────────────────────────────────────

USER_PROFILES = {
    "🦯 Blind Person": {
        "description": "Needs safe, simple routes — avoids complex junctions and highways",
        "emoji": "🦯",
        "weights": {
            "distance":      0.10,
            "safety":        0.55,
            "accessibility": 0.25,
            "complexity":    0.10,
        },
        "tips": [
            "Prefers footways and pedestrian paths",
            "Avoids complex junctions",
            "Values tactile paving and sidewalks",
        ],
    },
    "♿ Wheelchair User": {
        "description": "Needs accessible surfaces, lowered kerbs — avoids stairs and rough terrain",
        "emoji": "♿",
        "weights": {
            "distance":      0.15,
            "safety":        0.25,
            "accessibility": 0.50,
            "complexity":    0.10,
        },
        "tips": [
            "Requires paved/smooth surfaces",
            "Needs lowered or flush kerbs",
            "Avoids gravel, stairs, steep inclines",
        ],
    },
    "👴 Elderly Person": {
        "description": "Prefers shorter, safer routes — avoids high-speed roads",
        "emoji": "👴",
        "weights": {
            "distance":      0.30,
            "safety":        0.40,
            "accessibility": 0.20,
            "complexity":    0.10,
        },
        "tips": [
            "Prefers shorter walking distances",
            "Avoids busy roads and highways",
            "Benefits from resting spots and shelters",
        ],
    },
    "🚶 General Public": {
        "description": "Balanced route — reasonable distance and safety",
        "emoji": "🚶",
        "weights": {
            "distance":      0.25,
            "safety":        0.40,
            "accessibility": 0.25,
            "complexity":    0.10,
        },
        "tips": [
            "Balanced between distance and safety",
            "Standard navigation preferences",
        ],
    },
    "🎯 Custom (My Preferences)": {
        "description": "Set your own priorities using the sliders below",
        "emoji": "🎯",
        "weights": {
            "distance":      0.25,
            "safety":        0.40,
            "accessibility": 0.25,
            "complexity":    0.10,
        },
        "tips": ["Fully customisable — adjust sliders to match your needs"],
    },
}


def get_profile_names() -> List[str]:
    return list(USER_PROFILES.keys())


def get_weights(profile_name: str) -> Tuple[float, float, float, float]:
    """Return (distance, safety, accessibility, complexity) weights for a profile."""
    p = USER_PROFILES.get(profile_name, USER_PROFILES["🚶 General Public"])
    w = p["weights"]
    return (w["distance"], w["safety"], w["accessibility"], w["complexity"])


# ── Preference learning ───────────────────────────────────────────────────────

class PreferenceLearner:
    """
    Tracks user route selections and adapts objective weights over time.

    Algorithm: Exponential Moving Average (EMA) weight update.
    When a user picks a route, we infer which objectives they valued
    (the ones that route was relatively best at) and nudge the weights
    toward that direction.

    Learning rate alpha controls how fast the weights adapt:
      - High alpha (0.4): adapts quickly to recent choices
      - Low  alpha (0.1): slow, conservative adaptation
    """

    def __init__(self, profile: str = "🚶 General Public", alpha: float = 0.25):
        self.profile     = profile
        self.alpha       = alpha    # learning rate
        self.pick_count  = 0
        base_w           = USER_PROFILES[profile]["weights"]
        self.weights     = np.array([
            base_w["distance"],
            base_w["safety"],
            base_w["accessibility"],
            base_w["complexity"],
        ], dtype=float)
        self.history: List[dict] = []

    def record_pick(self, chosen_obj: list, pareto_front_objs: list) -> None:
        """
        Record that the user chose a route with objective values `chosen_obj`
        from a Pareto front whose objective matrix is `pareto_front_objs`.

        We infer which objectives the chosen route was relatively best at,
        and use that to update weights.
        """
        if not pareto_front_objs or not chosen_obj:
            return

        pf = np.array(pareto_front_objs)   # shape (N, 4)
        c  = np.array(chosen_obj)          # shape (4,)

        # Normalise each objective to [0, 1] within the Pareto front
        pf_min = pf.min(axis=0)
        pf_max = pf.max(axis=0)
        pf_range = np.where(pf_max - pf_min > 1e-9, pf_max - pf_min, 1.0)

        # rank of chosen solution per objective (0 = best)
        c_norm = (c - pf_min) / pf_range  # 0 = best, 1 = worst for minimise
        # invert: 1 = user clearly valued this objective (chose a route that was great at it)
        c_inv = 1.0 - c_norm

        # softmax to turn relative ranks into a probability vector (new weight signal)
        exp_c = np.exp(c_inv - c_inv.max())
        new_signal = exp_c / exp_c.sum()

        # EMA update: blend current weights toward the new signal
        self.weights = (1 - self.alpha) * self.weights + self.alpha * new_signal
        # re-normalise
        self.weights = self.weights / self.weights.sum()

        self.pick_count += 1
        self.history.append({
            "pick":   self.pick_count,
            "chosen": c.tolist(),
            "weights_after": self.weights.tolist(),
        })

    def get_weights(self) -> Tuple[float, float, float, float]:
        return tuple(self.weights.tolist())

    def weight_dict(self) -> Dict[str, float]:
        labels = ["distance", "safety", "accessibility", "complexity"]
        return {l: round(float(w), 3) for l, w in zip(labels, self.weights)}

    def confidence(self) -> str:
        """How confident are we in the learned preferences?"""
        if self.pick_count == 0:
            return "No data yet — using profile defaults"
        elif self.pick_count < 3:
            return f"Learning... ({self.pick_count} picks recorded)"
        elif self.pick_count < 7:
            return f"Getting personalised ({self.pick_count} picks)"
        else:
            return f"Well personalised! ({self.pick_count} picks recorded) ✅"

    def to_dict(self) -> dict:
        return {
            "profile":    self.profile,
            "alpha":      self.alpha,
            "pick_count": self.pick_count,
            "weights":    self.weights.tolist(),
            "history":    self.history[-20:],  # keep last 20 picks
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PreferenceLearner":
        obj = cls(profile=d.get("profile", "🚶 General Public"),
                  alpha=d.get("alpha", 0.25))
        obj.pick_count = d.get("pick_count", 0)
        obj.weights    = np.array(d.get("weights", obj.weights))
        obj.history    = d.get("history", [])
        return obj


# ── Persistence ───────────────────────────────────────────────────────────────

def save_learner(learner: PreferenceLearner, path: str = PREFS_FILE) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(learner.to_dict(), f, indent=2)


def load_learner(path: str = PREFS_FILE) -> Optional[PreferenceLearner]:
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            d = json.load(f)
        return PreferenceLearner.from_dict(d)
    except Exception:
        return None
