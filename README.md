# SightPath — Multi-Objective Route Optimization for Assistive Navigation

A navigation system built for visually impaired and mobility-challenged users.
Instead of just finding the shortest path, it optimises across **four objectives simultaneously**:
distance, safety, accessibility, and route complexity.

This was done as part of the subject *Optimization Techniques for Intelligent Systems*.

---

## Why SightPath?

Standard GPS apps optimise for one thing — time or distance.
For users with visual impairment or mobility challenges, that's not enough.
A route through a busy intersection with no tactile paving might be short
but completely unusable. SightPath addresses this by treating navigation
as a **multi-objective optimisation problem**.

---

## Algorithms Used

| Algorithm | Type | What it does |
|---|---|---|
| **Dijkstra** | Single-objective baseline | Shortest path by one metric at a time |
| **A\*** | Composite heuristic | Weighted combination of all objectives |
| **NSGA-II** | True multi-objective (Pareto) | Finds a *set* of optimal trade-off routes |

NSGA-II is the main contribution — it generates a **Pareto front** of routes
where each solution is optimal for a different user preference profile.

---

## Objectives

```
Minimise:
  obj[0] = total distance (km)
  obj[1] = average risk (1 − safety_score)
  obj[2] = average inaccessibility (1 − accessibility)
  obj[3] = average route complexity
```

Edge features are extracted from OpenStreetMap tags
(`surface`, `wheelchair`, `tactile_paving`, `sidewalk`, `kerb`, `highway` etc.)
and assigned synthetic plausible values where OSM tags are missing.

---

## Project Structure

```
SightPath/
├── config.py                    # all settings in one place
├── main.py                      # run the full pipeline
├── app.py                       # streamlit demo
├── requirements.txt
├── src/
│   ├── graph_builder.py         # OSM + synthetic graph construction
│   ├── feature_scorer.py        # multi-objective edge scoring
│   ├── evaluator.py             # metrics and comparison table
│   ├── visualizer.py            # all plots
│   └── algorithms/
│       ├── dijkstra.py          # baseline
│       ├── astar.py             # composite heuristic A*
│       └── nsga2_optimizer.py   # NSGA-II from scratch
├── data/                        # cached OSM graphs
├── outputs/                     # generated plots and CSVs
└── notebooks/                   # exploration notebooks
```

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Running

**Full pipeline (synthetic graph, no internet needed):**
```bash
python main.py
```

**With real OSM data:**
```bash
python main.py --osm --area "Anna Nagar, Chennai, India"
```

**Interactive Streamlit demo:**
```bash
streamlit run app.py
```

---

## Outputs

After running, the `outputs/` folder contains:

- `graph_safety.png` — road network coloured by safety score
- `route_comparison.png` — all three algorithm routes on the same map
- `pareto_front.png` — NSGA-II Pareto front scatter plot
- `metrics_bar.png` — bar chart comparison across algorithms
- `comparison.csv` — full results table

---

## Key Results (expected)

- Dijkstra gives the shortest *distance* route but ignores safety
- A* balances objectives but is a single weighted trade-off
- NSGA-II produces multiple routes covering the full range of trade-offs,
  with higher average safety and accessibility scores compared to Dijkstra

---

## References

- Deb et al. (2002). A fast and elitist multiobjective genetic algorithm: NSGA-II.
  *IEEE Transactions on Evolutionary Computation*, 6(2), 182–197.
- Boeing, G. (2017). OSMnx: New methods for acquiring and constructing complex
  street networks. *Computers, Environment and Urban Systems*, 65, 126–139.
- Hart, Nilsson & Raphael (1968). A formal basis for the heuristic determination
  of minimum cost paths. *IEEE Transactions on Systems Science and Cybernetics*, 4(2).

