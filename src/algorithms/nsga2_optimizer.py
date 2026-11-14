"""
nsga2_optimizer.py  —  NSGA-II for multi-objective route optimisation

This is the main algorithm of the project. NSGA-II finds a *set* of routes
that represent different trade-offs between our objectives — the Pareto front.
No single route on the Pareto front is "better" than another across all objectives;
each is optimal for a different preference.

I implemented this from scratch rather than using pymoo directly because
the route-finding problem doesn't fit pymoo's standard "vector of floats"
interface cleanly (chromosomes are variable-length node sequences, and crossover
has to happen at shared graph nodes). Using pymoo would have needed a custom
Problem class anyway, so it was easier to just write the whole thing.

References:
  Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002). A fast and elitist
  multiobjective genetic algorithm: NSGA-II. IEEE Transactions on Evolutionary
  Computation, 6(2), 182–197.
"""

import time
import math
import random
import networkx as nx
import numpy as np
from typing import List, Tuple, Optional


# ── Individual (one candidate route) ─────────────────────────────────────────

class Individual:
    """
    Represents a single candidate route (chromosome = list of node ids).
    objectives is a list [distance, risk, inaccessibility, complexity]
    where lower is always better (we convert maximise objectives by inverting).
    """

    __slots__ = ("path", "objectives", "rank", "crowding_dist")

    def __init__(self, path: list):
        self.path          = path
        self.objectives    = []     # filled in by evaluate()
        self.rank          = 0
        self.crowding_dist = 0.0

    def __repr__(self):
        return (f"Individual(len={len(self.path)}, "
                f"obj={[round(o, 3) for o in self.objectives]})")


# ── Objective evaluation ──────────────────────────────────────────────────────

def evaluate(ind: Individual, G: nx.MultiDiGraph) -> None:
    """
    Compute objective values for an individual.
    All objectives are to be MINIMISED:
        obj[0] = total distance (metres, normalised)
        obj[1] = average risk  (1 - safety_score)
        obj[2] = average inaccessibility (1 - accessibility)
        obj[3] = average complexity
    """
    path = ind.path
    if len(path) < 2:
        ind.objectives = [1e9, 1e9, 1e9, 1e9]
        return

    total_dist, risks, inaccs, comps = 0.0, [], [], []
    valid = True

    for u, v in zip(path[:-1], path[1:]):
        if not G.has_edge(u, v):
            valid = False
            break
        edges = G[u][v]
        # pick the most accessible parallel edge (if multiple exist)
        best_k = max(edges, key=lambda k: edges[k].get("accessibility", 0))
        d = edges[best_k]

        total_dist += d.get("distance_m",    50.0)
        risks.append(  1.0 - d.get("safety_score",  0.5))
        inaccs.append( 1.0 - d.get("accessibility", 0.5))
        comps.append(        d.get("complexity",     0.5))

    if not valid or not risks:
        ind.objectives = [1e9, 1e9, 1e9, 1e9]
        return

    n = len(risks)
    ind.objectives = [
        total_dist / 1000.0,       # convert to km-ish for nicer numbers
        sum(risks)  / n,
        sum(inaccs) / n,
        sum(comps)  / n,
    ]


# ── Pareto dominance ──────────────────────────────────────────────────────────

def dominates(a: Individual, b: Individual) -> bool:
    """Return True if a dominates b (a is at least as good on all objectives
    and strictly better on at least one)."""
    obj_a, obj_b = a.objectives, b.objectives
    at_least_as_good = all(x <= y for x, y in zip(obj_a, obj_b))
    strictly_better  = any(x <  y for x, y in zip(obj_a, obj_b))
    return at_least_as_good and strictly_better


def fast_non_dominated_sort(population: List[Individual]) -> List[List[Individual]]:
    """
    Non-dominated sorting — splits population into ranked fronts.
    Front 0 = Pareto-optimal set, front 1 = second-best, etc.
    """
    n = len(population)
    dominated_by   = [[] for _ in range(n)]   # dominated_by[i] = list of individuals that dominate i
    domination_cnt = [0] * n                   # how many individuals dominate i
    fronts         = [[]]

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if dominates(population[i], population[j]):
                dominated_by[i].append(j)
            elif dominates(population[j], population[i]):
                domination_cnt[i] += 1
        if domination_cnt[i] == 0:
            population[i].rank = 0
            fronts[0].append(population[i])

    current_front_idx = 0
    while fronts[current_front_idx]:
        next_front = []
        for ind in fronts[current_front_idx]:
            idx = population.index(ind)
            for j in dominated_by[idx]:
                domination_cnt[j] -= 1
                if domination_cnt[j] == 0:
                    population[j].rank = current_front_idx + 1
                    next_front.append(population[j])
        current_front_idx += 1
        fronts.append(next_front)

    return [f for f in fronts if f]   # drop empty last one


def crowding_distance(front: List[Individual]) -> None:
    """Assign crowding distance to individuals in a single front."""
    n = len(front)
    if n == 0:
        return
    for ind in front:
        ind.crowding_dist = 0.0

    n_obj = len(front[0].objectives)
    for m in range(n_obj):
        front.sort(key=lambda x: x.objectives[m])
        front[0].crowding_dist  = math.inf
        front[-1].crowding_dist = math.inf
        obj_range = front[-1].objectives[m] - front[0].objectives[m]
        if obj_range == 0:
            continue
        for i in range(1, n - 1):
            front[i].crowding_dist += (
                front[i + 1].objectives[m] - front[i - 1].objectives[m]
            ) / obj_range


def crowded_comparison(a: Individual, b: Individual) -> int:
    """Return -1 if a is better, 1 if b is better, 0 if equal."""
    if a.rank < b.rank:
        return -1
    if a.rank > b.rank:
        return 1
    # same rank — prefer higher crowding distance (more spread out)
    if a.crowding_dist > b.crowding_dist:
        return -1
    if a.crowding_dist < b.crowding_dist:
        return 1
    return 0


# ── Genetic operators ─────────────────────────────────────────────────────────

def _random_path(G: nx.MultiDiGraph, source: int, target: int,
                 rng: random.Random, max_len: int = 40) -> Optional[list]:
    """
    Generate a random walk from source toward target.
    We do a biased random walk — with probability 0.7 we pick a neighbour
    that is closer to the target (by node id heuristic), otherwise random.
    Returns None if we can't reach target within max_len steps.
    """
    path = [source]
    visited = {source}
    current = source

    for _ in range(max_len):
        if current == target:
            return path
        nbrs = [n for n in G.successors(current) if n not in visited]
        if not nbrs:
            break
        # biased toward target — not perfect but good enough
        if rng.random() < 0.7:
            # sort by node id proximity to target as a cheap heuristic
            nbrs.sort(key=lambda n: abs(n - target))
            nxt = nbrs[0]
        else:
            nxt = rng.choice(nbrs)
        path.append(nxt)
        visited.add(nxt)
        current = nxt

    # try to connect to target with shortest path as a fallback
    if current != target:
        try:
            tail = nx.shortest_path(G, current, target)
            # merge but skip overlap
            path = path + tail[1:]
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    return path if path[-1] == target else None


def crossover(parent_a: Individual, parent_b: Individual,
              rng: random.Random) -> Tuple[Individual, Individual]:
    """
    Path crossover at a shared node.
    If the two paths share a node other than start/end, we swap tails there.
    Otherwise we just return copies of the parents.
    """
    pa, pb = parent_a.path, parent_b.path
    shared = set(pa[1:-1]) & set(pb[1:-1])

    if not shared:
        return Individual(pa[:]), Individual(pb[:])

    pivot = rng.choice(list(shared))
    ia = pa.index(pivot)
    ib = pb.index(pivot)

    child_a = pa[:ia] + pb[ib:]
    child_b = pb[:ib] + pa[ia:]
    return Individual(child_a), Individual(child_b)


def mutate(ind: Individual, G: nx.MultiDiGraph, rng: random.Random,
           mutation_prob: float = 0.15) -> Individual:
    """
    Mutation: pick a random internal segment of the path and reroute it
    using a shortest path (by composite_cost) between the same endpoints.
    """
    path = ind.path
    if len(path) < 4 or rng.random() > mutation_prob:
        return ind

    # pick two random internal nodes
    idx = sorted(rng.sample(range(1, len(path) - 1), min(2, len(path) - 2)))
    if len(idx) < 2:
        return ind
    u_idx, v_idx = idx[0], idx[1]
    u, v = path[u_idx], path[v_idx]

    try:
        new_seg = nx.shortest_path(G, u, v, weight="composite_cost")
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return ind

    new_path = path[:u_idx] + new_seg + path[v_idx + 1:]
    return Individual(new_path)


def tournament_select(population: List[Individual], rng: random.Random,
                      k: int = 3) -> Individual:
    """Binary (k=2 default) or k-way tournament selection."""
    contestants = rng.sample(population, min(k, len(population)))
    contestants.sort(key=lambda x: (x.rank, -x.crowding_dist))
    return contestants[0]


# ── Main NSGA-II loop ─────────────────────────────────────────────────────────

def run(G: nx.MultiDiGraph, source: int, target: int,
        pop_size: int = 80, n_generations: int = 120,
        crossover_prob: float = 0.85, mutation_prob: float = 0.15,
        seed: int = 42) -> dict:
    """
    Run NSGA-II and return the Pareto front.

    Returns
    -------
    dict:
        pareto_front  : list of Individual objects on the Pareto front
        all_fronts    : all ranked fronts (for analysis)
        generations   : number of generations completed
        time_ms       : wall-clock runtime in ms
        algorithm     : "nsga2"
    """
    rng   = random.Random(seed)
    start = time.perf_counter()

    print(f"[nsga2] Initialising population (size={pop_size}) ...")

    # ── initialise population ─────────────────────────────────────────────
    population: List[Individual] = []
    attempts = 0
    while len(population) < pop_size and attempts < pop_size * 10:
        attempts += 1
        path = _random_path(G, source, target, rng)
        if path is not None and len(path) >= 2:
            ind = Individual(path)
            evaluate(ind, G)
            if ind.objectives[0] < 1e8:   # skip disconnected/broken paths
                population.append(ind)

    if len(population) < 4:
        return {
            "pareto_front": [], "all_fronts": [], "generations": 0,
            "time_ms": 0, "algorithm": "nsga2",
            "error": f"Could not initialise population (got {len(population)} individuals)",
        }

    print(f"[nsga2] Population initialised with {len(population)} individuals")

    fronts = fast_non_dominated_sort(population)
    for f in fronts:
        crowding_distance(f)

    # track convergence metrics each generation
    hv_history     = []
    spread_history = []

    # ── generational loop ─────────────────────────────────────────────────
    for gen in range(n_generations):
        offspring: List[Individual] = []

        while len(offspring) < pop_size:
            p1 = tournament_select(population, rng)
            p2 = tournament_select(population, rng)

            if rng.random() < crossover_prob:
                c1, c2 = crossover(p1, p2, rng)
            else:
                c1, c2 = Individual(p1.path[:]), Individual(p2.path[:])

            c1 = mutate(c1, G, rng, mutation_prob)
            c2 = mutate(c2, G, rng, mutation_prob)

            for child in (c1, c2):
                evaluate(child, G)
                if child.objectives[0] < 1e8:
                    offspring.append(child)

        # combine parent + offspring and select next generation
        combined = population + offspring
        fronts   = fast_non_dominated_sort(combined)
        for f in fronts:
            crowding_distance(f)

        next_gen: List[Individual] = []
        for f in fronts:
            if len(next_gen) + len(f) <= pop_size:
                next_gen.extend(f)
            else:
                # fill remaining slots by crowding distance (descending)
                remaining = pop_size - len(next_gen)
                f.sort(key=lambda x: -x.crowding_dist)
                next_gen.extend(f[:remaining])
                break
        population = next_gen

        # record convergence metrics every generation
        pareto_now = [i for i in population if i.rank == 0]
        hv_now     = _hypervolume_2d(pareto_now)
        sp_now     = _spread_2d(pareto_now)
        hv_history.append(hv_now)
        spread_history.append(sp_now)

        if (gen + 1) % 20 == 0:
            print(f"[nsga2] Gen {gen+1:3d}/{n_generations} | "
                  f"Pareto front size: {len(pareto_now)} | "
                  f"HV: {hv_now:.4f}")

    elapsed = (time.perf_counter() - start) * 1000.0
    final_fronts  = fast_non_dominated_sort(population)
    pareto_front  = [i for i in population if i.rank == 0]

    print(f"[nsga2] Done in {elapsed:.1f} ms | Pareto front: {len(pareto_front)} solutions")

    return {
        "pareto_front":   pareto_front,
        "all_fronts":     final_fronts,
        "generations":    n_generations,
        "time_ms":        round(elapsed, 3),
        "algorithm":      "nsga2",
        "hv_history":     hv_history,
        "spread_history": spread_history,
    }


def _hypervolume_2d(front: List[Individual]) -> float:
    """Quick 2-D hypervolume used inside the generational loop for convergence tracking."""
    pts = [(i.objectives[0], i.objectives[1]) for i in front if i.objectives[0] < 1e8]
    if not pts:
        return 0.0
    ref_x = max(p[0] for p in pts) + 0.1
    ref_y = max(p[1] for p in pts) + 0.1
    pts.sort(key=lambda p: p[0])
    hv, prev_y = 0.0, ref_y
    for x, y in pts:
        if y < prev_y:
            hv    += (ref_x - x) * (prev_y - y)
            prev_y = y
    return round(hv, 6)


def _spread_2d(front: List[Individual]) -> float:
    """Average consecutive distance on (obj0, obj1) — used for convergence tracking."""
    pts = sorted(
        [(i.objectives[0], i.objectives[1]) for i in front if i.objectives[0] < 1e8],
        key=lambda p: p[0]
    )
    if len(pts) < 2:
        return 0.0
    dists = [
        math.sqrt((pts[k+1][0]-pts[k][0])**2 + (pts[k+1][1]-pts[k][1])**2)
        for k in range(len(pts)-1)
    ]
    return round(sum(dists)/len(dists), 6)


def best_compromise(pareto_front: List[Individual],
                    weights: Tuple[float, ...] = (0.25, 0.40, 0.25, 0.10)) -> Optional[Individual]:
    """
    From the Pareto front, pick the solution with the smallest weighted sum
    of objectives (using user preference weights).
    Good for when you need to recommend ONE specific route.
    """
    if not pareto_front:
        return None

    # collect objective ranges for normalisation
    n_obj = len(pareto_front[0].objectives)
    obj_matrix = np.array([ind.objectives for ind in pareto_front])
    obj_min = obj_matrix.min(axis=0)
    obj_max = obj_matrix.max(axis=0)
    obj_range = np.where(obj_max - obj_min > 0, obj_max - obj_min, 1.0)

    best_ind   = None
    best_score = math.inf

    for ind in pareto_front:
        norm_obj = (np.array(ind.objectives) - obj_min) / obj_range
        score    = float(np.dot(norm_obj, weights[:n_obj]))
        if score < best_score:
            best_score = score
            best_ind   = ind

    return best_ind

