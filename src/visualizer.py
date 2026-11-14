"""
visualizer.py  —  high-definition plotting logic for SightPath

Plots:
  1. Graph map — road network coloured by safety
  2. Accessibility heatmap — road network coloured by accessibility
  3. Route comparison — zoomed into the actual routes with equal aspect ratio
  4. Pareto front — scatter plot of NSGA-II solutions
  5. Metrics bar chart — compare algorithms across all objectives
  6. Convergence plot — NSGA-II hypervolume improving over generations
  7. Radar / spider chart — all 4 objectives compared visually
  8. Feature distribution histograms — edge feature analysis
"""

import os
import math
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from typing import Optional


OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def _get_node_positions(G: nx.MultiDiGraph) -> dict:
    """Extract (x, y) coordinates from node attributes."""
    pos = {}
    for node, data in G.nodes(data=True):
        x = data.get("x", data.get("lon", 0))
        y = data.get("y", data.get("lat", 0))
        pos[node] = (float(x), float(y))
    return pos


# ── 1. Safety Heatmap ────────────────────────────────────────────────────────

def plot_graph(G: nx.MultiDiGraph, title: str = "Chennai Road Network — Safety Heatmap",
               save_path: Optional[str] = None) -> str:
    """
    Draw the full street network coloured by safety score with equal aspect ratio.
    """
    pos = _get_node_positions(G)
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_facecolor("#0b132b")
    fig.patch.set_facecolor("#0b132b")

    cmap = cm.RdYlGn
    for u, v, data in G.edges(data=True):
        if u not in pos or v not in pos:
            continue
        safety = float(data.get("safety_score", 0.5))
        colour = cmap(np.clip(safety, 0, 1))
        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                color=colour, linewidth=1.2, alpha=0.75, zorder=2)

    xs = [pos[n][0] for n in G.nodes() if n in pos]
    ys = [pos[n][1] for n in G.nodes() if n in pos]
    if xs and ys:
        cx = (min(xs) + max(xs)) / 2.0
        cy = (min(ys) + max(ys)) / 2.0
        hspan = max(max(xs) - min(xs), max(ys) - min(ys)) / 2.0 * 1.05
        ax.set_xlim(cx - hspan, cx + hspan)
        ax.set_ylim(cy - hspan, cy + hspan)
        ax.set_aspect("equal", adjustable="box")
        ax.scatter(xs, ys, c="#e2e8f0", s=6, zorder=3, alpha=0.6)

    sm = cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(0, 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Safety Score (0 = High Risk, 1 = Safe Pedestrian)", color="white", fontsize=11)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    ax.set_title(title, color="#38bdf8", fontsize=15, pad=14, fontweight="bold")
    ax.tick_params(colors="#94a3b8")
    for spine in ax.spines.values():
        spine.set_edgecolor("#334155")

    path = save_path or os.path.join(OUTPUT_DIR, "graph_safety.png")
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[visualizer] Saved graph plot → {path}")
    return path


# ── 2. Accessibility Heatmap ──────────────────────────────────────────────────

def plot_accessibility_heatmap(G: nx.MultiDiGraph, save_path: Optional[str] = None) -> str:
    """
    Draw street network coloured by accessibility score (tactile paving, wheelchair, ramps).
    """
    pos = _get_node_positions(G)
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_facecolor("#0b132b")
    fig.patch.set_facecolor("#0b132b")

    cmap = cm.Blues
    for u, v, data in G.edges(data=True):
        if u not in pos or v not in pos:
            continue
        acc = float(data.get("accessibility", 0.5))
        colour = cmap(0.3 + 0.7 * np.clip(acc, 0, 1))
        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                color=colour, linewidth=1.2, alpha=0.8, zorder=2)

    xs = [pos[n][0] for n in G.nodes() if n in pos]
    ys = [pos[n][1] for n in G.nodes() if n in pos]
    if xs and ys:
        cx = (min(xs) + max(xs)) / 2.0
        cy = (min(ys) + max(ys)) / 2.0
        hspan = max(max(xs) - min(xs), max(ys) - min(ys)) / 2.0 * 1.05
        ax.set_xlim(cx - hspan, cx + hspan)
        ax.set_ylim(cy - hspan, cy + hspan)
        ax.set_aspect("equal", adjustable="box")
        ax.scatter(xs, ys, c="#e2e8f0", s=6, zorder=3, alpha=0.6)

    sm = cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(0, 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Accessibility Score (Tactile, Ramps, Surfaces)", color="white", fontsize=11)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    ax.set_title("Chennai Road Network — Accessibility Heatmap", color="#38bdf8", fontsize=15, pad=14, fontweight="bold")
    ax.tick_params(colors="#94a3b8")
    for spine in ax.spines.values():
        spine.set_edgecolor("#334155")

    path = save_path or os.path.join(OUTPUT_DIR, "graph_accessibility.png")
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[visualizer] Saved accessibility heatmap → {path}")
    return path


# ── 3. Route Comparison (Auto-Zoomed & Clear) ────────────────────────────────

ROUTE_STYLES = {
    "dijkstra": {"color": "#f87171", "width": 3.8, "style": "--", "label": "Dijkstra (Shortest)"},
    "astar":    {"color": "#38bdf8", "width": 4.0, "style": "-.", "label": "A* Composite"},
    "nsga2":    {"color": "#4ade80", "width": 4.5, "style": "-",  "label": "NSGA-II (Optimal Compromise)"},
}

def plot_routes(G: nx.MultiDiGraph,
                routes: dict,
                source: int,
                target: int,
                save_path: Optional[str] = None) -> str:
    """
    Draw routes zoomed into the active area with 1:1 true physical aspect ratio.
    """
    pos = _get_node_positions(G)
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_facecolor("#0f172a")
    fig.patch.set_facecolor("#0f172a")

    # Determine bounding box around the routes to zoom in perfectly
    active_nodes = []
    for p in routes.values():
        active_nodes.extend([n for n in p if n in pos])
    if source in pos: active_nodes.append(source)
    if target in pos: active_nodes.append(target)

    if active_nodes:
        xs = [pos[n][0] for n in active_nodes]
        ys = [pos[n][1] for n in active_nodes]
        span_x = max(xs) - min(xs)
        span_y = max(ys) - min(ys)
        pad = max(max(span_x, span_y) * 0.20, 180.0)

        center_x = (min(xs) + max(xs)) / 2.0
        center_y = (min(ys) + max(ys)) / 2.0
        half_span = max(span_x, span_y) / 2.0 + pad

        min_x, max_x = center_x - half_span, center_x + half_span
        min_y, max_y = center_y - half_span, center_y + half_span

        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)
        ax.set_aspect("equal", adjustable="box")
    else:
        min_x, max_x, min_y, max_y = -1e9, 1e9, -1e9, 1e9

    # Draw contextual street network within the visible region
    for u, v in G.edges():
        if u not in pos or v not in pos:
            continue
        xu, yu = pos[u]
        xv, yv = pos[v]
        # Only plot roads that touch the visible window
        if (min(xu, xv) <= max_x and max(xu, xv) >= min_x and
            min(yu, yv) <= max_y and max(yu, yv) >= min_y):
            ax.plot([xu, xv], [yu, yv], color="#334155", linewidth=1.0, alpha=0.55, zorder=1)

    # Draw routes with distinct styles and offset slightly if paths overlap
    legend_elements = []
    for label, path in routes.items():
        if not path or len(path) < 2:
            continue
        style = ROUTE_STYLES.get(label, {"color": "#fbbf24", "width": 3.0, "style": "-", "label": label.upper()})
        px = [pos[n][0] for n in path if n in pos]
        py = [pos[n][1] for n in path if n in pos]

        ax.plot(px, py, color=style["color"], linewidth=style["width"],
                linestyle=style["style"], alpha=0.92, zorder=4,
                solid_capstyle="round", solid_joinstyle="round")

        legend_elements.append(
            Line2D([0], [0], color=style["color"], linewidth=style["width"],
                   linestyle=style["style"], label=style["label"])
        )

    # Highlight Start and End points with prominent badges
    if source in pos:
        ax.scatter(*pos[source], c="#facc15", s=320, marker="*", zorder=7, edgecolors="black", linewidths=1.5)
        ax.annotate("  START", pos[source], color="#facc15", fontsize=11, fontweight="bold",
                    zorder=8, textcoords="offset points", xytext=(8, -4))

    if target in pos:
        ax.scatter(*pos[target], c="#f97316", s=260, marker="D", zorder=7, edgecolors="black", linewidths=1.5)
        ax.annotate("  DESTINATION", pos[target], color="#f97316", fontsize=11, fontweight="bold",
                    zorder=8, textcoords="offset points", xytext=(8, -4))

    ax.legend(handles=legend_elements, loc="upper left",
              facecolor="#1e293b", edgecolor="#475569",
              labelcolor="white", fontsize=11)
    ax.set_title("SightPath — Assistive Navigation Route Comparison", color="#38bdf8", fontsize=15, pad=14, fontweight="bold")
    ax.tick_params(colors="#94a3b8")
    ax.grid(True, color="#1e293b", linestyle=":", alpha=0.6)
    for spine in ax.spines.values():
        spine.set_edgecolor("#334155")

    path = save_path or os.path.join(OUTPUT_DIR, "route_comparison.png")
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[visualizer] Saved clear route comparison → {path}")
    return path


# ── 4. Pareto Front ───────────────────────────────────────────────────────────

def plot_pareto_front(pareto_front: list,
                      highlight: Optional[object] = None,
                      save_path: Optional[str] = None) -> str:
    """
    Scatter plot of NSGA-II Pareto front in (distance, risk) objective space.
    """
    if not pareto_front:
        return ""

    xs = [ind.objectives[0] for ind in pareto_front]
    ys = [ind.objectives[1] for ind in pareto_front]

    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")

    accs = [1.0 - ind.objectives[2] for ind in pareto_front]
    sc = ax.scatter(xs, ys, c=accs, cmap="YlGn", s=110, zorder=4,
                    edgecolors="#cbd5e1", linewidths=0.9, vmin=0, vmax=1)

    sorted_pts = sorted(zip(xs, ys), key=lambda p: p[0])
    ax.plot([p[0] for p in sorted_pts], [p[1] for p in sorted_pts],
            "--", color="#64748b", linewidth=1.5, alpha=0.7, zorder=3,
            label="Pareto Frontier")

    if highlight is not None:
        hx, hy = highlight.objectives[0], highlight.objectives[1]
        ax.scatter(hx, hy, c="#f97316", s=280, zorder=6,
                   marker="*", edgecolors="white", linewidths=1.5,
                   label="Best Compromise Route")

    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Accessibility Score", color="white", fontsize=11)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    ax.set_xlabel("Route Distance (km)", color="white", fontsize=12)
    ax.set_ylabel("Average Risk (1 − Safety)", color="white", fontsize=12)
    ax.set_title("NSGA-II Multi-Objective Pareto Front", color="#38bdf8", fontsize=14, pad=12, fontweight="bold")
    ax.tick_params(colors="white")
    ax.grid(True, color="#334155", linestyle="--", alpha=0.5)
    for spine in ax.spines.values():
        spine.set_edgecolor("#475569")

    ax.legend(facecolor="#1e293b", edgecolor="#475569", labelcolor="white", fontsize=10)

    path = save_path or os.path.join(OUTPUT_DIR, "pareto_front.png")
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[visualizer] Saved Pareto front → {path}")
    return path


# ── 5. Metrics Bar Chart ──────────────────────────────────────────────────────

def plot_metrics_bar(df, save_path: Optional[str] = None) -> str:
    if df.empty:
        return ""

    metrics   = ["safety", "accessibility", "complexity"]
    labels    = df["label"].tolist()
    x         = np.arange(len(labels))
    width     = 0.25

    fig, ax = plt.subplots(figsize=(13, 6))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")

    bar_colours = ["#4ade80", "#38bdf8", "#f87171"]
    for i, (metric, colour) in enumerate(zip(metrics, bar_colours)):
        vals = df[metric].tolist() if metric in df.columns else [0] * len(labels)
        ax.bar(x + i * width, vals, width, label=metric.capitalize(),
               color=colour, alpha=0.9, edgecolor="#0f172a")

    ax.set_xticks(x + width)
    ax.set_xticklabels(labels, rotation=20, ha="right", color="white", fontsize=10)
    ax.set_ylabel("Score (0 to 1)", color="white", fontsize=12)
    ax.set_title("Algorithm Performance Across Objectives", color="#38bdf8", fontsize=14, pad=12, fontweight="bold")
    ax.tick_params(colors="white")
    ax.set_ylim(0, 1.15)
    for spine in ax.spines.values():
        spine.set_edgecolor("#475569")
    ax.legend(facecolor="#1e293b", edgecolor="#475569", labelcolor="white", fontsize=10)

    path = save_path or os.path.join(OUTPUT_DIR, "metrics_bar.png")
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[visualizer] Saved metrics bar chart → {path}")
    return path


# ── 6. Convergence Plot ───────────────────────────────────────────────────────

def plot_convergence(hv_history: list, spread_history: list, save_path: Optional[str] = None) -> str:
    if not hv_history:
        return ""

    gens = list(range(1, len(hv_history) + 1))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.patch.set_facecolor("#0f172a")

    for ax in (ax1, ax2):
        ax.set_facecolor("#1e293b")
        for spine in ax.spines.values():
            spine.set_edgecolor("#475569")
        ax.tick_params(colors="white")
        ax.grid(True, color="#334155", linestyle=":", alpha=0.5)

    ax1.plot(gens, hv_history, color="#4ade80", linewidth=2.5, label="Hypervolume Indicator")
    ax1.fill_between(gens, hv_history, alpha=0.18, color="#4ade80")
    ax1.set_ylabel("Hypervolume (HV)", color="white", fontsize=11)
    ax1.set_title("NSGA-II Evolution Convergence Over Generations", color="#38bdf8", fontsize=13, pad=10, fontweight="bold")
    ax1.legend(facecolor="#1e293b", edgecolor="#475569", labelcolor="white")

    ax2.plot(gens, spread_history, color="#38bdf8", linewidth=2.5, label="Spread (Diversity Metric)")
    ax2.fill_between(gens, spread_history, alpha=0.18, color="#38bdf8")
    ax2.set_xlabel("Generations", color="white", fontsize=11)
    ax2.set_ylabel("Spread (Δ)", color="white", fontsize=11)
    ax2.legend(facecolor="#1e293b", edgecolor="#475569", labelcolor="white")

    path = save_path or os.path.join(OUTPUT_DIR, "convergence.png")
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[visualizer] Saved convergence plot → {path}")
    return path


# ── 7. Radar Chart ────────────────────────────────────────────────────────────

def plot_radar_chart(df, save_path: Optional[str] = None) -> str:
    if df.empty:
        return ""

    categories = ["Safety", "Accessibility", "Low Complexity", "Short Distance"]
    n_cats     = len(categories)
    angles     = [n / float(n_cats) * 2 * math.pi for n in range(n_cats)]
    angles    += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, color="white", fontsize=11, fontweight="bold")
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], color="#94a3b8", fontsize=9)
    ax.set_ylim(0, 1)
    ax.spines["polar"].set_color("#475569")
    ax.grid(color="#334155", linewidth=0.8)

    colours  = ["#f87171", "#fbbf24", "#38bdf8", "#4ade80", "#a855f7"]
    max_dist = df["distance_m"].max() if "distance_m" in df.columns else 1

    for i, (_, row) in enumerate(df.iterrows()):
        safety   = float(row.get("safety", 0))
        acc      = float(row.get("accessibility", 0))
        low_comp = 1.0 - float(row.get("complexity", 0))
        short_d  = 1.0 - float(row.get("distance_m", 0)) / (max_dist + 1)

        values = [safety, acc, low_comp, short_d]
        values += values[:1]

        col   = colours[i % len(colours)]
        label = str(row.get("label", f"Alg {i}"))
        ax.plot(angles, values, linewidth=2.4, color=col, label=label)
        ax.fill(angles, values, alpha=0.10, color=col)

    ax.set_title("Algorithm Multi-Criteria Evaluation\n(Higher = Better)",
                 color="#38bdf8", fontsize=13, pad=22, fontweight="bold")
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15),
              facecolor="#1e293b", edgecolor="#475569", labelcolor="white", fontsize=9)

    path = save_path or os.path.join(OUTPUT_DIR, "radar_chart.png")
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[visualizer] Saved radar chart → {path}")
    return path


# ── 8. Feature Distributions ──────────────────────────────────────────────────

def plot_feature_distributions(G: nx.MultiDiGraph, save_path: Optional[str] = None) -> str:
    features = ["safety_score", "accessibility", "complexity", "distance_m"]
    labels   = ["Safety Score", "Accessibility", "Complexity", "Distance (meters)"]
    colours  = ["#4ade80", "#38bdf8", "#f87171", "#fbbf24"]

    data = {f: [] for f in features}
    for _, _, d in G.edges(data=True):
        for f in features:
            if f in d:
                data[f].append(float(d[f]))

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.patch.set_facecolor("#0f172a")
    axes = axes.flatten()

    for ax, feat, label, col in zip(axes, features, labels, colours):
        ax.set_facecolor("#1e293b")
        vals = data[feat]
        if vals:
            ax.hist(vals, bins=30, color=col, alpha=0.85, edgecolor="#0f172a", linewidth=0.5)
            mean_v = sum(vals) / len(vals)
            ax.axvline(mean_v, color="white", linestyle="--", linewidth=1.5, label=f"Mean: {mean_v:.2f}")
            ax.legend(facecolor="#1e293b", edgecolor="#475569", labelcolor="white", fontsize=9)
        ax.set_title(label, color="white", fontsize=11, fontweight="bold")
        ax.set_xlabel("Score / Value", color="#94a3b8", fontsize=9)
        ax.set_ylabel("Edge Count", color="#94a3b8", fontsize=9)
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#475569")

    fig.suptitle("Chennai Road Network — Attribute Distributions", color="#38bdf8", fontsize=14, y=1.01, fontweight="bold")
    path = save_path or os.path.join(OUTPUT_DIR, "feature_distributions.png")
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[visualizer] Saved feature distributions → {path}")
    return path
