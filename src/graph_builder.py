"""
graph_builder.py

Loads the real Chennai road network from data/chennai.osm.
Cleans spatial outliers and extracts the primary connected component
so navigation and visualizations are crisp and 100% reliable.
"""

import os
import pickle
import numpy as np
import networkx as nx
import osmnx as ox

from config import DATA_DIR, OSM_TIMEOUT


OSM_LOCAL_FILE = os.path.join(DATA_DIR, "chennai.osm")
CACHE_FILE     = os.path.join(DATA_DIR, "chennai_graph.pkl")


def clean_graph(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """
    Remove extreme spatial outlier nodes (e.g., stray transit/boundary nodes
    spanning 50km away from Chennai export).
    """
    if G.number_of_nodes() < 5:
        return G

    xs = [d.get("x", 0) for _, d in G.nodes(data=True)]
    ys = [d.get("y", 0) for _, d in G.nodes(data=True)]

    med_x = float(np.median(xs))
    med_y = float(np.median(ys))

    # Any node more than 8km away from median of the exported area is an outlier
    outliers = [
        n for n, d in G.nodes(data=True)
        if abs(d.get("x", 0) - med_x) > 8000 or abs(d.get("y", 0) - med_y) > 8000
    ]
    if outliers:
        print(f"[graph_builder] Removing {len(outliers)} spatial outlier node(s)...")
        G.remove_nodes_from(outliers)

    # Keep the largest weakly connected component so every node is reachable
    if not nx.is_weakly_connected(G):
        largest_comp = max(nx.weakly_connected_components(G), key=len)
        G = G.subgraph(largest_comp).copy()

    return G


def get_graph() -> nx.MultiDiGraph:
    """
    Load the Chennai road network graph.
    Uses cached pickle if available.
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(CACHE_FILE):
        print(f"[graph_builder] Loading cached graph ...")
        with open(CACHE_FILE, "rb") as f:
            G = pickle.load(f)
        print(f"[graph_builder] Graph loaded — nodes: {G.number_of_nodes()}, edges: {G.number_of_edges()}")
        return G

    if not os.path.exists(OSM_LOCAL_FILE):
        raise FileNotFoundError(
            f"OSM file not found: {OSM_LOCAL_FILE}\nPlease save your downloaded file to data/chennai.osm"
        )

    print(f"[graph_builder] Parsing {OSM_LOCAL_FILE} ...")
    ox.settings.timeout = OSM_TIMEOUT
    G = ox.graph_from_xml(OSM_LOCAL_FILE, retain_all=False)
    G = ox.project_graph(G)   # project to UTM (meters)
    G = clean_graph(G)

    print(f"[graph_builder] Cleaned real OSM graph — nodes: {G.number_of_nodes()}, edges: {G.number_of_edges()}")

    with open(CACHE_FILE, "wb") as f:
        pickle.dump(G, f)
    print(f"[graph_builder] Cached to {CACHE_FILE}")

    return G
