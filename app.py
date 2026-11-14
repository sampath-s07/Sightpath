"""
app.py  —  SightPath Streamlit Web Demo

Run with:  streamlit run app.py
"""

import os
import sys
import time
import random

sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import networkx as nx
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import NSGA2_PARAMS
from src.graph_builder  import get_graph
from src.feature_scorer import score_graph, normalise_graph_weights
from src.algorithms     import dijkstra, astar, nsga2_optimizer as nsga2
from src.evaluator      import (build_comparison_table, hypervolume, spread)
from src.visualizer     import (plot_graph, plot_routes,
                                plot_pareto_front, plot_metrics_bar)


# ── page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="SightPath — Assistive Navigation",
    page_icon="🦯",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* ── main background ── */
    .stApp { background-color: #f0fdf4; color: #0f172a; }
    .main  { background-color: #f0fdf4; }

    /* ── sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(160deg, #bae6fd 0%, #bbf7d0 100%);
    }
    [data-testid="stSidebar"] * { color: #0c4a6e !important; }

    /* ── headings ── */
    h1 { color: #0284c7 !important; }
    h2, h3 { color: #16a34a !important; }

    /* ── buttons ── */
    .stButton > button {
        background: linear-gradient(90deg, #38bdf8, #4ade80);
        color: white !important;
        border: none;
        border-radius: 8px;
        font-weight: 600;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #0ea5e9, #22c55e);
        transform: scale(1.02);
    }

    /* ── metric cards ── */
    [data-testid="stMetric"] {
        background: white;
        border-radius: 10px;
        padding: 12px;
        border-left: 4px solid #38bdf8;
        box-shadow: 0 2px 8px rgba(56,189,248,0.15);
    }
    [data-testid="stMetricLabel"]  { color: #0369a1 !important; }
    [data-testid="stMetricValue"]  { color: #15803d !important; }

    /* ── success / info boxes ── */
    .stSuccess { background-color: #dcfce7 !important; border-left: 4px solid #22c55e !important; }
    .stInfo    { background-color: #e0f2fe !important; border-left: 4px solid #38bdf8 !important; }
    .stWarning { background-color: #fef9c3 !important; }
    .stError   { background-color: #fee2e2 !important; }

    /* ── dataframe ── */
    [data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; }

    /* ── tabs ── */
    .stTabs [data-baseweb="tab-list"] { background: #e0f2fe; border-radius: 8px; }
    .stTabs [data-baseweb="tab"]      { color: #0369a1 !important; font-weight: 600; }
    .stTabs [aria-selected="true"]    { background: #38bdf8 !important; color: white !important; border-radius: 6px; }

    /* ── slider ── */
    [data-testid="stSlider"] > div > div > div { background: #38bdf8 !important; }

    /* ── progress bar ── */
    .stProgress > div > div { background: linear-gradient(90deg, #38bdf8, #4ade80) !important; }

    /* ── selectbox ── */
    [data-testid="stSelectbox"] > div { border-color: #38bdf8 !important; }

    /* ── general text ── */
    p, label, .stMarkdown { color: #1e3a5f; }
</style>
""", unsafe_allow_html=True)


# ── sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.image("https://img.icons8.com/fluency/96/blind.png", width=72)
st.sidebar.title("⚙️ Configuration")

seed = st.sidebar.slider("Random Seed", 1, 100, 42)

st.sidebar.markdown("---")
st.sidebar.subheader("NSGA-II Parameters")
pop_size   = st.sidebar.slider("Population Size", 20, 150, NSGA2_PARAMS["pop_size"])
n_gens     = st.sidebar.slider("Generations",     20, 200, NSGA2_PARAMS["n_generations"])
cross_prob  = st.sidebar.slider("Crossover Prob", 0.5, 1.0, NSGA2_PARAMS["crossover_prob"], 0.05)
mut_prob    = st.sidebar.slider("Mutation Prob",  0.05, 0.5, NSGA2_PARAMS["mutation_prob"], 0.05)

st.sidebar.markdown("---")
st.sidebar.markdown("**SightPath v1.0**  \nOptimisation Techniques — Final Project")


# ── main content ──────────────────────────────────────────────────────────────

st.title("🦯 SightPath")
st.markdown("### Multi-Objective Route Optimisation for Assistive Navigation")
st.markdown(
    "This system finds routes that balance **distance, safety, accessibility, and "
    "complexity** for visually impaired and mobility-challenged users. "
    "Three algorithms are compared: Dijkstra (baseline), A\\* (composite heuristic), "
    "and **NSGA-II** (Pareto multi-objective)."
)
st.markdown("---")


# ── load / cache graph ────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False, max_entries=3)
def load_scored_graph(seed_: int):
    try:
        G = get_graph()
        G = score_graph(G, seed=seed_)
        G = normalise_graph_weights(G)
        return G, None
    except Exception as e:
        return None, str(e)


with st.spinner("Loading Chennai road network from OpenStreetMap data ..."):
    try:
        G, err = load_scored_graph(seed)
        if err:
            st.error(f"Failed to load graph: {err}")
            st.stop()
        st.success(f"✅ Real Chennai OSM graph loaded — {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    except Exception as e:
        st.error(f"Failed to load graph: {e}")
        st.stop()


# ── pick endpoints ────────────────────────────────────────────────────────────

nodes = list(G.nodes())
rng   = random.Random(seed)
well_connected = [n for n in nodes if G.degree(n) >= 3] or nodes

col1, col2 = st.columns(2)
with col1:
    source = st.selectbox("🟡 Start Node", options=well_connected,
                          index=0, format_func=lambda x: f"Node {x}")
with col2:
    target_options = [n for n in well_connected if n != source]
    default_tgt_idx = min(len(target_options) - 1, len(target_options) // 2)
    target = st.selectbox("🔴 End Node",   options=target_options,
                          index=default_tgt_idx, format_func=lambda x: f"Node {x}")

if source == target:
    st.warning("Source and target must be different nodes.")
    st.stop()

try:
    test_path = nx.shortest_path(G, source, target)
    st.info(f"Direct shortest path exists ({len(test_path)} nodes)")
except nx.NetworkXNoPath:
    st.error("No path exists between these nodes. Please choose different endpoints.")
    st.stop()


# ── run algorithms ────────────────────────────────────────────────────────────

if st.button("🚀 Run Optimisation", type="primary", use_container_width=True):
    progress = st.progress(0, text="Starting ...")
    results_placeholder = st.empty()

    with st.spinner("Running Dijkstra ..."):
        t0 = time.time()
        dijkstra_results = dijkstra.run_all_objectives(G, source, target)
        t_dijk = (time.time() - t0) * 1000
    progress.progress(25, text="Dijkstra done ✓")

    with st.spinner("Running A* ..."):
        t0 = time.time()
        astar_result = astar.run(G, source, target)
        t_astar = (time.time() - t0) * 1000
    progress.progress(50, text="A* done ✓")

    with st.spinner(f"Running NSGA-II ({n_gens} generations) ..."):
        nsga2_result = nsga2.run(
            G, source, target,
            pop_size=pop_size, n_generations=n_gens,
            crossover_prob=cross_prob, mutation_prob=mut_prob,
            seed=seed,
        )
    progress.progress(85, text="NSGA-II done ✓")

    df = build_comparison_table(dijkstra_results, astar_result, nsga2_result)
    progress.progress(100, text="Done!")

    # ── display results ───────────────────────────────────────────────────────

    st.markdown("---")
    st.subheader("📊 Algorithm Comparison")

    if not df.empty:
        st.dataframe(df.style.format({
            "distance_m":    "{:.1f}",
            "safety":        "{:.4f}",
            "accessibility": "{:.4f}",
            "complexity":    "{:.4f}",
            "time_ms":       "{:.1f}",
        }), use_container_width=True)

    # NSGA-II summary stats
    pf = nsga2_result.get("pareto_front", [])
    if pf:
        st.markdown("#### NSGA-II Metrics")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Pareto Front Size", len(pf))
        m2.metric("Hypervolume (2D)", f"{hypervolume(pf):.4f}")
        m3.metric("Spread",           f"{spread(pf):.4f}")
        m4.metric("Time (ms)",        f"{nsga2_result.get('time_ms', 0):.1f}")

    # ── plots ─────────────────────────────────────────────────────────────────

    st.markdown("---")
    st.subheader("🗺️ Route Visualisations")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Route Comparison", "Safety Heatmap",
        "Pareto Front",     "Metrics Bar Chart"
    ])

    with tab1:
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
            img_path = plot_routes(G, routes, source, target, "outputs/route_cmp_app.png")
            st.image(img_path, use_container_width=True)
        else:
            st.warning("No routes to display.")

    with tab2:
        img_path = plot_graph(G, save_path="outputs/graph_app.png")
        st.image(img_path, use_container_width=True)

    with tab3:
        if pf:
            best_ind = nsga2.best_compromise(pf)
            img_path = plot_pareto_front(pf, highlight=best_ind,
                                         save_path="outputs/pareto_app.png")
            st.image(img_path, use_container_width=True)
            st.caption(
                "Each point = one Pareto-optimal route. "
                "No point is strictly better than another — they represent different trade-offs. "
                "The ★ marks the best-compromise route (using weighted user preferences)."
            )
        else:
            st.info("NSGA-II did not produce a Pareto front.")

    with tab4:
        if not df.empty:
            img_path = plot_metrics_bar(df, save_path="outputs/bar_app.png")
            st.image(img_path, use_container_width=True)

    st.balloons()
    st.success("Optimisation complete! All results saved to outputs/")

else:
    st.info("👆 Configure the parameters in the sidebar and click **Run Optimisation** to start.")

