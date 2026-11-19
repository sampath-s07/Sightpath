"""
app.py  —  SightPath Streamlit Web Demo  (v2.0 with ML)

Run with:  python -m streamlit run app.py

New in v2:
  - User profiles (Blind Person, Wheelchair User, Elderly, General)
  - ML-based safety & accessibility scoring (Random Forest)
  - Preference learning — system adapts to your route choices over time
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
from src.ml_safety_model import train_models, load_models, apply_ml_scores
from src.user_profiles   import (USER_PROFILES, get_profile_names, get_weights,
                                  PreferenceLearner, save_learner, load_learner)
from src.algorithms      import dijkstra, astar, nsga2_optimizer as nsga2
from src.evaluator       import (build_comparison_table, hypervolume, spread)
from src.visualizer      import (plot_graph, plot_routes,
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
    .stApp { background-color: #f0fdf4; color: #0f172a; }
    .main  { background-color: #f0fdf4; }
    [data-testid="stSidebar"] {
        background: linear-gradient(160deg, #bae6fd 0%, #bbf7d0 100%);
    }
    [data-testid="stSidebar"] * { color: #0c4a6e !important; }
    h1 { color: #0284c7 !important; }
    h2, h3 { color: #16a34a !important; }
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
    [data-testid="stMetric"] {
        background: white;
        border-radius: 10px;
        padding: 12px;
        border-left: 4px solid #38bdf8;
        box-shadow: 0 2px 8px rgba(56,189,248,0.15);
    }
    [data-testid="stMetricLabel"]  { color: #0369a1 !important; }
    [data-testid="stMetricValue"]  { color: #15803d !important; }
    .stSuccess { background-color: #dcfce7 !important; border-left: 4px solid #22c55e !important; }
    .stInfo    { background-color: #e0f2fe !important; border-left: 4px solid #38bdf8 !important; }
    .stTabs [data-baseweb="tab-list"] { background: #e0f2fe; border-radius: 8px; }
    .stTabs [data-baseweb="tab"]      { color: #0369a1 !important; font-weight: 600; }
    .stTabs [aria-selected="true"]    { background: #38bdf8 !important; color: white !important; border-radius: 6px; }
    [data-testid="stSlider"] > div > div > div { background: #38bdf8 !important; }
    .stProgress > div > div { background: linear-gradient(90deg, #38bdf8, #4ade80) !important; }
    p, label, .stMarkdown { color: #1e3a5f; }
</style>
""", unsafe_allow_html=True)


# ── sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.image("https://img.icons8.com/fluency/96/blind.png", width=72)
st.sidebar.title("⚙️ Configuration")

# ── User profile ──────────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.subheader("👤 Who Are You?")

profile_names = get_profile_names()
selected_profile = st.sidebar.selectbox(
    "Select your profile",
    profile_names,
    index=3,  # default: General Public
)

profile_info = USER_PROFILES[selected_profile]
st.sidebar.info(f"{profile_info['emoji']} {profile_info['description']}")

# ── Preference weights ────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Your Priorities")

if selected_profile == "🎯 Custom (My Preferences)":
    # fully manual sliders
    w_distance = st.sidebar.slider("📏 Distance",      0.05, 1.0, 0.25, 0.05)
    w_safety   = st.sidebar.slider("🛡️ Safety",        0.05, 1.0, 0.40, 0.05)
    w_access   = st.sidebar.slider("♿ Accessibility", 0.05, 1.0, 0.25, 0.05)
    w_complex  = st.sidebar.slider("🧩 Complexity",    0.05, 1.0, 0.10, 0.05)
    total = w_distance + w_safety + w_access + w_complex
    pref_weights = (w_distance/total, w_safety/total, w_access/total, w_complex/total)
else:
    pw = profile_info["weights"]
    pref_weights = (pw["distance"], pw["safety"], pw["accessibility"], pw["complexity"])
    # show read-only progress bars
    st.sidebar.markdown(f"📏 Distance: **{pref_weights[0]:.0%}**")
    st.sidebar.progress(pref_weights[0])
    st.sidebar.markdown(f"🛡️ Safety: **{pref_weights[1]:.0%}**")
    st.sidebar.progress(pref_weights[1])
    st.sidebar.markdown(f"♿ Accessibility: **{pref_weights[2]:.0%}**")
    st.sidebar.progress(pref_weights[2])
    st.sidebar.markdown(f"🧩 Complexity: **{pref_weights[3]:.0%}**")
    st.sidebar.progress(pref_weights[3])

# ── ML Settings ───────────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.subheader("🤖 ML Safety Model")
use_ml_scoring = st.sidebar.toggle("Use ML-based safety scoring", value=True)
st.sidebar.caption("Trains a Random Forest model on OSM features for more accurate safety/accessibility scores.")

# ── NSGA-II parameters ────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.subheader("NSGA-II Parameters")
seed       = st.sidebar.slider("Random Seed",    1, 100, 42)
pop_size   = st.sidebar.slider("Population Size",20, 150, NSGA2_PARAMS["pop_size"])
n_gens     = st.sidebar.slider("Generations",    20, 200, NSGA2_PARAMS["n_generations"])
cross_prob  = st.sidebar.slider("Crossover Prob",0.5, 1.0, NSGA2_PARAMS["crossover_prob"], 0.05)
mut_prob    = st.sidebar.slider("Mutation Prob", 0.05, 0.5, NSGA2_PARAMS["mutation_prob"], 0.05)

st.sidebar.markdown("---")
st.sidebar.markdown("**SightPath v2.0**  \nML-Enhanced Assistive Navigation")


# ── main content ──────────────────────────────────────────────────────────────

st.title("🦯 SightPath")
st.markdown("### Multi-Objective Route Optimisation for Assistive Navigation")
st.markdown(
    "This system finds routes that balance **distance, safety, accessibility, and "
    "complexity** for visually impaired and mobility-challenged users. "
    "Three algorithms are compared: Dijkstra (baseline), A\\* (composite heuristic), "
    "and **NSGA-II** (Pareto multi-objective). Now powered by **ML safety scoring** 🤖"
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


@st.cache_resource(show_spinner=False, max_entries=3)
def load_ml_scored_graph(seed_: int):
    """Load graph, score it, train/load ML models, apply ML scores — all cached."""
    import copy
    try:
        G, err = load_scored_graph(seed_)
        if err or G is None:
            return None, None, str(err)

        safety_model, access_model = load_models()
        if safety_model is None:
            safety_model, access_model, metrics = train_models(G, verbose=False)
        else:
            metrics = {"note": "Loaded from disk", "safety_mae": "N/A", "safety_r2": "N/A", "access_mae": "N/A"}

        G_ml = copy.deepcopy(G)
        G_ml = apply_ml_scores(G_ml, safety_model, access_model)
        return G_ml, metrics, None
    except Exception as e:
        return None, None, str(e)


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

# Apply ML scoring if enabled — fully cached, no re-training on reload
G_active = G
if use_ml_scoring:
    with st.spinner("🤖 Loading ML safety model..."):
        try:
            G_ml, ml_metrics, ml_err = load_ml_scored_graph(seed)
            if ml_err:
                st.warning(f"ML model error: {ml_err} — using rule-based scores.")
            elif G_ml is not None:
                G_active = G_ml
                st.info(
                    f"🤖 ML Safety Model active (Random Forest) — "
                    f"Safety MAE: {ml_metrics.get('safety_mae', 'N/A')} | "
                    f"R²: {ml_metrics.get('safety_r2', 'N/A')}"
                )
        except Exception as e:
            st.warning(f"ML model error: {e} — using rule-based scores.")


# ── preference learner state ──────────────────────────────────────────────────

if "learner" not in st.session_state:
    saved = load_learner()
    if saved and saved.profile == selected_profile:
        st.session_state.learner = saved
    else:
        st.session_state.learner = PreferenceLearner(profile=selected_profile)

learner: PreferenceLearner = st.session_state.learner

# If profile changed, reset learner
if learner.profile != selected_profile:
    st.session_state.learner = PreferenceLearner(profile=selected_profile)
    learner = st.session_state.learner

# Show learning status
if learner.pick_count > 0:
    st.info(f"🧠 **Preference Learning:** {learner.confidence()}  |  "
            f"Learned weights — Safety: {learner.weights[1]:.0%}, "
            f"Accessibility: {learner.weights[2]:.0%}, "
            f"Distance: {learner.weights[0]:.0%}")
    # Use learned weights if enough picks
    if learner.pick_count >= 3:
        pref_weights = learner.get_weights()

# ── pick endpoints ────────────────────────────────────────────────────────────

nodes          = list(G_active.nodes())
rng            = random.Random(seed)
well_connected = [n for n in nodes if G_active.degree(n) >= 3] or nodes

col1, col2 = st.columns(2)
with col1:
    source = st.selectbox("🟡 Start Node", options=well_connected,
                          index=0, format_func=lambda x: f"Node {x}")
with col2:
    target_options   = [n for n in well_connected if n != source]
    default_tgt_idx  = min(len(target_options) - 1, len(target_options) // 2)
    target = st.selectbox("🔴 End Node",   options=target_options,
                          index=default_tgt_idx, format_func=lambda x: f"Node {x}")

if source == target:
    st.warning("Source and target must be different nodes.")
    st.stop()

try:
    test_path = nx.shortest_path(G_active, source, target)
    st.info(f"Direct shortest path exists ({len(test_path)} nodes)")
except nx.NetworkXNoPath:
    st.error("No path exists between these nodes. Please choose different endpoints.")
    st.stop()


# ── run algorithms ────────────────────────────────────────────────────────────

if st.button("🚀 Run Optimisation", type="primary", use_container_width=True):
    progress = st.progress(0, text="Starting ...")

    with st.spinner("Running Dijkstra ..."):
        t0 = time.time()
        dijkstra_results = dijkstra.run_all_objectives(G_active, source, target)
        t_dijk = (time.time() - t0) * 1000
    progress.progress(25, text="Dijkstra done ✓")

    with st.spinner("Running A* ..."):
        t0 = time.time()
        astar_result = astar.run(G_active, source, target)
        t_astar = (time.time() - t0) * 1000
    progress.progress(50, text="A* done ✓")

    with st.spinner(f"Running NSGA-II ({n_gens} generations) ..."):
        nsga2_result = nsga2.run(
            G_active, source, target,
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

    # ── NSGA-II Pareto front ──────────────────────────────────────────────────
    pf = nsga2_result.get("pareto_front", [])
    if pf:
        # Pick best compromise using the USER's personalised weights
        best_ind = nsga2.best_compromise(pf, weights=pref_weights)

        st.markdown("#### 🤖 ML + NSGA-II Personalised Result")
        profile_emoji = USER_PROFILES[selected_profile]["emoji"]
        st.success(
            f"{profile_emoji} **Best route for {selected_profile}** — "
            f"Distance: {best_ind.objectives[0]*1000:.0f} m | "
            f"Safety: {1-best_ind.objectives[1]:.2f} | "
            f"Accessibility: {1-best_ind.objectives[2]:.2f}"
        )

        # Weight display
        wl = ["Distance", "Safety", "Accessibility", "Complexity"]
        wv = list(pref_weights)
        wcols = st.columns(4)
        for col, label, val in zip(wcols, wl, wv):
            col.metric(label + " weight", f"{val:.0%}")

        st.markdown("#### NSGA-II Metrics")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Pareto Front Size", len(pf))
        m2.metric("Hypervolume (2D)",  f"{hypervolume(pf):.4f}")
        m3.metric("Spread",            f"{spread(pf):.4f}")
        m4.metric("Time (ms)",         f"{nsga2_result.get('time_ms', 0):.1f}")

    # ── Preference learning — let user rate / pick their preferred route ───────
    st.markdown("---")
    st.subheader("🧠 Preference Learning")
    st.markdown(
        "Tell the system which objective matters most to you — "
        "it will learn and personalise your routes over time."
    )

    if pf:
        pf_obj = [ind.objectives for ind in pf]

        pick_col1, pick_col2 = st.columns([2, 1])
        with pick_col1:
            priority_choice = st.radio(
                "Which did you care most about for THIS route?",
                ["🏃 Shortest Distance", "🛡️ Safest Route",
                 "♿ Most Accessible", "⚖️ Best Balance"],
                horizontal=True,
            )
        with pick_col2:
            if st.button("✅ Record My Preference", use_container_width=True):
                # pick a Pareto solution matching the stated preference
                if priority_choice == "🏃 Shortest Distance":
                    chosen = min(pf, key=lambda x: x.objectives[0])
                elif priority_choice == "🛡️ Safest Route":
                    chosen = min(pf, key=lambda x: x.objectives[1])
                elif priority_choice == "♿ Most Accessible":
                    chosen = min(pf, key=lambda x: x.objectives[2])
                else:
                    chosen = best_ind

                learner.record_pick(chosen.objectives, pf_obj)
                save_learner(learner)
                st.success(
                    f"✅ Preference recorded! ({learner.pick_count} total picks)  \n"
                    f"Updated weights — Safety: {learner.weights[1]:.0%}, "
                    f"Accessibility: {learner.weights[2]:.0%}"
                )
                st.rerun()

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
        if pf and best_ind and best_ind.path:
            routes["nsga2"] = best_ind.path
        if routes:
            img_path = plot_routes(G_active, routes, source, target, "outputs/route_cmp_app.png")
            st.image(img_path, use_container_width=True)
            st.caption(
                f"Blue = Dijkstra | Orange = A* | Green = NSGA-II best for **{selected_profile}**"
            )
        else:
            st.warning("No routes to display.")

    with tab2:
        img_path = plot_graph(G_active, save_path="outputs/graph_app.png")
        st.image(img_path, use_container_width=True)

    with tab3:
        if pf:
            img_path = plot_pareto_front(pf, highlight=best_ind,
                                         save_path="outputs/pareto_app.png")
            st.image(img_path, use_container_width=True)
            st.caption(
                "Each point = one Pareto-optimal route. "
                "No point is strictly better than another — different trade-offs. "
                f"★ marks the best route for {selected_profile}."
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
    # Show profile tips on the landing page
    profile_tips = profile_info.get("tips", [])
    if profile_tips:
        st.subheader(f"{profile_info['emoji']} Tips for {selected_profile}")
        for tip in profile_tips:
            st.markdown(f"• {tip}")
    st.info("👆 Configure your profile in the sidebar and click **Run Optimisation** to start.")
