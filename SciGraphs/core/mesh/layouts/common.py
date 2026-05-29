"""Shared layout imports, dependency checks, logging, and graph helpers."""

import bpy
import numpy as np
import networkx as nx
from scipy.spatial import distance_matrix
import time
import os
import sys
import math
import cmath
import json
import tempfile
from ...repro.determinism import get_layout_seed

# Module-level RNG for reproducible layouts
_layout_rng = None

def _get_layout_rng(seed=None):
    """Get a reproducible random number generator for layout operations."""
    global _layout_rng
    if seed is not None:
        _layout_rng = np.random.RandomState(seed)
    elif _layout_rng is None:
        _layout_rng = np.random.RandomState(get_layout_seed())
    return _layout_rng

def _reset_layout_rng(seed=None):
    """Reset the layout RNG with a new seed (call at start of layout operations)."""
    global _layout_rng
    if seed is None:
        seed = get_layout_seed()
    _layout_rng = np.random.RandomState(seed)
    return _layout_rng

# Fast layout libraries
try:
    import igraph as ig
    IGRAPH_AVAILABLE = True
except ImportError:
    IGRAPH_AVAILABLE = False
    print("Warning: python-igraph not available. Some fast layouts will be disabled.")

def _log_layout(algorithm, num_nodes, num_edges, params=None, start_time=None, success=True, error=None, actual_algorithm=None):
    """Log layout algorithm execution with parameters and result."""
    separator = "=" * 70
    print(f"\n{separator}")

    if actual_algorithm and actual_algorithm != algorithm:
        print(f"Layout algorithm: {algorithm}")
        print(f"Fallback: using {actual_algorithm} instead")
        print(f"   Reason: {algorithm} library not available")
    else:
        print(f"Layout algorithm: {algorithm}")

    print(f"Graph: {num_nodes} nodes, {num_edges} edges")

    if params:
        print("Parameters:")
        for key, value in params.items():
            if isinstance(value, float):
                print(f"  - {key}: {value:.4f}")
            else:
                print(f"  - {key}: {value}")

    if start_time is not None:
        elapsed = time.time() - start_time
        print(f"Execution time: {elapsed:.3f}s")

    if success:
        if actual_algorithm and actual_algorithm != algorithm:
            print("Status: success (with fallback)")
        else:
            print("Status: success")
    else:
        print("Status: failed")
        if error:
            print(f"   Error: {error}")

    print(f"{separator}\n")

# ForceAtlas2: Optional, most implementations don't have Python 3.11 wheels
# We use igraph's DrL as alternative (similar performance, better support)
FA2_AVAILABLE = False
try:
    from fa2 import ForceAtlas2
    FA2_AVAILABLE = True
    print("ForceAtlas2 available (optional)")
except ImportError:
    pass  # Not critical, igraph provides similar algorithms

# Graphviz layouts via bundled scigraphs-utils
GRAPHVIZ_AVAILABLE = False
try:
    import scigraphs_utils
    GRAPHVIZ_AVAILABLE = True
    print("scigraphs-utils available - Graphviz layouts enabled")
except ImportError:
    pass

def _build_networkx_graph(obj):
    """Helper function to build NetworkX graph from object data."""
    if not obj or "num_nodes" not in obj:
        return None, 0

    num_nodes = obj["num_nodes"]

    if num_nodes == 0:
        return None, 0

    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))

    # Parse nodes from stored string
    nodes_str = obj.get("nodes_data", "")
    if nodes_str:
        nodes_list = nodes_str.split(",")
    else:
        nodes_list = []

    # Parse edges from stored string
    edges_str = obj.get("edges_data", "")
    if edges_str:
        edges_flat = edges_str.split(",")
        edges_data = [(edges_flat[i], edges_flat[i+1]) for i in range(0, len(edges_flat), 2)]
    else:
        edges_data = []

    node_to_idx = {node: i for i, node in enumerate(nodes_list)}

    edge_indices = []
    for src, tgt in edges_data:
        if src in node_to_idx and tgt in node_to_idx:
            edge_indices.append((node_to_idx[src], node_to_idx[tgt]))

    G.add_edges_from(edge_indices)
    return G, num_nodes

def _get_drl_kwargs_from_props(props):
    """Extract all DrL per-phase parameters from Blender scene properties."""
    return dict(
        edge_cut=props.igraph_drl_edge_cut,
        init_iterations=props.igraph_drl_init_iterations,
        init_temperature=props.igraph_drl_init_temperature,
        init_attraction=props.igraph_drl_init_attraction,
        init_damping_mult=props.igraph_drl_init_damping_mult,
        liquid_iterations=props.igraph_drl_liquid_iterations,
        liquid_temperature=props.igraph_drl_liquid_temperature,
        liquid_attraction=props.igraph_drl_liquid_attraction,
        liquid_damping_mult=props.igraph_drl_liquid_damping_mult,
        expansion_iterations=props.igraph_drl_expansion_iterations,
        expansion_temperature=props.igraph_drl_expansion_temperature,
        expansion_attraction=props.igraph_drl_expansion_attraction,
        expansion_damping_mult=props.igraph_drl_expansion_damping_mult,
        cooldown_iterations=props.igraph_drl_cooldown_iterations,
        cooldown_temperature=props.igraph_drl_cooldown_temperature,
        cooldown_attraction=props.igraph_drl_cooldown_attraction,
        cooldown_damping_mult=props.igraph_drl_cooldown_damping_mult,
        crunch_iterations=props.igraph_drl_crunch_iterations,
        crunch_temperature=props.igraph_drl_crunch_temperature,
        crunch_attraction=props.igraph_drl_crunch_attraction,
        crunch_damping_mult=props.igraph_drl_crunch_damping_mult,
        simmer_iterations=props.igraph_drl_simmer_iterations,
        simmer_temperature=props.igraph_drl_simmer_temperature,
        simmer_attraction=props.igraph_drl_simmer_attraction,
        simmer_damping_mult=props.igraph_drl_simmer_damping_mult,
    )

def _nx_to_igraph(G):
    """Convert NetworkX graph to igraph graph."""
    g_igraph = ig.Graph()
    g_igraph.add_vertices(len(G.nodes()))
    g_igraph.add_edges(list(G.edges()))
    return g_igraph

__all__ = [name for name in globals() if not name.startswith('__')]
