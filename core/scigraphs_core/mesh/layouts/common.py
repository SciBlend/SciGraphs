"""Shared layout imports, dependency checks, logging and graph helpers. Keep
Blender out: every layout does ``from .common import *``, so anything reading a
mesh datablock (``mesh_edge_pairs``) stays out too; pass edge lists in instead."""

import numpy as np
import time
import os
import sys
import math
import cmath
import json
import tempfile
from ...repro.determinism import get_layout_seed

# Optional deps bind to None on ImportError, so layouts can test ``if nx is None``.
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
    NETWORKX_REASON = None
except ImportError:
    nx = None
    NETWORKX_AVAILABLE = False
    NETWORKX_REASON = ("networkx is not installed; install the 'networkx' "
                       "extra to enable the NetworkX layouts (spring, "
                       "Kamada-Kawai, spectral, shell, hierarchical, "
                       "bipartite splitting)")
    print("Warning: %s" % NETWORKX_REASON)

try:
    from scipy.spatial import distance_matrix
    SCIPY_AVAILABLE = True
    SCIPY_REASON = None
except ImportError:
    distance_matrix = None
    SCIPY_AVAILABLE = False
    SCIPY_REASON = ("scipy is not installed; install the 'scipy' extra to "
                    "enable the distance-matrix and k-d tree paths")

_layout_rng = None

def _get_layout_rng(seed=None):
    global _layout_rng
    if seed is not None:
        _layout_rng = np.random.RandomState(seed)
    elif _layout_rng is None:
        _layout_rng = np.random.RandomState(get_layout_seed())
    return _layout_rng

def _reset_layout_rng(seed=None):
    """Reseed the layout RNG (call at the start of a layout)."""
    global _layout_rng
    if seed is None:
        seed = get_layout_seed()
    _layout_rng = np.random.RandomState(seed)
    return _layout_rng

try:
    import igraph as ig
    IGRAPH_AVAILABLE = True
except ImportError:
    IGRAPH_AVAILABLE = False
    print("Warning: python-igraph not available. Some fast layouts will be disabled.")

def _log_layout(algorithm, num_nodes, num_edges, params=None, start_time=None, success=True, error=None, actual_algorithm=None):
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

# fa2 rarely has a 3.11 wheel; prefer igraph DrL when missing.
FA2_AVAILABLE = False
try:
    from fa2 import ForceAtlas2
    FA2_AVAILABLE = True
    print("ForceAtlas2 available (optional)")
except ImportError:
    pass

GRAPHVIZ_AVAILABLE = False
try:
    import scigraphs_utils
    GRAPHVIZ_AVAILABLE = True
    print("scigraphs-utils available - Graphviz layouts enabled")
except ImportError:
    pass

def _build_networkx_graph(obj, edge_pairs=None):
    """Build a NetworkX graph from object data, ``(None, 0)`` on refusal. *obj*
    needs ``num_nodes`` plus ``nodes_data``/``edges_data``; mesh-native objects
    instead pass ``mesh_edge_pairs(...)`` as *edge_pairs*, where None means "not
    supplied" and is refused while ``[]`` means empty."""
    if not NETWORKX_AVAILABLE:
        print("Layout unavailable: %s" % NETWORKX_REASON)
        return None, 0

    if not obj or "num_nodes" not in obj:
        return None, 0

    num_nodes = obj["num_nodes"]

    if num_nodes == 0:
        return None, 0

    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))

    nodes_str = obj.get("nodes_data", "")
    if nodes_str:
        nodes_list = nodes_str.split(",")
    else:
        nodes_list = []

    edges_str = obj.get("edges_data", "")
    if edges_str:
        edges_flat = edges_str.split(",")
        edges_data = [(edges_flat[i], edges_flat[i+1]) for i in range(0, len(edges_flat), 2)]

        node_to_idx = {node: i for i, node in enumerate(nodes_list)}

        edge_indices = []
        for src, tgt in edges_data:
            if src in node_to_idx and tgt in node_to_idx:
                edge_indices.append((node_to_idx[src], node_to_idx[tgt]))
    elif edge_pairs is not None:
        edge_indices = list(edge_pairs)
    else:
        print("Layout unavailable: object %r stores no 'edges_data' and the "
              "caller supplied no edge_pairs, so the graph's edges are unknown. "
              "Mesh-native graph objects keep their topology in mesh.edges; "
              "read it with core.mesh.mesh_utils.mesh_edge_pairs(obj, "
              "num_nodes) and pass the result as edge_pairs. Refusing rather "
              "than laying out %d isolated nodes."
              % (getattr(obj, "name", type(obj).__name__), num_nodes))
        return None, 0

    G.add_edges_from(edge_indices)
    return G, num_nodes

def _get_drl_kwargs_from_props(props):
    """DrL per-phase parameters from scene properties."""
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
    g_igraph = ig.Graph()
    g_igraph.add_vertices(len(G.nodes()))
    g_igraph.add_edges(list(G.edges()))
    return g_igraph

__all__ = [name for name in globals() if not name.startswith('__')]
