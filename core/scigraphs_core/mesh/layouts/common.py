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
import inspect
import random
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
    """The shared layout RandomState. NetworkX takes it directly as ``seed=``,
    so a layout never needs the global ``np.random``."""
    global _layout_rng
    if seed is not None:
        _layout_rng = np.random.RandomState(seed)
    elif _layout_rng is None:
        _layout_rng = np.random.RandomState(get_layout_seed())
    return _layout_rng

def _reset_layout_rng(seed=None):
    """Reseed the layout RNG (call at the start of a layout). Also seeds the
    stdlib ``random``, process-wide: igraph draws from it and takes no seed
    argument, so that is the only way its layouts reproduce."""
    global _layout_rng
    if seed is None:
        seed = get_layout_seed()
    _layout_rng = np.random.RandomState(seed)
    random.seed(seed)
    return _layout_rng

_props_capable = {}

def _accepts_props(fn):
    """True when *fn* takes a ``props`` keyword."""
    try:
        cached = _props_capable[fn]
    except KeyError:
        try:
            params = inspect.signature(fn).parameters
        except (TypeError, ValueError):
            cached = False
        else:
            cached = 'props' in params
        _props_capable[fn] = cached
    return cached

def _call_with_props(fn, *args, props=None, **kwargs):
    """Call *fn*, forwarding *props* only if its signature takes it, so a layout
    that has not grown the parameter yet still runs."""
    if props is not None and _accepts_props(fn):
        kwargs['props'] = props
    return fn(*args, **kwargs)

try:
    import igraph as ig
    IGRAPH_AVAILABLE = True
except ImportError:
    IGRAPH_AVAILABLE = False
    print("Warning: python-igraph not available. Some fast layouts will be disabled.")

def _log_layout(algorithm, num_nodes, num_edges, params=None, start_time=None, success=True, error=None, actual_algorithm=None, missing_library=None):
    separator = "=" * 70
    print(f"\n{separator}")

    print(f"Layout algorithm: {algorithm}")
    if actual_algorithm and actual_algorithm != algorithm:
        print(f"Fallback: using {actual_algorithm} instead")
        print(f"   Reason: {missing_library or 'required'} library not available")

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
    from fa2_modified import ForceAtlas2
    FA2_AVAILABLE = True
except ImportError:
    try:
        from fa2 import ForceAtlas2
        FA2_AVAILABLE = True
    except ImportError:
        pass
if FA2_AVAILABLE:
    print("ForceAtlas2 available (optional)")

GRAPHVIZ_AVAILABLE = False
try:
    import scigraphs_utils
    GRAPHVIZ_AVAILABLE = True
    print("scigraphs-utils available - Graphviz layouts enabled")
except ImportError:
    pass

_IGRAPH_FALLBACKS = {
    'IGRAPH_FR': 'SPRING (2D)',
    'IGRAPH_KK': 'SPRING (2D)',
    'IGRAPH_DRL': 'RANDOM',
    'IGRAPH_DRL_2D': 'RANDOM',
    'IGRAPH_LGL': 'RANDOM',
    'IGRAPH_DH': 'SPRING (3D)',
    'IGRAPH_GRAPHOPT': 'SPRING (3D)',
}

def _resolve_fallback(algorithm):
    """``(missing_library, substitute)`` when *algorithm* cannot run as asked,
    ``(None, None)`` when it can. The Graphviz engines have no fallback: they
    raise, which the dispatcher reports as a failure rather than as a layout."""
    if algorithm == 'FORCEATLAS2':
        if getattr(nx, "forceatlas2_layout", None) is not None:
            return None, None
        if FA2_AVAILABLE:
            return "networkx >= 3.4", "FORCEATLAS2 (2D, fa2 package)"
        return "networkx >= 3.4 or fa2", "SPRING (2D)"
    if not IGRAPH_AVAILABLE and algorithm in _IGRAPH_FALLBACKS:
        return "python-igraph", _IGRAPH_FALLBACKS[algorithm]
    return None, None

def _check_positions(pos, num_nodes, algorithm):
    """Return *pos* as a finite ``(num_nodes, 3)`` float array, raising if it is
    not: NaN, inf or a short array would go straight into vertex coordinates."""
    arr = np.asarray(pos, dtype=np.float64)
    if arr.shape != (num_nodes, 3):
        raise ValueError("%s produced positions of shape %s, expected (%d, 3)"
                         % (algorithm, arr.shape, num_nodes))
    if not np.isfinite(arr).all():
        raise ValueError("%s produced %d non-finite coordinate(s)"
                         % (algorithm, int((~np.isfinite(arr)).sum())))
    return arr

def _edge_pairs_to_indices(edge_pairs, num_nodes, name):
    """Validate caller-supplied edge pairs, or None on refusal. An index outside
    the node range used to add nodes the mesh does not have."""
    edge_indices = []
    for pair in edge_pairs:
        try:
            src, tgt = (int(v) for v in pair)
        except (TypeError, ValueError):
            print("Layout unavailable: object %r has the malformed edge %r."
                  % (name, pair))
            return None
        if not (0 <= src < num_nodes and 0 <= tgt < num_nodes):
            print("Layout unavailable: object %r has the edge (%d, %d), outside "
                  "its 0..%d nodes. Laying it out would add nodes the mesh does "
                  "not have and silently drop or misplace positions."
                  % (name, src, tgt, num_nodes - 1))
            return None
        edge_indices.append((src, tgt))
    return edge_indices

def _build_networkx_graph(obj, edge_pairs=None):
    """Build a NetworkX graph from object data, ``(None, 0)`` on refusal. *obj*
    needs ``num_nodes`` plus ``nodes_data``/``edges_data``; mesh-native objects
    instead pass ``mesh_edge_pairs(...)`` as *edge_pairs*, where None means "not
    supplied" and is refused while ``[]`` means empty. *edge_pairs* wins over
    ``edges_data``, being the object's real topology. Data that cannot describe
    exactly ``num_nodes`` nodes is refused, not turned into another graph."""
    if not NETWORKX_AVAILABLE:
        print("Layout unavailable: %s" % NETWORKX_REASON)
        return None, 0

    if not obj or "num_nodes" not in obj:
        return None, 0

    name = getattr(obj, "name", type(obj).__name__)

    try:
        num_nodes = int(obj["num_nodes"])
    except (TypeError, ValueError):
        print("Layout unavailable: object %r stores the non-numeric num_nodes %r."
              % (name, obj["num_nodes"]))
        return None, 0

    if num_nodes < 0:
        print("Layout unavailable: object %r stores num_nodes=%d; a node count "
              "cannot be negative." % (name, num_nodes))
        return None, 0

    if num_nodes == 0:
        return None, 0

    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))

    nodes_str = obj.get("nodes_data", "")
    nodes_list = nodes_str.split(",") if nodes_str else []
    if nodes_list and len(nodes_list) != num_nodes:
        print("Layout unavailable: object %r stores %d 'nodes_data' identifiers "
              "for %d nodes. A comma inside a node name splits into extra "
              "identifiers and shifts every edge after it; quote it out of the "
              "source or rename the node." % (name, len(nodes_list), num_nodes))
        return None, 0

    edges_str = obj.get("edges_data", "")

    if edge_pairs is not None:
        edge_indices = _edge_pairs_to_indices(edge_pairs, num_nodes, name)
        if edge_indices is None:
            return None, 0
        if edges_str and len(edges_str.split(",")) // 2 != len(edge_indices):
            print("Note: object %r also stores 'edges_data' describing %d edges; "
                  "the caller's %d edge_pairs are the mesh's own topology and win."
                  % (name, len(edges_str.split(",")) // 2, len(edge_indices)))
    elif edges_str:
        edges_flat = edges_str.split(",")
        if len(edges_flat) % 2:
            print("Layout unavailable: object %r stores %d 'edges_data' "
                  "identifiers, an odd number, so they do not pair into edges. "
                  "A comma inside a node name does this."
                  % (name, len(edges_flat)))
            return None, 0
        if not nodes_list:
            print("Layout unavailable: object %r stores 'edges_data' but no "
                  "'nodes_data', so its identifiers resolve to no node at all. "
                  "Store the identifiers, or pass the topology as edge_pairs."
                  % name)
            return None, 0

        node_to_idx = {node: i for i, node in enumerate(nodes_list)}

        edge_indices = []
        unknown = 0
        for i in range(0, len(edges_flat), 2):
            src, tgt = edges_flat[i], edges_flat[i + 1]
            if src in node_to_idx and tgt in node_to_idx:
                edge_indices.append((node_to_idx[src], node_to_idx[tgt]))
            else:
                unknown += 1
        if unknown:
            print("Warning: object %r has %d 'edges_data' entries naming nodes "
                  "absent from 'nodes_data'; they are dropped." % (name, unknown))
    else:
        print("Layout unavailable: object %r stores no 'edges_data' and the "
              "caller supplied no edge_pairs, so the graph's edges are unknown. "
              "Mesh-native graph objects keep their topology in mesh.edges; "
              "read it with core.mesh.mesh_utils.mesh_edge_pairs(obj, "
              "num_nodes) and pass the result as edge_pairs. Refusing rather "
              "than laying out %d isolated nodes." % (name, num_nodes))
        return None, 0

    G.add_edges_from(edge_indices)
    return G, num_nodes

def _positive_prop(props, name, minimum=0.0):
    """A property's value when it is above *minimum*, else None, for the layouts
    that read None as "use your own default". Never hand None to one that
    forwards it to igraph: igraph wants a real number."""
    value = getattr(props, name, None)
    if value is None or value <= minimum:
        return None
    return value

def _graphopt_kwargs_from_props(props):
    """Graphopt parameters from scene properties. A property left at its "auto"
    zero is omitted, so :func:`_igraph_graphopt` keeps its own default; passing
    None instead raises TypeError inside igraph."""
    kwargs = {}
    for keyword, prop_name in (('niter', 'igraph_graphopt_niter'),
                               ('node_charge', 'igraph_graphopt_node_charge'),
                               ('node_mass', 'igraph_graphopt_node_mass'),
                               ('spring_length', 'igraph_graphopt_spring_length'),
                               ('spring_constant', 'igraph_graphopt_spring_constant'),
                               ('max_sa_movement', 'igraph_graphopt_max_sa_movement')):
        value = _positive_prop(props, prop_name)
        if value is not None:
            kwargs[keyword] = value
    return kwargs

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
