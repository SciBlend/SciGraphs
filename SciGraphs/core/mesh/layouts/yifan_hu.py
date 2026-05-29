"""Yifan Hu and Graphviz-backed layout algorithms."""

from .common import *
from .basic import _random_layout
from .networkx_layouts import _spring_layout_2d, _spring_layout_3d, _spectral_layout_3d, _generate_z_component

GRAPHVIZ_ENGINES = {
    'GRAPHVIZ_DOT': 'dot',
    'GRAPHVIZ_NEATO': 'neato',
    'GRAPHVIZ_FDP': 'fdp',
    'GRAPHVIZ_SFDP': 'sfdp',
    'GRAPHVIZ_TWOPI': 'twopi',
    'GRAPHVIZ_CIRCO': 'circo',
    'GRAPHVIZ_OSAGE': 'osage',
    'GRAPHVIZ_PATCHWORK': 'patchwork',
}

GRAPHVIZ_DIMENSION_ENGINES = {'neato', 'fdp', 'sfdp'}


def _graphviz_edges(G):
    """Return Graphviz-compatible edge indices for scigraphs-utils."""
    num_nodes = len(G.nodes())
    nodes_list = list(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(nodes_list)}
    edges = np.array(
        [[node_to_idx[u], node_to_idx[v]] for u, v in G.edges()],
        dtype=np.int32,
    )
    if edges.size == 0:
        edges = np.empty((0, 2), dtype=np.int32)
    return num_nodes, edges


def _parse_graphviz_attrs(raw_attrs):
    """Parse comma/newline separated key=value Graphviz attributes."""
    attrs = {}
    if not raw_attrs:
        return attrs

    for item in raw_attrs.replace("\n", ",").split(","):
        item = item.strip()
        if not item or "=" not in item:
            continue
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            attrs[key] = value
    return attrs


def _optional_float(value):
    return value if value and value > 0 else None


def _optional_int(value):
    return value if value and value > 0 else None


def _optional_string(value):
    return value if value and value != "DEFAULT" else None


def _graphviz_default_attrs(engine, iterations, props=None, dimension=None):
    """Map SciGraphs layout settings to Graphviz graph attributes."""
    K = getattr(props, "sfdp_k", 0.3) if props else 0.3
    maxiter = getattr(props, "sfdp_maxiter", iterations) if props else iterations
    overlap = getattr(props, "sfdp_overlap", "prism") if props else "prism"
    overlap_scaling = getattr(props, "sfdp_overlap_scaling", -4.0) if props else -4.0
    attrs = _parse_graphviz_attrs(getattr(props, "graphviz_extra_graph_attrs", "") if props else "")

    if engine in GRAPHVIZ_DIMENSION_ENGINES:
        dimension = dimension or (getattr(props, "graphviz_dimension", "2") if props else "2")
        dimension = "2" if dimension == "2Z" else dimension
        attrs.update({
            "dim": dimension,
            "dimen": dimension,
        })

    if engine in {'sfdp', 'fdp'}:
        attrs.update({
            "K": K,
            "maxiter": maxiter,
            "overlap": overlap,
        })
        if overlap == 'prism':
            attrs["overlap_scaling"] = overlap_scaling
        if engine == 'fdp':
            attrs["start"] = _optional_string(getattr(props, "graphviz_fdp_start", "") if props else "") or get_layout_seed()
        else:
            attrs.update({
                "start": get_layout_seed(),
                "repulsiveforce": getattr(props, "sfdp_repulsive_force", 1.0) if props else 1.0,
                "smoothing": getattr(props, "sfdp_smoothing", "spring") if props else "spring",
                "quadtree": getattr(props, "sfdp_quadtree", "normal") if props else "normal",
                "levels": _optional_int(getattr(props, "sfdp_levels", 0) if props else 0),
            })
            if getattr(props, "sfdp_beautify", False) if props else False:
                attrs["beautify"] = "true"
    elif engine == 'neato':
        attrs.update({
            "mode": _optional_string(getattr(props, "graphviz_neato_mode", "DEFAULT") if props else "DEFAULT"),
            "model": _optional_string(getattr(props, "graphviz_neato_model", "DEFAULT") if props else "DEFAULT"),
            "start": _optional_string(getattr(props, "graphviz_neato_start", "") if props else "") or get_layout_seed(),
            "maxiter": _optional_int(getattr(props, "graphviz_neato_maxiter", 0) if props else 0),
        })
    elif engine == 'dot':
        attrs.update({
            "rankdir": getattr(props, "graphviz_dot_rankdir", "TB") if props else "TB",
            "ranksep": _optional_float(getattr(props, "graphviz_dot_ranksep", 0.0) if props else 0.0),
            "nodesep": _optional_float(getattr(props, "graphviz_dot_nodesep", 0.0) if props else 0.0),
            "splines": getattr(props, "graphviz_dot_splines", "false") if props else "false",
        })
    elif engine == 'twopi':
        attrs.update({
            "root": _optional_string(getattr(props, "graphviz_twopi_root", "") if props else ""),
            "ranksep": _optional_float(getattr(props, "graphviz_twopi_ranksep", 0.0) if props else 0.0),
        })
    elif engine == 'circo':
        attrs["mindist"] = _optional_float(getattr(props, "graphviz_circo_mindist", 0.0) if props else 0.0)
    elif engine == 'osage':
        attrs.update({
            "pack": getattr(props, "graphviz_osage_pack", True) if props else True,
            "packmode": _optional_string(getattr(props, "graphviz_osage_packmode", "array") if props else "array"),
        })

    return {key: value for key, value in attrs.items() if value is not None}


def _scigraphs_utils_graphviz_layout(G, engine, iterations, scale, props=None, dimension=None):
    """Run a Graphviz layout through the bundled scigraphs-utils wheel."""
    from scigraphs_utils import graphviz_layout

    num_nodes, edges = _graphviz_edges(G)
    attrs = _graphviz_default_attrs(engine, iterations, props, dimension=dimension)
    node_attrs = _parse_graphviz_attrs(getattr(props, "graphviz_node_attrs", "") if props else "")
    edge_attrs = _parse_graphviz_attrs(getattr(props, "graphviz_edge_attrs", "") if props else "")
    directed = getattr(props, "graphviz_dot_directed", True) if engine == 'dot' and props else engine == 'dot'
    quiet = getattr(props, "graphviz_quiet", True) if props else True
    raw = graphviz_layout(
        num_nodes,
        edges,
        engine=engine,
        directed=directed,
        node_attrs=node_attrs,
        edge_attrs=edge_attrs,
        quiet=quiet,
        **attrs,
    )
    raw = np.asarray(raw, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != num_nodes or raw.shape[1] < 2:
        print(f"  scigraphs-utils returned invalid positions shape: {raw.shape}")
        raise ValueError(f"Invalid scigraphs-utils layout shape: {raw.shape}")
    graphviz_dim = dimension or (getattr(props, "graphviz_dimension", "2") if props else "2")
    if graphviz_dim == "3" and engine in GRAPHVIZ_DIMENSION_ENGINES and raw.shape[1] < 3:
        raise RuntimeError(
            "Native 3D was requested, but scigraphs-utils returned only XY coordinates. "
            "Install scigraphs-utils 0.1.1 or newer for native Yifan Hu 3D."
        )

    raw_range = raw.max(axis=0) - raw.min(axis=0)
    xy_max = max(raw_range[0], raw_range[1])
    raw = raw - raw.mean(axis=0)
    raw = raw / (xy_max if xy_max > 0 else 1.0)

    positions = np.zeros((num_nodes, 3), dtype=float)
    positions[:, 0] = raw[:, 0]
    positions[:, 1] = raw[:, 1]
    if raw.shape[1] >= 3:
        positions[:, 2] = raw[:, 2]
    positions *= scale

    if graphviz_dim == "2Z" and engine in GRAPHVIZ_DIMENSION_ENGINES:
        z_method = getattr(props, "sfdp_z_method", "SPECTRAL") if props else "SPECTRAL"
        z_scale = getattr(props, "sfdp_z_scale", 0.3) if props else 0.3
        z = _generate_z_component(G, num_nodes, z_method)
        positions[:, 2] = z * z_scale * scale
        print(f"  Generated Graphviz Z via {z_method}: {positions[:, 2].min():.3f} to {positions[:, 2].max():.3f}")

    print(f"  scigraphs-utils Graphviz {engine} computed {num_nodes} nodes")
    return positions


def _graphviz_engine_layout(G, algorithm, iterations, scale, props=None):
    """Compute any Graphviz layout exposed by scigraphs-utils."""
    engine = GRAPHVIZ_ENGINES[algorithm]
    return _scigraphs_utils_graphviz_layout(G, engine, iterations, scale, props)

def _yifan_hu_layout(G, iterations, scale, props=None):
    """
    Yifan Hu layout - Scalable Force-Directed Placement via scigraphs-utils sfdp.

    Dimension modes:
      '2'  - Flat 2D sfdp (Z=0)
      '2Z' - 2D sfdp XY + Z generated from graph structure (recommended)
      '3'  - Native Graphviz/scigraphs-utils 3D sfdp
    """
    import time
    start = time.time()

    num_nodes = len(G.nodes())
    print(f"Computing Yifan Hu layout for {num_nodes} nodes...")

    if num_nodes == 0:
        return np.zeros((0, 3))

    # Read sfdp params from props
    dim_mode = getattr(props, "sfdp_dim", "2Z") if props else '2Z'
    z_method = getattr(props, "sfdp_z_method", "SPECTRAL") if props else "SPECTRAL"
    z_scale = getattr(props, "sfdp_z_scale", 0.3) if props else 0.3

    print(f"  mode={dim_mode}, backend=scigraphs-utils, engine=sfdp")
    if dim_mode == '2Z':
        print(f"  z_method={z_method}, z_scale={z_scale}")

    dimension = dim_mode if dim_mode in {"2", "2Z", "3"} else "2Z"
    positions = _scigraphs_utils_graphviz_layout(G, "sfdp", iterations, scale, props, dimension=dimension)

    if dim_mode == '2Z':
        print(f"  Z range: {positions[:, 2].min():.3f} to {positions[:, 2].max():.3f}")

    final_range = positions.max(axis=0) - positions.min(axis=0)
    print(f"  Final ranges (scale={scale:.1f}): X={final_range[0]:.1f}, Y={final_range[1]:.1f}, Z={final_range[2]:.1f}")
    print(f"  Yifan Hu (scigraphs-utils sfdp, mode={dim_mode}) completed in {time.time() - start:.2f}s")
    return positions

__all__ = [name for name in globals() if not name.startswith('__')]
