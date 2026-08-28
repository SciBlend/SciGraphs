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

GRAPHVIZ_NATIVE_3D_ENGINES = {'neato', 'sfdp'}

UNSAFE_SFDP_SMOOTHING = {'spring', 'avg_dist', 'graph_dist', 'power_dist'}

GRAPHVIZ_VORONOI_OVERLAP = {'false', 'prism'}
GRAPHVIZ_VORONOI_MAX_NODES = 500

_libc_rng_handle = None
_libc_rng_looked_up = False


def _libc_rng():
    """Handle exposing the C srand/srand48 Graphviz draws from, or None."""
    global _libc_rng_handle, _libc_rng_looked_up
    if _libc_rng_looked_up:
        return _libc_rng_handle
    _libc_rng_looked_up = True
    try:
        import ctypes
        import ctypes.util
        if sys.platform == "win32":
            candidates = ["ucrtbase", "msvcrt"]
        else:
            candidates = [ctypes.util.find_library("c"), None]
        for name in candidates:
            try:
                lib = ctypes.CDLL(name)
            except OSError:
                continue
            try:
                lib.srand
            except AttributeError:
                continue
            _libc_rng_handle = lib
            break
    except Exception as error:
        print(f"  Could not reach the C RNG for Graphviz seeding: {error}")
    return _libc_rng_handle


def _seed_graphviz_rng(seed):
    """Seed the C RNG before calling Graphviz, and report whether it worked.

    sfdp coarsens with gv_permutation (Multilevel.c), which calls rand() before
    spring_electrical.c ever reaches its own srand(). Without this the first
    sfdp layout in a process differs from every later one."""
    lib = _libc_rng()
    if lib is None:
        return False
    import ctypes
    seeded = False
    for name, argtype in (("srand", ctypes.c_uint), ("srand48", ctypes.c_long)):
        try:
            func = getattr(lib, name)
        except AttributeError:
            continue
        func.argtypes = [argtype]
        func.restype = None
        func(seed)
        seeded = True
    return seeded


def _has_isolated_node(G):
    """True for a degree-0 or self-loop-only node; sfdp aborts on these."""
    directed = G.is_directed()
    for node in G.nodes():
        linked = any(other != node for other in G.neighbors(node))
        if not linked and directed:
            linked = any(other != node for other in G.predecessors(node))
        if not linked:
            return True
    return False


def _unsafe_sfdp_smoothing(value):
    """True unless the value is a no-op smoothing.

    late_smooth (sfdpinit.c) also accepts digits, so smoothing='2' reaches the
    aborting code that the string names are checked for. Everything except
    'none' is treated as unsafe: triangle, rng and unknown names all fall back
    to SMOOTHING_NONE in this build anyway, so nothing is lost."""
    if value is None:
        return False
    return str(value).strip().lower() not in {'none', '0', ''}


def _graphviz_edges(G):
    """Edge index pairs for scigraphs-utils.

    _build_networkx_graph labels nodes 0..n-1, so the array usually comes
    straight off the edge view instead of through a Python remap."""
    num_nodes = len(G.nodes())
    nodes_list = list(G.nodes())
    if nodes_list == list(range(num_nodes)):
        edges = np.asarray(G.edges(), dtype=np.int32)
    else:
        node_to_idx = {n: i for i, n in enumerate(nodes_list)}
        edges = np.array(
            [[node_to_idx[u], node_to_idx[v]] for u, v in G.edges()],
            dtype=np.int32,
        )
    if edges.size == 0:
        edges = np.empty((0, 2), dtype=np.int32)
    return num_nodes, edges


def _parse_graphviz_attrs(raw_attrs):
    """Parse key=value Graphviz attributes separated by newlines, ';' or ','.

    Commas inside quotes are kept, so size="10,10" and label="Hello, World"
    survive; a fragment with no '=' is reported rather than dropped."""
    attrs = {}
    if not raw_attrs:
        return attrs

    items = []
    current = []
    quote = None
    for char in raw_attrs:
        if quote:
            current.append(char)
            if char == quote:
                quote = None
        elif char in "\"'":
            quote = char
            current.append(char)
        elif char in ",;\n":
            items.append("".join(current))
            current = []
        else:
            current.append(char)
    items.append("".join(current))

    for item in items:
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            print(f"  Ignoring Graphviz attribute fragment {item!r}: no '='. "
                  f"Quote values holding a comma, as in size=\"10,10\"")
            continue
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key:
            attrs[key] = value
    return attrs


def _optional_float(value):
    return value if value and value > 0 else None


def _optional_int(value):
    return value if value and value > 0 else None


def _optional_string(value):
    return value if value and value != "DEFAULT" else None


GRAPHVIZ_QUADTREE_HYBRID_SIZE = 10_000


def _resolve_quadtree(quadtree, num_nodes):
    if quadtree != 'AUTO':
        return quadtree
    return 'fast' if num_nodes > GRAPHVIZ_QUADTREE_HYBRID_SIZE else 'normal'


def _resolve_overlap(overlap, num_nodes):
    """Map a UI overlap mode onto what this Graphviz build really does."""
    if overlap is None:
        return None
    overlap = str(overlap).strip()
    if overlap == 'scale':
        return 'true'
    if overlap in GRAPHVIZ_VORONOI_OVERLAP:
        if num_nodes > GRAPHVIZ_VORONOI_MAX_NODES:
            print(f"  overlap='{overlap}' runs Graphviz's Voronoi loop, which grows "
                  f"about n^2.3 (269 s at 1000 nodes); skipping it for {num_nodes} nodes")
            return 'true'
        if overlap == 'prism':
            print("  overlap='prism' is unsupported by this Graphviz build; it runs "
                  "the same Voronoi pass as 'false'")
        return 'false'
    return overlap


def _graphviz_default_attrs(engine, iterations, props=None, dimension=None, num_nodes=0):
    K = getattr(props, "sfdp_k", 0.3) if props else 0.3
    maxiter = getattr(props, "sfdp_maxiter", iterations) if props else iterations
    overlap = getattr(props, "sfdp_overlap", "scale") if props else "scale"
    attrs = {}

    if engine in GRAPHVIZ_NATIVE_3D_ENGINES:
        dimension = dimension or (getattr(props, "graphviz_dimension", "2") if props else "2")
        if dimension == "3":
            attrs.update({
                "dim": "3",
                "dimen": "3",
            })

    if engine in {'sfdp', 'fdp'}:
        attrs.update({
            "K": K,
            "overlap": overlap,
        })
        if engine == 'fdp':
            attrs["maxiter"] = maxiter
            attrs["start"] = _optional_string(getattr(props, "graphviz_fdp_start", "") if props else "") or get_layout_seed()
        else:
            attrs.update({
                "start": get_layout_seed(),
                "repulsiveforce": getattr(props, "sfdp_repulsive_force", 1.0) if props else 1.0,
                "smoothing": getattr(props, "sfdp_smoothing", "spring") if props else "spring",
                "quadtree": getattr(props, "sfdp_quadtree", "AUTO") if props else "AUTO",
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

    attrs.update(_parse_graphviz_attrs(getattr(props, "graphviz_extra_graph_attrs", "") if props else ""))
    if "overlap" in attrs:
        attrs["overlap"] = _resolve_overlap(attrs["overlap"], num_nodes)
    if "quadtree" in attrs:
        attrs["quadtree"] = _resolve_quadtree(attrs["quadtree"], num_nodes)

    return {key: value for key, value in attrs.items() if value is not None}


def _scigraphs_utils_graphviz_layout(G, engine, iterations, scale, props=None, dimension=None):
    from scigraphs_utils import graphviz_layout

    num_nodes, edges = _graphviz_edges(G)
    if num_nodes == 0:
        return np.zeros((0, 3))

    attrs = _graphviz_default_attrs(engine, iterations, props, dimension=dimension, num_nodes=num_nodes)
    if engine == 'sfdp' and _unsafe_sfdp_smoothing(attrs.get("smoothing")) and _has_isolated_node(G):
        print(
            f"  Isolated node detected - sfdp smoothing '{attrs['smoothing']}' "
            f"would abort Graphviz, falling back to 'none'"
        )
        attrs["smoothing"] = "none"
    node_attrs = _parse_graphviz_attrs(getattr(props, "graphviz_node_attrs", "") if props else "")
    edge_attrs = _parse_graphviz_attrs(getattr(props, "graphviz_edge_attrs", "") if props else "")
    directed = getattr(props, "graphviz_dot_directed", True) if engine == 'dot' and props else engine == 'dot'
    quiet = getattr(props, "graphviz_quiet", True) if props else True
    if not _seed_graphviz_rng(get_layout_seed()) and engine == 'sfdp':
        print("  Warning: could not seed the C RNG, so sfdp is not reproducible here")
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
    dims = min(3, raw.shape[1])
    native_3d = graphviz_dim == "3" and engine in GRAPHVIZ_NATIVE_3D_ENGINES and dims == 3

    raw = raw - raw.mean(axis=0)
    raw_range = raw.max(axis=0) - raw.min(axis=0)
    extent = float(raw_range[:dims].max())
    raw = raw / (extent if extent > 0 else 1.0)

    positions = np.zeros((num_nodes, 3), dtype=float)
    positions[:, :dims] = raw[:, :dims]
    positions *= scale

    if graphviz_dim in {"2Z", "3"} and not native_3d:
        if graphviz_dim == "3":
            print(f"  Graphviz {engine} has no native 3D in this build; deriving Z instead")
        z_method = getattr(props, "sfdp_z_method", "SPECTRAL") if props else "SPECTRAL"
        z_scale = getattr(props, "sfdp_z_scale", 0.3) if props else 0.3
        z = _generate_z_component(G, num_nodes, z_method)
        positions[:, 2] = z * z_scale * scale
        print(f"  Generated Graphviz Z via {z_method}: {positions[:, 2].min():.3f} to {positions[:, 2].max():.3f}")

    print(f"  scigraphs-utils Graphviz {engine} computed {num_nodes} nodes")
    return positions


def _graphviz_engine_layout(G, algorithm, iterations, scale, props=None):
    engine = GRAPHVIZ_ENGINES[algorithm]
    return _scigraphs_utils_graphviz_layout(G, engine, iterations, scale, props)

def _yifan_hu_layout(G, iterations, scale, props=None):
    """Yifan Hu scalable force-directed placement, via scigraphs-utils sfdp.
    ``props.sfdp_dim`` picks the mode: '2' is flat with Z = 0, '2Z' runs sfdp in
    2D and derives Z from graph structure, '3' asks Graphviz for native 3D."""
    import time
    start = time.time()

    num_nodes = len(G.nodes())
    print(f"Computing Yifan Hu layout for {num_nodes} nodes...")

    if num_nodes == 0:
        return np.zeros((0, 3))

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
