"""Network centrality computations with optional rustworkx fast path.

Exposes:
    - node_betweenness(G, weight=..., fast=True)
    - edge_betweenness_line(G, weight=..., fast=True)   # via line graph
    - node_closeness(G, weight=..., fast=True)
    - get_colors_by_attr(values, cmap_name="viridis")   # returns (N, 4) RGBA
    - orientation_rose_mesh_data(bearings, bins, radius, height_scale)
"""

from ...utils.logger import log


# ---------------------------------------------------------------------------
# Rustworkx interop helpers
# ---------------------------------------------------------------------------

def _try_rustworkx_digraph(G, weight_attr):
    """Convert a (Multi)DiGraph to a rustworkx PyDiGraph, or return None."""
    try:
        import rustworkx as rx
    except Exception:
        return None, None, None

    try:
        import networkx as nx

        # Collapse MultiDiGraph to a simple DiGraph by keeping min-weight edges.
        if G.is_multigraph():
            simple = nx.DiGraph()
            for u, v, data in G.edges(data=True):
                w = data.get(weight_attr, None)
                if simple.has_edge(u, v):
                    prev = simple[u][v][weight_attr]
                    if w is not None and (prev is None or w < prev):
                        simple[u][v][weight_attr] = w
                else:
                    simple.add_edge(u, v, **{weight_attr: w})
            H = simple
        else:
            H = G

        rx_graph = rx.PyDiGraph(check_cycle=False, multigraph=False)
        node_map = {}
        for n in H.nodes:
            node_map[n] = rx_graph.add_node(n)
        reverse = {i: n for n, i in node_map.items()}

        for u, v, data in H.edges(data=True):
            w = data.get(weight_attr)
            if w is None:
                w = 1.0
            rx_graph.add_edge(node_map[u], node_map[v], float(w))

        return rx_graph, node_map, reverse
    except Exception as e:
        log(f"rustworkx conversion failed: {e}")
        return None, None, None


# ---------------------------------------------------------------------------
# Node betweenness centrality
# ---------------------------------------------------------------------------

def node_betweenness(G, weight="length", fast=True):
    """Compute node betweenness centrality. Returns {node_id: value}.

    Uses rustworkx when ``fast=True`` and the module is available.
    """
    if G is None:
        return {}

    if fast:
        rx_graph, node_map, reverse = _try_rustworkx_digraph(G, weight)
        if rx_graph is not None:
            try:
                import rustworkx as rx
                vals = rx.betweenness_centrality(rx_graph, normalized=True)
                return {reverse[i]: float(vals[i]) for i in range(len(vals))}
            except Exception as e:
                log(f"rustworkx betweenness failed, falling back: {e}")

    try:
        import networkx as nx
        return nx.betweenness_centrality(G, weight=weight, normalized=True)
    except Exception as e:
        log(f"NetworkX betweenness failed: {e}")
        return {}


# ---------------------------------------------------------------------------
# Edge betweenness via line graph
# ---------------------------------------------------------------------------

def edge_betweenness_line(G, weight="length", fast=True):
    """Compute edge betweenness via the line graph.

    Returns ``{(u, v): value}`` on the original graph's edges. Uses rustworkx
    on the line graph for speed when available.
    """
    if G is None:
        return {}

    try:
        import networkx as nx
    except Exception:
        return {}

    # Build line graph with simple DiGraph semantics.
    simple = nx.DiGraph()
    for u, v, data in G.edges(data=True):
        w = data.get(weight, 1.0)
        if simple.has_edge(u, v):
            prev = simple[u][v].get(weight, float("inf"))
            if w is not None and w < prev:
                simple[u][v][weight] = w
        else:
            simple.add_edge(u, v, **{weight: w})

    try:
        L = nx.line_graph(simple)
    except Exception as e:
        log(f"Line graph construction failed: {e}")
        return {}

    # Edge weights in L come from incidence; use uniform weights unless we
    # already aggregate.
    if fast:
        try:
            import rustworkx as rx
            rx_L = rx.PyDiGraph(check_cycle=False, multigraph=False)
            node_map = {n: rx_L.add_node(n) for n in L.nodes}
            reverse = {i: n for n, i in node_map.items()}
            for u, v in L.edges():
                rx_L.add_edge(node_map[u], node_map[v], 1.0)
            vals = rx.betweenness_centrality(rx_L, normalized=True)
            return {reverse[i]: float(vals[i]) for i in range(len(vals))}
        except Exception as e:
            log(f"rustworkx edge-line betweenness failed: {e}")

    try:
        bc = nx.betweenness_centrality(L, normalized=True)
        return {k: float(v) for k, v in bc.items()}
    except Exception as e:
        log(f"NetworkX edge-line betweenness failed: {e}")
        return {}


# ---------------------------------------------------------------------------
# Closeness centrality
# ---------------------------------------------------------------------------

def node_closeness(G, weight="length", fast=True):
    """Compute node closeness centrality. Returns {node_id: value}."""
    if G is None:
        return {}

    if fast:
        rx_graph, node_map, reverse = _try_rustworkx_digraph(G, weight)
        if rx_graph is not None:
            try:
                import rustworkx as rx
                vals = rx.closeness_centrality(rx_graph)
                return {reverse[i]: float(vals[i]) for i in range(len(vals))}
            except Exception as e:
                log(f"rustworkx closeness failed, falling back: {e}")

    try:
        import networkx as nx
        return nx.closeness_centrality(G, distance=weight)
    except Exception as e:
        log(f"NetworkX closeness failed: {e}")
        return {}


# ---------------------------------------------------------------------------
# Colormap utilities
# ---------------------------------------------------------------------------

# Control points (r,g,b) @ t=0.0, 0.25, 0.5, 0.75, 1.0 for the main colormaps.
# Used as a matplotlib-free fallback so the addon works with just numpy.
_CMAP_FALLBACK_POINTS = {
    "viridis": [
        (0.267, 0.005, 0.329),
        (0.229, 0.322, 0.546),
        (0.127, 0.567, 0.550),
        (0.369, 0.789, 0.382),
        (0.993, 0.906, 0.144),
    ],
    "plasma": [
        (0.050, 0.030, 0.528),
        (0.427, 0.000, 0.617),
        (0.798, 0.214, 0.495),
        (0.991, 0.551, 0.236),
        (0.940, 0.975, 0.131),
    ],
    "inferno": [
        (0.001, 0.000, 0.014),
        (0.258, 0.039, 0.406),
        (0.578, 0.148, 0.404),
        (0.865, 0.317, 0.228),
        (0.988, 1.000, 0.645),
    ],
    "magma": [
        (0.001, 0.000, 0.014),
        (0.232, 0.060, 0.439),
        (0.551, 0.162, 0.507),
        (0.867, 0.335, 0.413),
        (0.987, 0.991, 0.749),
    ],
    "turbo": [
        (0.189, 0.072, 0.232),
        (0.192, 0.513, 0.984),
        (0.216, 0.941, 0.482),
        (0.940, 0.716, 0.200),
        (0.479, 0.016, 0.011),
    ],
    "coolwarm": [
        (0.230, 0.299, 0.754),
        (0.545, 0.622, 0.919),
        (0.866, 0.866, 0.866),
        (0.957, 0.620, 0.479),
        (0.706, 0.016, 0.150),
    ],
    "RdYlBu_r": [
        (0.192, 0.211, 0.584),
        (0.510, 0.789, 0.753),
        (1.000, 1.000, 0.749),
        (0.992, 0.600, 0.357),
        (0.647, 0.000, 0.149),
    ],
    "YlGnBu": [
        (1.000, 1.000, 0.851),
        (0.780, 0.914, 0.706),
        (0.255, 0.714, 0.769),
        (0.141, 0.408, 0.674),
        (0.031, 0.114, 0.345),
    ],
    "hot": [
        (0.042, 0.000, 0.000),
        (0.500, 0.000, 0.000),
        (1.000, 0.500, 0.000),
        (1.000, 1.000, 0.333),
        (1.000, 1.000, 1.000),
    ],
}


def _fallback_colormap(name, arr_norm):
    """Numpy-only replacement for matplotlib colormaps (see table above)."""
    import numpy as np

    pts = _CMAP_FALLBACK_POINTS.get(name, _CMAP_FALLBACK_POINTS["viridis"])
    stops = np.linspace(0.0, 1.0, len(pts))
    arr = np.clip(arr_norm, 0.0, 1.0)

    r = np.interp(arr, stops, [p[0] for p in pts])
    g = np.interp(arr, stops, [p[1] for p in pts])
    b = np.interp(arr, stops, [p[2] for p in pts])
    a = np.ones_like(arr)
    return np.stack([r, g, b, a], axis=-1)


def get_colors_by_values(values, cmap_name="viridis", vmin=None, vmax=None):
    """Map an iterable of floats to RGBA tuples using a matplotlib colormap
    when available, or a numpy-only fallback otherwise.
    """
    try:
        import numpy as np
    except ImportError as e:
        log(f"numpy missing: {e}")
        return None

    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return np.zeros((0, 4), dtype=float)

    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros((arr.size, 4), dtype=float)

    vmin = float(np.nanmin(arr[finite])) if vmin is None else float(vmin)
    vmax = float(np.nanmax(arr[finite])) if vmax is None else float(vmax)
    if vmax == vmin:
        vmax = vmin + 1e-9

    norm = (arr - vmin) / (vmax - vmin)
    norm = np.clip(norm, 0.0, 1.0)

    try:
        from matplotlib import cm
        try:
            cmap = cm.get_cmap(cmap_name)
        except Exception:
            cmap = cm.get_cmap("viridis")
        rgba = cmap(norm)
    except Exception:
        rgba = _fallback_colormap(cmap_name, norm)

    rgba[~finite] = [0.3, 0.3, 0.3, 1.0]
    return rgba


# ---------------------------------------------------------------------------
# Orientation rose mesh
# ---------------------------------------------------------------------------

def orientation_rose_mesh_data(bearings, bins=36, radius=2.0, height_scale=1.0):
    """Build vertex/face lists for a 3D polar histogram of edge bearings.

    Returns a dict ``{"verts": [(x,y,z), ...], "faces": [(i,j,k,l), ...]}``
    ready for a Blender ``mesh.from_pydata`` call. Each angular bin becomes
    a radial wedge whose height is proportional to the count.

    Args:
        bearings: Iterable of bearings in degrees (0..360).
        bins: Number of angular bins.
        radius: Outer radius of the rose.
        height_scale: Vertical scale for bar heights.
    """
    import math

    if not bearings:
        return {"verts": [], "faces": []}

    counts = [0] * bins
    bin_size = 360.0 / bins

    for b in bearings:
        if b is None:
            continue
        try:
            angle = float(b) % 360.0
        except Exception:
            continue
        idx = int(angle // bin_size) % bins
        counts[idx] += 1

    max_count = max(counts) or 1
    heights = [c / max_count * height_scale for c in counts]

    verts = []
    faces = []
    for i in range(bins):
        a0 = math.radians(i * bin_size)
        a1 = math.radians((i + 1) * bin_size)
        x0, y0 = math.cos(a0) * radius, math.sin(a0) * radius
        x1, y1 = math.cos(a1) * radius, math.sin(a1) * radius
        z = heights[i]
        base_i = len(verts)
        verts.extend([
            (0.0, 0.0, 0.0),     # base center 0
            (x0, y0, 0.0),       # base outer 1
            (x1, y1, 0.0),       # base outer 2
            (0.0, 0.0, z),       # top center 3
            (x0, y0, z),         # top outer 4
            (x1, y1, z),         # top outer 5
        ])
        faces.extend([
            (base_i + 0, base_i + 1, base_i + 2),           # bottom
            (base_i + 3, base_i + 5, base_i + 4),           # top
            (base_i + 0, base_i + 2, base_i + 5, base_i + 3),  # side1
            (base_i + 0, base_i + 3, base_i + 4, base_i + 1),  # side2
            (base_i + 1, base_i + 4, base_i + 5, base_i + 2),  # outer
        ])

    return {"verts": verts, "faces": faces, "counts": counts}
