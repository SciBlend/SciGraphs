# Deterministic graphs for the API tests. Seeds are fixed: thinning and sampled
# betweenness promise the same answer everywhere. clustered() weights edges
# heavy-tailed so the disparity and top-k backbones cannot agree by accident.

import numpy as np


def clustered(seed=11, per_community=50, communities=3, p_in=0.10, bridges=12):
    """(coords, edges, node_attrs, edge_attrs), the workhorse fixture."""
    rng = np.random.default_rng(seed)
    centers = [(-3.0, 0.0, 0.0), (3.0, 0.0, 0.0), (0.0, 3.0, 1.0),
               (0.0, -3.0, -1.0)][:communities]
    coords = np.vstack([rng.normal(c, 0.6, (per_community, 3))
                        for c in centers]).astype(np.float32)
    n = coords.shape[0]
    pairs = []
    for b in range(communities):
        lo, hi = b * per_community, (b + 1) * per_community
        for i in range(lo, hi):
            for j in range(i + 1, hi):
                if rng.random() < p_in:
                    pairs.append((i, j))
    for _ in range(bridges):
        a, b = int(rng.integers(0, n)), int(rng.integers(0, n))
        if a != b:
            pairs.append((min(a, b), max(a, b)))
    edges = np.array(sorted(set(pairs)), dtype=np.int32)
    community = np.repeat(np.arange(communities, dtype=np.float32),
                          per_community)
    weight = rng.lognormal(0.0, 1.2, edges.shape[0]).astype(np.float32)
    return (coords, edges,
            {"community": community},
            {"weight": weight})


def plain(seed=3, n=120, degree=3):
    """No attributes, and the last six nodes isolated so the component
    channel has something to say."""
    rng = np.random.default_rng(seed)
    coords = rng.normal(0.0, 2.0, (n, 3)).astype(np.float32)
    found = set()
    for i in range(n - 6):
        for _ in range(degree):
            j = int(rng.integers(0, n - 6))
            if i != j:
                found.add((min(i, j), max(i, j)))
    return coords, np.array(sorted(found), dtype=np.int32)


def baked(seed=17, n=40, degree=3):
    """The shape a baked Blender mesh takes, and the only fixture exercising
    recover_logical_edges. The logical edges it returns last are the answer."""
    rng = np.random.default_rng(seed)
    node_coords = rng.normal(0.0, 2.0, (n, 3)).astype(np.float32)
    found = set()
    for i in range(n):
        for _ in range(degree):
            j = int(rng.integers(0, n))
            if i != j:
                found.add((min(i, j), max(i, j)))
    logical = np.array(sorted(found), dtype=np.int32)

    mids = 0.5 * (node_coords[logical[:, 0]] + node_coords[logical[:, 1]])
    # Off the chord, since a real baked style produces a curve.
    mids = mids + np.array([0.0, 0.0, 0.35], dtype=np.float32)
    coords = np.vstack([node_coords, mids]).astype(np.float32)
    mid_ids = np.arange(n, n + logical.shape[0], dtype=np.int32)
    mesh_edges = np.empty((logical.shape[0] * 2, 2), dtype=np.int32)
    mesh_edges[0::2, 0] = logical[:, 0]
    mesh_edges[0::2, 1] = mid_ids
    mesh_edges[1::2, 0] = mid_ids
    mesh_edges[1::2, 1] = logical[:, 1]
    node_mask = np.zeros(coords.shape[0], dtype=bool)
    node_mask[:n] = True
    return coords, mesh_edges, node_mask, logical
