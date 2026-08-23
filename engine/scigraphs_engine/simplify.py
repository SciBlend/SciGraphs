# Backbone extraction, community coarsening, and the tree that hierarchical
# bundling routes along. Deterministic, so the viewport and the render agree.

import math
import numpy as np


# The detector lives in GPLv3 add-on code, so the host registers one here.
_DETECT = None


def set_community_detector(fn):
    """Register the fallback for ``build_hierarchy``:
    ``fn(edges_int, num_nodes, algorithm)`` returns a community id per node."""
    global _DETECT
    _DETECT = fn


def community_detector():
    return _DETECT


def _rank_within_node(edges, w):
    """Rank each half-edge among its node's, strongest first: (node, eid, rank)."""
    e = edges.shape[0]
    node = np.concatenate([edges[:, 0], edges[:, 1]]).astype(np.int64)
    eid = np.concatenate([np.arange(e), np.arange(e)])
    ww = np.concatenate([w, w])
    order = np.lexsort((-ww, node))
    n_sorted = node[order]
    new_grp = np.ones(n_sorted.size, dtype=bool)
    new_grp[1:] = n_sorted[1:] != n_sorted[:-1]
    grp_start = np.maximum.accumulate(
        np.where(new_grp, np.arange(n_sorted.size), 0))
    rank = np.arange(n_sorted.size) - grp_start
    return node[order], eid[order], rank


def _topk_mask(edges, w, k):
    """Keep an edge when it is in the top-k (by weight) of either endpoint."""
    _, eid, rank = _rank_within_node(edges, w)
    mask = np.zeros(edges.shape[0], dtype=bool)
    mask[eid[rank < k]] = True
    return mask


def _disparity_mask(edges, w, alpha):
    """Keep edges significant for at least one endpoint (Serrano-Boguna)."""
    e = edges.shape[0]
    n = int(edges.max()) + 1
    node = np.concatenate([edges[:, 0], edges[:, 1]]).astype(np.int64)
    eid = np.concatenate([np.arange(e), np.arange(e)])
    ww = np.concatenate([w, w])
    deg = np.bincount(node, minlength=n)
    strength = np.bincount(node, weights=ww, minlength=n)
    p = ww / np.maximum(strength[node], 1e-12)
    kdeg = deg[node]
    a = np.power(np.clip(1.0 - p, 0.0, 1.0), np.maximum(kdeg - 1, 1))
    keep_half = (a < alpha) | (kdeg <= 1)
    mask = np.zeros(e, dtype=bool)
    mask[eid[keep_half]] = True
    return mask


def _mst_mask(edges, w):
    """Mark maximum spanning tree membership. None when networkx is missing."""
    try:
        import networkx as nx
    except ImportError:
        return None
    graph = nx.Graph()
    graph.add_weighted_edges_from(
        (int(a), int(b), float(wt))
        for (a, b), wt in zip(edges.tolist(), w.tolist()))
    tree = {frozenset(uv[:2])
            for uv in nx.maximum_spanning_edges(graph, data=False)}
    return np.fromiter(
        (frozenset((int(a), int(b))) in tree for a, b in edges.tolist()),
        dtype=bool, count=edges.shape[0])


def backbone_mask(edges, weights, mode, k=3, alpha=0.05, sample=0.25):
    """Return ``(mask, stats)`` for the backbone edges. ``weights`` may be None."""
    if edges is None or edges.shape[0] == 0 or mode == 'ALL':
        return None, None
    e = edges.shape[0]
    w = np.asarray(weights, dtype=np.float64) if weights is not None \
        else np.ones(e, dtype=np.float64)
    w = np.maximum(w, 0.0)
    degraded = None

    if mode == 'TOPK':
        mask = _topk_mask(edges, w, max(1, int(k)))
    elif mode == 'DISPARITY':
        mask = _disparity_mask(edges, w, float(alpha))
    elif mode == 'MST':
        mask = _mst_mask(edges, w)
        if mask is None:
            mask = np.ones(e, dtype=bool)
            degraded = ("backbone 'mst' did not run: it needs the 'networkx' "
                        "extra, so every edge was kept")
        else:
            mask |= _topk_mask(edges, w, max(1, int(k)))
    elif mode == 'SAMPLE':
        rng = np.random.default_rng(0x5C16)  # fixed seed: deterministic/cacheable
        mask = rng.random(e) < float(sample)
        if not mask.any():
            mask[np.argmax(w)] = True
    else:
        return None, None

    total_w = float(w.sum())
    stats = {
        "kept": int(mask.sum()),
        "total": int(e),
        "weight_frac": (float(w[mask].sum()) / total_w) if total_w > 0 else 1.0,
    }
    if degraded:
        stats["degraded"] = degraded
    return mask, stats


def aggregate_per_community(values, inv, n_comm, counts, agg):
    if agg == 'SUM':
        return np.bincount(inv, weights=values, minlength=n_comm)
    if agg == 'MAX':
        out = np.full(n_comm, -np.inf)
        np.maximum.at(out, inv, values)
        return np.where(np.isfinite(out), out, 0.0)
    if agg == 'MIN':
        out = np.full(n_comm, np.inf)
        np.minimum.at(out, inv, values)
        return np.where(np.isfinite(out), out, 0.0)
    return np.bincount(inv, weights=values, minlength=n_comm) / counts


def community_radii(mode, counts, rms, driver, base_radius, max_mult):
    """World-space radius per supernode. COUNT and ATTRIBUTE (an aggregated
    POINT scalar) ramp by sqrt against the largest group; EXTENT uses RMS."""
    n = counts.size
    if mode == 'EXTENT':
        return np.maximum(rms * 0.75, base_radius)
    if mode == 'UNIFORM' or driver is None:
        return np.full(n, base_radius * max_mult, dtype=np.float64)
    d = np.maximum(np.asarray(driver, dtype=np.float64), 0.0)
    dmax = float(d.max()) if d.size else 0.0
    f = np.sqrt(d / dmax) if dmax > 0.0 else np.ones(n)
    return base_radius * (1.0 + (max_mult - 1.0) * f)


# Exponent p in base_radius * n**p; a third conserves volume.
STAND_IN_EXPONENT = 1.0 / 3.0


def stand_in_radii(counts, base_radius, exponent=None):
    """Radius of a supernode standing in for its members. Absolute, since a cut
    draws several levels at once, and monotone up the tree for ``adaptive``."""
    p = STAND_IN_EXPONENT if exponent is None else exponent
    return base_radius * np.power(np.maximum(
        np.asarray(counts, dtype=np.float64), 1.0), p)


def build_coarse_level(coords, edges, labels, colors, weights, base_radius,
                       size_mode='COUNT', size_values=None, size_agg='MEAN',
                       size_max_mult=6.0, stand_in=False):
    """Aggregate nodes by ``labels`` into supernodes and superedges. None under
    2 groups. se_src/se_dst point along the dominant direction of the pair."""
    uniq, inv = np.unique(labels, return_inverse=True)
    n_comm = uniq.size
    if n_comm < 2 or n_comm >= labels.size:
        return None
    counts = np.bincount(inv).astype(np.float64)
    centers = np.stack(
        [np.bincount(inv, weights=coords[:, i]) for i in range(3)],
        axis=1) / counts[:, None]

    ccolors = np.stack(
        [np.bincount(inv, weights=colors[:, i]) for i in range(4)],
        axis=1) / counts[:, None]

    d = np.linalg.norm(coords - centers[inv], axis=1)
    rms = np.sqrt(np.bincount(inv, weights=d * d) / counts)

    driver = None
    if size_mode == 'COUNT':
        driver = counts
    elif size_mode == 'ATTRIBUTE' and size_values is not None:
        driver = aggregate_per_community(
            np.asarray(size_values, dtype=np.float64), inv, n_comm, counts,
            size_agg)
    elif size_mode == 'EXTENT':
        driver = rms
    if stand_in:
        radii = stand_in_radii(counts, base_radius)
    else:
        radii = community_radii(
            size_mode, counts, rms, driver, base_radius, float(size_max_mult))

    out = {
        "centers": centers.astype(np.float32),
        "colors": ccolors.astype(np.float32),
        "radii": radii.astype(np.float32),
        "counts": counts.astype(np.int64),
        "rms": rms,
        "size_driver": driver,
        "n_comm": int(n_comm),
        "se_src": np.empty(0, dtype=np.int64),
        "se_dst": np.empty(0, dtype=np.int64),
        "se_count": np.empty(0, dtype=np.float64),
        "se_weight": np.empty(0, dtype=np.float64),
        "se_dom": np.empty(0, dtype=np.float64),
    }
    if edges is None or edges.shape[0] == 0:
        return out

    ca, cb = inv[edges[:, 0]], inv[edges[:, 1]]
    inter = ca != cb
    if not inter.any():
        return out
    ea, eb = ca[inter], cb[inter]
    w = np.asarray(weights, dtype=np.float64)[inter] if weights is not None \
        else np.ones(ea.size, dtype=np.float64)

    lo = np.minimum(ea, eb)
    hi = np.maximum(ea, eb)
    key = lo * np.int64(n_comm) + hi
    uk, kinv = np.unique(key, return_inverse=True)
    se_count = np.bincount(kinv).astype(np.float64)
    se_weight = np.bincount(kinv, weights=w)
    frac_fwd = np.bincount(kinv, weights=(ea == lo).astype(np.float64)) / se_count

    pa = (uk // n_comm).astype(np.int64)
    pb = (uk % n_comm).astype(np.int64)
    swap = frac_fwd < 0.5
    out["se_src"] = np.where(swap, pb, pa)
    out["se_dst"] = np.where(swap, pa, pb)
    out["se_count"] = se_count
    out["se_weight"] = se_weight
    out["se_dom"] = np.maximum(frac_fwd, 1.0 - frac_fwd)
    return out


# Infomap and the other pySurprise algorithms return a flat partition, so the
# recursion below is where the tree comes from.
HIERARCHY_MIN_NODES = 8

# Refuse to recurse when a round barely merges; else a tower of near-clones.
HIERARCHY_MIN_SHRINK = 0.9

# Cap on the expanded edge list; detection past this outweighs everything else.
HIERARCHY_MAX_EXPANDED = 200_000


def weighted_edge_list(edges, weights, max_edges=HIERARCHY_MAX_EXPANDED):
    """Repeat each edge, since multiplicity is the only weighting pySurprise
    honors. Scaled so the lightest edge appears once, then down to ``max_edges``."""
    edges = np.asarray(edges, dtype=np.int64)
    if edges.shape[0] == 0:
        return []
    if weights is None:
        return [(int(a), int(b)) for a, b in edges]

    w = np.maximum(np.asarray(weights, dtype=np.float64), 0.0)
    positive = w[w > 0.0]
    if positive.size == 0:
        return [(int(a), int(b)) for a, b in edges]

    reps = np.maximum(1, np.rint(w / positive.min())).astype(np.int64)
    total = int(reps.sum())
    if total > max_edges:
        reps = np.maximum(1, (reps * (max_edges / total)).astype(np.int64))
    expanded = np.repeat(edges, reps, axis=0)
    return [(int(a), int(b)) for a, b in expanded]


def build_hierarchy(coords, edges, labels, colors, weights, base_radius,
                    size_mode='COUNT', size_values=None, size_agg='MEAN',
                    size_max_mult=6.0, algorithm='infomap', max_levels=6,
                    detect=None):
    """Recursive community coarsening, finest level first. ``labels`` is the
    level-1 partition, normally the ``cluster_id`` already written, so the most
    expensive round is not recomputed. ``member_of`` maps original nodes here."""
    if detect is None:
        detect = _DETECT
    if detect is None:
        raise RuntimeError(
            "build_hierarchy needs a community detector: pass detect=..., or "
            "call scigraphs_engine.simplify.set_community_detector(fn) once at "
            "start-up")

    levels = []
    level = build_coarse_level(
        coords, edges, labels, colors, weights, base_radius,
        size_mode=size_mode, size_values=size_values, size_agg=size_agg,
        size_max_mult=size_max_mult, stand_in=True,
    )
    if level is None:
        return levels
    # build_coarse_level does not return np.unique's inverse.
    _, member_of = np.unique(labels, return_inverse=True)
    level["member_of"] = member_of.astype(np.int64)
    levels.append(level)

    while len(levels) < max_levels:
        prev = levels[-1]
        n_prev = prev["centers"].shape[0]
        if n_prev <= HIERARCHY_MIN_NODES or prev["se_src"].size == 0:
            break

        super_edges = np.stack([prev["se_src"], prev["se_dst"]], axis=1)
        found = detect(weighted_edge_list(super_edges, prev["se_weight"]),
                       n_prev, algorithm)
        if found is None:
            break
        next_labels = np.asarray(found, dtype=np.int64)
        n_next = np.unique(next_labels).size
        if n_next < 2 or n_next > HIERARCHY_MIN_SHRINK * n_prev:
            break

        level = build_coarse_level(
            prev["centers"].astype(np.float64), super_edges, next_labels,
            prev["colors"].astype(np.float64), prev["se_weight"], base_radius,
            size_mode=size_mode,
            size_values=prev["counts"].astype(np.float64),
            size_agg='SUM' if size_mode == 'ATTRIBUTE' else size_agg,
            size_max_mult=size_max_mult, stand_in=True,
        )
        if level is None:
            break
        _, inv = np.unique(next_labels, return_inverse=True)
        prev["parent"] = inv.astype(np.int64)
        level["member_of"] = inv[prev["member_of"]].astype(np.int64)

        # Above level 1, build_coarse_level counted supernodes, not nodes.
        # Recount against the previous level to track the original population.
        counts = np.bincount(inv, weights=prev["counts"].astype(np.float64),
                             minlength=level["centers"].shape[0])
        level["counts"] = counts.astype(np.int64)

        # Same correction for the spread, by the parallel axis theorem weighted
        # by population, so each child's own spread comes along.
        offset = prev["centers"].astype(np.float64) - level["centers"][inv]
        term = prev["counts"] * (prev["rms"] ** 2 + np.sum(offset ** 2, axis=1))
        rms = np.sqrt(np.bincount(inv, weights=term, minlength=counts.size)
                      / np.maximum(counts, 1.0))
        level["rms"] = rms
        level["size_driver"] = counts
        level["radii"] = stand_in_radii(counts, base_radius).astype(np.float32)
        levels.append(level)

    levels[-1]["parent"] = None
    return levels
