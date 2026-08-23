# Community detection for edge bundling. The add-on's Infomap needs GPLv3
# pySurprise, so this uses igraph greedy modularity, or numpy agglomeration.

import numpy as np

try:
    import igraph as _ig
    IGRAPH_AVAILABLE = True
except ImportError:      # pragma: no cover - optional dependency
    IGRAPH_AVAILABLE = False


def detect(edges, num_nodes, algorithm=None):
    """One community id per node, or None when the graph will not partition,
    which stops ``build_hierarchy`` recursing. An unknown ``algorithm`` gets the
    best available: raising made ``style(edges="hierarchical")`` crash."""
    pairs = _as_pairs(edges)
    n = int(num_nodes)
    if n < 2 or pairs.shape[0] == 0:
        return None

    if algorithm != "agglomerative" and IGRAPH_AVAILABLE:
        labels = _fastgreedy(pairs, n)
    else:
        labels = _agglomerate(pairs, n)
    if labels is None:
        return None
    uniq = np.unique(labels)
    if uniq.size < 2 or uniq.size >= n:
        return None
    return labels


def name():
    """Which implementation ``detect`` would use here."""
    return ("greedy modularity (igraph)" if IGRAPH_AVAILABLE
            else "agglomerative (numpy)")


def _as_pairs(edges):
    arr = np.asarray(edges, dtype=np.int64)
    if arr.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    return arr.reshape(-1, 2)


def _fastgreedy(pairs, n):
    """Clauset-Newman-Moore greedy modularity via igraph, deterministic. Not
    Louvain, which is randomized: two calls on the same 90-node graph gave
    different partitions, and the tree decides where every curve routes."""
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    if pairs.shape[0] == 0:
        return None
    lo = np.minimum(pairs[:, 0], pairs[:, 1])
    hi = np.maximum(pairs[:, 0], pairs[:, 1])
    key = lo.astype(np.int64) * int(n) + hi
    unique_key, counts = np.unique(key, return_counts=True)
    simple = np.stack([(unique_key // n).astype(np.int64),
                       (unique_key % n).astype(np.int64)], axis=1)
    graph = _ig.Graph(n=int(n), edges=simple.tolist(), directed=False)
    try:
        clustering = graph.community_fastgreedy(
            weights=counts.astype(np.float64).tolist()).as_clustering()
    except Exception:       # noqa: BLE001 - igraph refuses some degenerate graphs
        return None
    return np.asarray(clustering.membership, dtype=np.int64)


def _agglomerate(pairs, n, target_ratio=0.125, max_rounds=20):
    """Deterministic agglomerative clustering, used when igraph is missing. Not
    label propagation, which split three planted communities into 19 on a
    90-node fixture. Boruvka: each group joins the one it shares most edges with."""
    labels = np.arange(n, dtype=np.int64)
    if pairs.shape[0] == 0:
        return labels
    current = n
    floor = max(2, int(n * target_ratio))
    for _ in range(max_rounds):
        uniq, comp = np.unique(labels, return_inverse=True)
        m = uniq.size
        a, b = comp[pairs[:, 0]], comp[pairs[:, 1]]
        between = a != b
        if not between.any():
            break
        lo = np.minimum(a[between], b[between])
        hi = np.maximum(a[between], b[between])
        # How many edges join each pair of groups; multiplicity is the weight.
        key = lo.astype(np.int64) * m + hi
        unique_key, counts = np.unique(key, return_counts=True)
        pa = (unique_key // m).astype(np.int64)
        pb = (unique_key % m).astype(np.int64)

        # Each group's strongest partner from both ends; low index wins ties.
        node = np.concatenate([pa, pb])
        partner = np.concatenate([pb, pa])
        weight = np.concatenate([counts, counts])
        order = np.lexsort((partner, -weight, node))
        node, partner = node[order], partner[order]
        first = np.ones(node.size, dtype=bool)
        first[1:] = node[1:] != node[:-1]
        matched = np.stack([node[first], partner[first]], axis=1)

        merged = _components(m, matched)
        labels = merged[comp]
        shrunk = int(np.unique(labels).size)
        if shrunk >= current * 0.95:
            break
        current = shrunk
        if current <= floor:
            break
    return labels


def _components(num_nodes, edges):
    """Connected components by pointer jumping over short matched chains."""
    labels = np.arange(num_nodes, dtype=np.int64)
    a, b = edges[:, 0], edges[:, 1]
    while True:
        proposed = labels.copy()
        np.minimum.at(proposed, a, labels[b])
        np.minimum.at(proposed, b, labels[a])
        proposed = proposed[proposed]
        if np.array_equal(proposed, labels):
            return labels
        labels = proposed
