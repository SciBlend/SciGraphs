# Every named scalar the filter stack can ask for, computed from the graph, not
# read off it. Channels return native quantities and the caller normalizes,
# since the panel prints "coreness >= 6 of 25"; a missing dep gives None.

import numpy as np

from .channels import (channel_normalized, size_normalized,
                       value_channel)

try:
    import igraph as ig
    IGRAPH_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    IGRAPH_AVAILABLE = False

try:
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components as _scipy_components
    from scipy.sparse.csgraph import dijkstra as _scipy_dijkstra
    from scipy.spatial import cKDTree
    SCIPY_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    SCIPY_AVAILABLE = False


# In menu order.
CHANNEL_GROUPS = [
    ("Attributes", [
        ('ATTR', "Node Attribute", 'POINT',
         "A scalar POINT attribute, normalized over its own range"),
    ]),
    ("Connectivity", [
        ('DEGREE', "Degree", 'POINT',
         "How many edges meet the node. Derived, so it needs no imported "
         "attribute"),
        ('STRENGTH', "Weighted Degree", 'POINT',
         "The sum of the weights of a node's edges, over an EDGE attribute. "
         "Degree counts partners and this counts traffic, which on a weighted "
         "network are different nodes: an airport with four heavy routes "
         "outranks one with twenty empty ones"),
        ('CORE', "Coreness (k-core)", 'POINT',
         "The largest k for which the node survives peeling every vertex of "
         "degree below k. Keeps nodes that are mutually well connected, which "
         "is not what a weight threshold keeps: a hub joined to a hundred "
         "leaves has high degree and coreness 1"),
        ('COMPONENT', "Component Size", 'POINT',
         "Size of the connected component the node belongs to, over the "
         "largest one. The high end is the giant component; the low end is the "
         "confetti of isolated pairs that no attribute filter can name"),
        ('CLUSTERING', "Clustering Coefficient", 'POINT',
         "How many of a node's neighbors are joined to each other, over how "
         "many could be. Separates the two shapes degree cannot tell apart: "
         "the center of a clique and the hub of a star both have many edges, "
         "and only one of them sits inside a dense neighborhood"),
    ]),
    ("Centrality", [
        ('PAGERANK', "PageRank", 'POINT',
         "Importance that counts who is pointing rather than how many: a node "
         "cited by three central nodes outranks one cited by thirty "
         "peripheral ones. Degree cannot express that, because it weighs every "
         "neighbor the same"),
        ('BETWEEN', "Betweenness (sampled)", 'POINT',
         "How much of the network's shortest-path traffic crosses this node. "
         "The one thing no other channel finds: a bridge between two "
         "communities has degree 2 and coreness 1 and is the single node whose "
         "removal splits the picture. Estimated from a sample of sources, "
         "because the exact answer is 34 seconds on a 21k-node graph and the "
         "ranking it produces is the same"),
        ('ARTICULATION', "Articulation Point", 'POINT',
         "1 where removing the node would break its component into more "
         "pieces. The exact, unsampled version of 'this node is load-bearing'"),
    ]),
    ("Modules", [
        ('PARTICIPATION', "Participation Coefficient", 'POINT',
         "How evenly a node spreads its edges across the groups, by the "
         "coarsening attribute. High is a connector that holds modules "
         "together, 0 is a node that only talks to its own. The classic "
         "companion to the next channel"),
        ('MODULE_Z', "Within-Module Degree", 'POINT',
         "How well connected a node is inside its own group, in standard "
         "deviations from that group's mean. A provincial hub scores high here "
         "and low on participation; a connector is the other way round"),
    ]),
    ("Layout", [
        ('POS_X', "Position X", 'POINT',
         "The node's X coordinate. One clause is a slab, so this is the "
         "cutaway: a cross-section of a road network or a brain opened along "
         "an axis, which no structural channel can express"),
        ('POS_Y', "Position Y", 'POINT', "The node's Y coordinate"),
        ('POS_Z', "Position Z", 'POINT', "The node's Z coordinate"),
        ('RADIUS', "Distance From Center", 'POINT',
         "How far the node is from the center of the graph, so a clause is a "
         "shell or a ball"),
        ('CROWDING', "Local Crowding", 'POINT',
         "Distance to the sixth nearest node in the layout, inverted, so high "
         "means packed. Dissolves the hairball or keeps only it -- and it is "
         "not degree: a node of degree three can sit in the densest part of "
         "the picture"),
    ]),
    ("Selection", [
        ('HOPS', "Hops From Seed", 'POINT',
         "Edge distance to the nearest seeded node, over the farthest. This "
         "is the ego network: seed a node and keep three hops of it. Seeds come "
         "from a BOOLEAN attribute written by the button below the sliders, so "
         "they survive saving the file"),
        ('THIN', "Thinning", 'POINT',
         "A stable pseudorandom rank in [0, 1), so a range keeps that fraction "
         "of the graph and keeps the same one every frame. Computed within each "
         "group when there is a coarsening attribute, so thinning to a tenth "
         "keeps a tenth of every community instead of erasing the small ones"),
    ]),
    ("Edges", [
        ('EDGE_ATTR', "Edge Attribute", 'EDGE',
         "A scalar EDGE attribute, normalized over its own range"),
        ('LENGTH', "Edge Length", 'EDGE',
         "Distance between the endpoints in the current layout. On a spatial "
         "layout this separates the local mesh from the long-range shortcuts"),
        ('SPAN', "Edge Spans Groups", 'EDGE',
         "0 for an edge inside a group, 1 for one between groups, by the "
         "coarsening attribute. Keeps the modular structure or only what "
         "crosses it"),
        ('DISPARITY', "Backbone Significance", 'EDGE',
         "How unlikely an edge's share of its endpoint's total weight would be "
         "if that weight were spread at random (the disparity filter). This is "
         "what a weight threshold gets wrong: a threshold deletes every edge of "
         "the light nodes and keeps every edge of the heavy ones, and this asks "
         "each node which of its own edges matter. Needs an EDGE weight"),
        ('SUPPORT', "Triangle Support", 'EDGE',
         "How many triangles the edge belongs to -- the edge-side answer to "
         "coreness, and what a k-truss peels on. Keeps the cohesive skeleton "
         "and drops the edges that hold nothing together"),
        ('OVERLAP', "Neighborhood Overlap", 'EDGE',
         "How much the two endpoints' neighborhoods coincide (Jaccard). The "
         "continuous form of 'inside a community or between two', and unlike "
         "Edge Spans Groups it needs no community attribute: 0 is a bridge, "
         "high is an edge buried inside a dense group"),
        ('EDGE_MINDEG', "Rich Club", 'EDGE',
         "The smaller of the two endpoints' degrees, so the high end is the "
         "hub-to-hub edges and nothing else: the skeleton the important nodes "
         "form among themselves"),
        ('BRIDGE', "Bridge", 'EDGE',
         "1 where removing the edge would break its component in two. Every "
         "other edge is redundant, and on a road network that distinction is "
         "the one that matters"),
    ]),
]

_ALL = [entry for _group, entries in CHANNEL_GROUPS for entry in entries]

NODE_CHANNELS = tuple(e[0] for e in _ALL if e[2] == 'POINT')
EDGE_CHANNELS = tuple(e[0] for e in _ALL if e[2] == 'EDGE')
CHANNEL_LABELS = {e[0]: e[1] for e in _ALL}

# Channels needing more than the graph; the panel offers the matching field.
NEEDS_WEIGHT = ('STRENGTH', 'DISPARITY')
NEEDS_GROUPS = ('SPAN', 'PARTICIPATION', 'MODULE_Z')
NEEDS_SEEDS = ('HOPS',)

# 0-or-1 channels; the panel words them instead of showing "0.25 to 0.75 of 1".
FLAG_CHANNELS = ('ARTICULATION', 'BRIDGE', 'SPAN')

# Max betweenness sources: error falls as 1/sqrt(k) and cost rises as k.
BETWEEN_SAMPLES = 256

# Traversal budget, nodes plus edges: 256 sources is 40 s on a 2M-node road.
BETWEEN_BUDGET = 80_000_000

CROWDING_K = 6


def core_numbers(num_nodes, edges):
    """Coreness per node, igraph if installed: the numpy peeler only cleared 93
    waves in 2.4 s on the 2M-node graph. Self-loops dropped, so a leaf is 1."""
    if edges is None or edges.size == 0:
        return np.zeros(num_nodes, dtype=np.int32)
    real = edges[edges[:, 0] != edges[:, 1]]
    if IGRAPH_AVAILABLE:
        graph = ig.Graph(n=int(num_nodes), edges=real.tolist(), directed=False)
        return np.asarray(graph.coreness(), dtype=np.int32)
    return _peel_core(num_nodes, real)


def _peel_core(num_nodes, edges):
    deg = np.bincount(edges.ravel(), minlength=num_nodes).astype(np.int64)
    core = np.zeros(num_nodes, dtype=np.int32)
    alive = np.ones(num_nodes, dtype=bool)
    a, b = edges[:, 0], edges[:, 1]
    k = 0
    while alive.any():
        peel = alive & (deg <= k)
        if not peel.any():
            k += 1
            continue
        while peel.any():
            core[peel] = k
            alive &= ~peel
            # Only a peeled-to-survivor edge lowers a degree.
            hit_b = peel[a] & alive[b]
            hit_a = peel[b] & alive[a]
            np.subtract.at(deg, b[hit_b], 1)
            np.subtract.at(deg, a[hit_a], 1)
            peel = alive & (deg <= k)
    return core


def component_sizes(num_nodes, edges):
    """Component size per node. scipy takes 120 ms at 2M nodes; without it,
    label propagation. Isolated nodes are size one, so a filter can drop them."""
    if edges is None or edges.size == 0:
        return np.ones(num_nodes, dtype=np.int64)
    if SCIPY_AVAILABLE:
        weights = np.ones(edges.shape[0], dtype=np.int8)
        adjacency = coo_matrix(
            (weights, (edges[:, 0], edges[:, 1])),
            shape=(int(num_nodes), int(num_nodes)))
        _, labels = _scipy_components(adjacency, directed=False)
    else:
        labels = _propagate_labels(num_nodes, edges)
    return np.bincount(labels)[labels].astype(np.int64)


def _propagate_labels(num_nodes, edges):
    """Min-label propagation without scipy; the rounds equal the diameter."""
    labels = np.arange(num_nodes, dtype=np.int64)
    a, b = edges[:, 0], edges[:, 1]
    while True:
        proposed = labels.copy()
        np.minimum.at(proposed, a, labels[b])
        np.minimum.at(proposed, b, labels[a])
        # Jump to the root so long chains collapse geometrically.
        proposed = proposed[proposed]
        if np.array_equal(proposed, labels):
            return labels
        labels = proposed


def _igraph_of(num_nodes, edges):
    """Edges as given, self-loops included; ``bridges()`` indexes into them."""
    if not IGRAPH_AVAILABLE or edges is None or edges.size == 0:
        return None
    return ig.Graph(n=int(num_nodes), edges=edges.tolist(), directed=False)


def _adjacency(num_nodes, edges):
    if not SCIPY_AVAILABLE or edges is None or edges.size == 0:
        return None
    real = edges[edges[:, 0] != edges[:, 1]]
    if real.size == 0:
        return None
    rows = np.concatenate([real[:, 0], real[:, 1]])
    cols = np.concatenate([real[:, 1], real[:, 0]])
    data = np.ones(rows.size, dtype=np.float32)
    adjacency = coo_matrix((data, (rows, cols)),
                           shape=(int(num_nodes), int(num_nodes))).tocsr()
    # A parallel edge would otherwise count twice in every triangle sum.
    adjacency.data[:] = 1.0
    adjacency.sum_duplicates()
    adjacency.data[:] = 1.0
    return adjacency


def clustering_coefficients(num_nodes, edges):
    """Local clustering per node, or None. Degree under two gives zero, which is
    igraph's ``mode="zero"``; NaN compares false and would erase those nodes."""
    graph = _igraph_of(num_nodes, edges)
    if graph is not None:
        return np.asarray(graph.transitivity_local_undirected(mode="zero"),
                          dtype=np.float32)
    adjacency = _adjacency(num_nodes, edges)
    if adjacency is None:
        return None
    degree = np.asarray(adjacency.sum(axis=1), dtype=np.float64).ravel()
    # Twice the triangles per node: the diagonal of A^3, without forming A^3.
    twice = np.asarray(adjacency.multiply(adjacency @ adjacency).sum(axis=1),
                       dtype=np.float64).ravel()
    pairs = degree * (degree - 1.0)
    out = np.zeros(int(num_nodes), dtype=np.float32)
    np.divide(twice, pairs, out=out, where=pairs > 0,
              casting='unsafe')
    return out


def pagerank_scores(num_nodes, edges, weights=None):
    """PageRank per node, by igraph or power iteration, or None. The fallback
    teleports dangling mass instead of leaking it, so the two agree."""
    graph = _igraph_of(num_nodes, edges)
    if graph is not None:
        return np.asarray(
            graph.pagerank(damping=0.85, directed=False,
                           weights=None if weights is None
                           else weights.astype(np.float64).tolist()),
            dtype=np.float32)
    adjacency = _adjacency(num_nodes, edges)
    if adjacency is None:
        return None
    n = int(num_nodes)
    degree = np.asarray(adjacency.sum(axis=1), dtype=np.float64).ravel()
    inverse = np.divide(1.0, degree, out=np.zeros(n), where=degree > 0)
    transition = adjacency.T.multiply(inverse).tocsr()
    rank = np.full(n, 1.0 / n)
    dangling = degree == 0
    for _ in range(100):
        moved = 0.85 * (transition @ rank + rank[dangling].sum() / n) \
            + 0.15 / n
        if np.abs(moved - rank).sum() < 1e-10:
            rank = moved
            break
        rank = moved
    return rank.astype(np.float32)


def betweenness_sample_size(num_nodes, num_edges):
    """How many sources to traverse from. Below eight there is no ranking."""
    work = max(1, int(num_nodes) + int(num_edges))
    return int(min(BETWEEN_SAMPLES, max(8, BETWEEN_BUDGET // work)))


def betweenness_sampled(num_nodes, edges, samples=None):
    """Sampled betweenness scaled to the full estimate, or None. What is bounded
    is the source count, not the path length, at a fixed stride so it repeats."""
    graph = _igraph_of(num_nodes, edges)
    if graph is None:
        return None
    n = int(num_nodes)
    if samples is None:
        samples = betweenness_sample_size(n, edges.shape[0])
    if n <= samples:
        return np.asarray(graph.betweenness(), dtype=np.float32)
    sources = np.arange(0, n, max(1, n // samples), dtype=np.int64)[:samples]
    scores = np.asarray(graph.betweenness(sources=sources.tolist()),
                        dtype=np.float64)
    return (scores * (n / float(sources.size))).astype(np.float32)


def articulation_flags(num_nodes, edges):
    graph = _igraph_of(num_nodes, edges)
    if graph is None:
        return None
    out = np.zeros(int(num_nodes), dtype=np.float32)
    points = graph.articulation_points()
    if points:
        out[np.asarray(points, dtype=np.int64)] = 1.0
    return out


def bridge_flags(num_nodes, edges):
    graph = _igraph_of(num_nodes, edges)
    if graph is None:
        return None
    out = np.zeros(edges.shape[0], dtype=np.float32)
    found = graph.bridges()
    if found:
        out[np.asarray(found, dtype=np.int64)] = 1.0
    return out


def _endpoint_intersections(num_nodes, edges, chunk=400000):
    """``|N(u) ∩ N(v)|`` per edge, None without scipy; chunked at 2M edges."""
    adjacency = _adjacency(num_nodes, edges)
    if adjacency is None:
        return None
    out = np.zeros(edges.shape[0], dtype=np.float32)
    for start in range(0, edges.shape[0], chunk):
        stop = min(start + chunk, edges.shape[0])
        left = adjacency[edges[start:stop, 0]]
        right = adjacency[edges[start:stop, 1]]
        out[start:stop] = np.asarray(
            left.multiply(right).sum(axis=1), dtype=np.float32).ravel()
    return out


def triangle_support(num_nodes, edges):
    if edges is None or edges.size == 0:
        return None
    return _endpoint_intersections(num_nodes, edges)


def neighborhood_overlap(num_nodes, edges):
    if edges is None or edges.size == 0:
        return None
    shared = _endpoint_intersections(num_nodes, edges)
    if shared is None:
        return None
    real = edges[:, 0] != edges[:, 1]
    degree = np.bincount(edges[real].ravel(),
                         minlength=int(num_nodes)).astype(np.float64)
    # Each endpoint is in the other's neighbor list, so drop one from each.
    left = degree[edges[:, 0]] - 1.0
    right = degree[edges[:, 1]] - 1.0
    union = left + right - shared
    out = np.zeros(edges.shape[0], dtype=np.float32)
    np.divide(shared, union, out=out, where=union > 0, casting='unsafe')
    return out


def disparity_significance(num_nodes, edges, weights):
    """1 - alpha of the disparity filter (Serrano, Boguna, Vespignani) per
    edge, or None. Under a uniform null a degree-k node's edge holds a
    fraction p or more with chance ``(1 - p)^(k - 1)``; the smaller wins."""
    if edges is None or edges.size == 0 or weights is None:
        return None
    if weights.size != edges.shape[0]:
        return None
    w = np.where(np.isfinite(weights), weights, 0.0).astype(np.float64)
    w = np.maximum(w, 0.0)
    n = int(num_nodes)
    strength = np.zeros(n)
    np.add.at(strength, edges[:, 0], w)
    np.add.at(strength, edges[:, 1], w)
    degree = np.bincount(edges.ravel(), minlength=n).astype(np.float64)

    alpha = np.ones(edges.shape[0])
    for side in (0, 1):
        node = edges[:, side]
        total = strength[node]
        k = degree[node]
        share = np.divide(w, total, out=np.zeros_like(w), where=total > 0)
        # A single-edge node cannot be rejected: p-value stays (1-p)^0 = 1.
        exponent = np.maximum(k - 1.0, 0.0)
        alpha = np.minimum(
            alpha, np.power(np.clip(1.0 - share, 0.0, 1.0), exponent))
    return (1.0 - alpha).astype(np.float32)


def rich_club_channel(num_nodes, edges):
    if edges is None or edges.size == 0:
        return None
    degree = np.bincount(edges.ravel(), minlength=int(num_nodes))
    return np.minimum(degree[edges[:, 0]],
                      degree[edges[:, 1]]).astype(np.float32)


def strength_channel(num_nodes, edges, weights):
    if edges is None or edges.size == 0 or weights is None:
        return None
    if weights.size != edges.shape[0]:
        return None
    w = np.where(np.isfinite(weights), weights, 0.0).astype(np.float64)
    out = np.zeros(int(num_nodes))
    np.add.at(out, edges[:, 0], w)
    np.add.at(out, edges[:, 1], w)
    return out.astype(np.float32)


def module_roles_from_groups(values, edges, num_nodes):
    """``(participation, within_module_z)``, meaningful only together. Grouped
    by ``(node, other endpoint's group)``; the dense matrix is 10^8 cells."""
    if values is None or edges is None or edges.size == 0:
        return None, None
    groups = np.unique(values, return_inverse=True)[1].astype(np.int64)
    num_groups = int(groups.max()) + 1

    node = np.concatenate([edges[:, 0], edges[:, 1]]).astype(np.int64)
    reached = np.concatenate([groups[edges[:, 1]], groups[edges[:, 0]]])
    degree = np.bincount(node, minlength=int(num_nodes)).astype(np.float64)

    keys = node * num_groups + reached
    _unique, counts = np.unique(keys, return_counts=True)
    owner = (_unique // num_groups).astype(np.int64)
    squares = np.zeros(int(num_nodes))
    np.add.at(squares, owner, counts.astype(np.float64) ** 2)
    share = np.divide(squares, degree ** 2, out=np.zeros(int(num_nodes)),
                      where=degree > 0)
    participation = np.where(degree > 0, 1.0 - share, 0.0)

    within = np.zeros(int(num_nodes))
    same = reached == groups[node]
    np.add.at(within, node[same], 1.0)
    # Standardized inside each group: modules differ in density.
    per_group_sum = np.bincount(groups, weights=within, minlength=num_groups)
    per_group_sq = np.bincount(groups, weights=within ** 2,
                               minlength=num_groups)
    size = np.bincount(groups, minlength=num_groups).astype(np.float64)
    mean = np.divide(per_group_sum, size, out=np.zeros(num_groups),
                     where=size > 0)
    variance = np.divide(per_group_sq, size, out=np.zeros(num_groups),
                         where=size > 0) - mean ** 2
    deviation = np.sqrt(np.maximum(variance, 0.0))
    spread = deviation[groups]
    z = np.divide(within - mean[groups], spread, out=np.zeros(int(num_nodes)),
                  where=spread > 0)
    return participation.astype(np.float32), z.astype(np.float32)


def crowding(coords, k=CROWDING_K):
    """Inverted distance to the k-th nearest node, so high means packed, or
    None. Divided by distance plus the median, so coincident nodes aren't inf."""
    if not SCIPY_AVAILABLE or coords is None or coords.shape[0] <= k:
        return None
    tree = cKDTree(coords)
    distances = tree.query(coords, k=k + 1, workers=-1)[0][:, -1]
    scale = float(np.median(distances))
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    return (scale / (distances + scale)).astype(np.float32)


def thinning_ranks(num_nodes, groups=None):
    """A stable pseudorandom rank in [0, 1) per node, within groups if given. It
    hashes the index, so a thinned render repeats across machines."""
    n = int(num_nodes)
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    # SplitMix64's finalizer: stateless, scatters consecutive indices.
    h = np.arange(n, dtype=np.uint64) + np.uint64(0x9E3779B97F4A7C15)
    h ^= h >> np.uint64(30)
    h *= np.uint64(0xBF58476D1CE4E5B9)
    h ^= h >> np.uint64(27)
    h *= np.uint64(0x94D049BB133111EB)
    h ^= h >> np.uint64(31)
    keys = (h >> np.uint64(11)).astype(np.float64) / float(1 << 53)

    out = np.empty(n, dtype=np.float32)
    if groups is None:
        order = np.argsort(keys, kind='stable')
        out[order] = np.arange(n, dtype=np.float32) / n
        return out
    labels = np.unique(groups, return_inverse=True)[1]
    order = np.lexsort((keys, labels))
    sorted_labels = labels[order]
    size = np.bincount(sorted_labels)
    starts = np.concatenate([[0], np.cumsum(size)[:-1]])
    within = np.arange(n) - starts[sorted_labels]
    out[order] = (within / size[sorted_labels]).astype(np.float32)
    return out


def hop_distances(num_nodes, edges, seeds):
    """Hops to the nearest seed per node, plus the farthest reached, or None.
    Unreachable nodes come back one past the farthest to keep the range finite."""
    if not SCIPY_AVAILABLE or edges is None or edges.size == 0:
        return None
    if seeds is None or not np.any(seeds):
        return None
    n = int(num_nodes)
    adjacency = coo_matrix(
        (np.ones(edges.shape[0], dtype=np.int8),
         (edges[:, 0], edges[:, 1])), shape=(n, n)).tocsr()
    distances = _scipy_dijkstra(
        adjacency, directed=False, unweighted=True,
        indices=np.flatnonzero(seeds), min_only=True)
    finite = np.isfinite(distances)
    far = float(distances[finite].max()) if finite.any() else 0.0
    out = np.where(finite, distances, far + 1.0)
    return out.astype(np.float32), far + (0.0 if finite.all() else 1.0)


def group_crossings(values, edges):
    if values is None or edges is None or edges.size == 0:
        return None
    return (values[edges[:, 0]] != values[edges[:, 1]]).astype(np.float32)


def slot_mask(values, clause):
    """One clause's verdict over a normalized channel, inclusive at both ends."""
    keep = (values >= float(clause.range_min)) \
        & (values <= float(clause.range_max))
    return ~keep if clause.invert else keep


# Producers read through GraphSource, never a mesh; the host guesses defaults.

SEED_ATTR = "scigraphs_seed"


class ChannelContext:
    """What the channel producers read, fetched at most once each. The default
    ``group_attr``/``weight_attr`` arrive resolved; a GraphSource can't guess."""

    __slots__ = ("source", "settings", "attr", "num_nodes", "_default_group",
                 "_default_weight", "_edges", "_coords", "_weights")

    def __init__(self, source, settings, attr_name="", group_attr="",
                 weight_attr=""):
        self.source = source
        self.settings = settings
        self.attr = attr_name
        self.num_nodes = int(source.num_vertices)
        self._default_group = group_attr or ""
        self._default_weight = weight_attr or ""
        self._edges = False
        self._coords = None
        self._weights = False

    @property
    def edges(self):
        if self._edges is False:
            self._edges = self.source.edges()
        return self._edges

    @property
    def coords(self):
        if self._coords is None:
            self._coords = self.source.coords()
        return self._coords

    @property
    def weights(self):
        if self._weights is False:
            self._weights = self.source.edge_scalar(
                self.attr or self._default_weight)
        return self._weights

    @property
    def groups(self):
        name = self.group_attr()
        if not name:
            return None
        return value_channel(self.source.point_scalar(name), self.num_nodes)[0]

    def group_attr(self):
        return self.attr or self.settings.coarsen_attr or self._default_group

    def point_values(self, name):
        return value_channel(self.source.point_scalar(name), self.num_nodes)


# Each producer returns ``(values, native)``. None means raw numbers to
# normalize; a pair means already in [0, 1], which an all-zero flag must stay.
def _ch_attr(c):
    values, vmin, vmax = c.point_values(c.attr)
    return size_normalized(values, vmin, vmax, c.num_nodes), (vmin, vmax)


def _ch_degree(c):
    if c.edges is None:
        return None, None
    return np.bincount(c.edges.ravel(),
                       minlength=c.num_nodes).astype(np.float32), None


def _ch_length(c):
    if c.edges is None:
        return None, None
    span = c.coords[c.edges[:, 0]] - c.coords[c.edges[:, 1]]
    return np.linalg.norm(span, axis=1).astype(np.float32), None


def _ch_span(c):
    values = group_crossings(c.groups, c.edges) \
        if c.group_attr() and c.edges is not None and c.edges.size else None
    return values, (0.0, 1.0) if values is not None else None


def _ch_position(axis):
    def read(c):
        if c.num_nodes == 0:
            return None, None
        return c.coords[:, axis].astype(np.float32), None
    return read


def _ch_radius(c):
    if c.num_nodes == 0:
        return None, None
    center = c.coords.mean(axis=0)
    return np.linalg.norm(c.coords - center, axis=1).astype(np.float32), None


def _ch_hops(c):
    seeds = c.source.point_flags(c.attr or SEED_ATTR)
    if seeds is not None and not seeds.any():
        seeds = None
    found = hop_distances(c.num_nodes, c.edges, seeds)
    if found is None:
        return None, None
    distances, far = found
    if far <= 0.0:
        return np.zeros(c.num_nodes, dtype=np.float32), (0.0, 0.0)
    return (distances / far).astype(np.float32), (0.0, far)


def _ch_thin(c):
    return thinning_ranks(c.num_nodes, c.groups), (0.0, 1.0)


def _module_roles(c):
    if not c.group_attr() or c.edges is None or c.edges.size == 0:
        return None, None
    return module_roles_from_groups(c.groups, c.edges, c.num_nodes)


def _ch_participation(c):
    # Already a fraction: 0 means every edge stays inside the group.
    return _flag(_module_roles(c)[0])


def _ch_module_z(c):
    return _module_roles(c)[1], None


def _flag(values):
    return values, (0.0, 1.0) if values is not None else None


PRODUCERS = {
    'ATTR': _ch_attr,
    'DEGREE': _ch_degree,
    'STRENGTH': lambda c: (strength_channel(c.num_nodes, c.edges, c.weights),
                           None),
    'CORE': lambda c: (core_numbers(c.num_nodes, c.edges).astype(np.float32),
                       None),
    'COMPONENT': lambda c: (
        component_sizes(c.num_nodes, c.edges).astype(np.float32), None),
    'CLUSTERING': lambda c: (clustering_coefficients(c.num_nodes, c.edges),
                             (0.0, 1.0)),
    'PAGERANK': lambda c: (pagerank_scores(c.num_nodes, c.edges), None),
    'BETWEEN': lambda c: (betweenness_sampled(c.num_nodes, c.edges), None),
    'ARTICULATION': lambda c: _flag(articulation_flags(c.num_nodes, c.edges)),
    'PARTICIPATION': _ch_participation,
    'MODULE_Z': _ch_module_z,
    'POS_X': _ch_position(0),
    'POS_Y': _ch_position(1),
    'POS_Z': _ch_position(2),
    'RADIUS': _ch_radius,
    'CROWDING': lambda c: (crowding(c.coords), (0.0, 1.0)),
    'HOPS': _ch_hops,
    'THIN': _ch_thin,
    'EDGE_ATTR': lambda c: (c.source.edge_scalar(c.attr), None),
    'LENGTH': _ch_length,
    'SPAN': _ch_span,
    'DISPARITY': lambda c: (
        disparity_significance(c.num_nodes, c.edges, c.weights), (0.0, 1.0)),
    'SUPPORT': lambda c: (triangle_support(c.num_nodes, c.edges), None),
    'OVERLAP': lambda c: (neighborhood_overlap(c.num_nodes, c.edges),
                          (0.0, 1.0)),
    'EDGE_MINDEG': lambda c: (rich_club_channel(c.num_nodes, c.edges), None),
    'BRIDGE': lambda c: _flag(bridge_flags(c.num_nodes, c.edges)),
}


def channel_values(source, settings, name, attr_name="", group_attr="",
                   weight_attr=""):
    """``(values, native_min, native_max, domain)`` for one channel, uncached.
    ``values`` is normalized to [0, 1], or None when it cannot be computed; the
    native bounds are what the slider prints. Caching is the host's job."""
    domain = 'EDGE' if name in EDGE_CHANNELS else 'POINT'

    producer = PRODUCERS.get(name)
    if producer is None:
        values, native = None, None
    else:
        values, native = producer(ChannelContext(
            source, settings, attr_name, group_attr, weight_attr))

    if values is None:
        norm, native = None, (0.0, 1.0)
    elif native is None:
        native = (float(np.nanmin(values)), float(np.nanmax(values)))
        norm = channel_normalized(values)
    else:
        norm = np.asarray(values, dtype=np.float32)

    return (norm, native[0], native[1], domain)
