"""ForceAtlas2 layout helpers."""

from .common import *
from .networkx_layouts import _spring_layout_2d

try:
    from .simulation import (ForceSim, random_positions, quadtree_repulsion,
                             DIRECT_MAX, DEFAULT_THETA)
    FORCESIM_AVAILABLE = True
    FORCESIM_REASON = None
except Exception as _error:  # noqa: BLE001
    ForceSim = None
    random_positions = None
    quadtree_repulsion = None
    DIRECT_MAX = 400
    DEFAULT_THETA = 0.6
    FORCESIM_AVAILABLE = False
    FORCESIM_REASON = str(_error)

# NetworkX 3.4+ ForceAtlas2 is preferred: it takes ``dim``, so only it is 3D.
NX_FA2 = getattr(nx, "forceatlas2_layout", None)

_FA2_PROPS = ('scaling_ratio', 'gravity', 'strong_gravity', 'lin_log_mode',
              'barnes_hut_optimize', 'barnes_hut_theta', 'jitter_tolerance',
              'edge_weight_influence')

_FA2_GRAVITY_NORM = 0.1

_FA2_SIM_SCALE = 5.0

_FA2_SETTLED = 0.1


def _fa2_kwargs_from_props(props):
    """ForceAtlas2 parameters from scene properties, keyed as this module's
    keywords: each is ``props.fa2_<keyword>``."""
    if not props:
        return {}
    kwargs = {}
    for name in _FA2_PROPS:
        value = getattr(props, 'fa2_' + name, None)
        if value is not None:
            kwargs[name] = value
    return kwargs


def _fa2_rescale(arr, scale):
    """Center on the origin and fit the widest axis to *scale*, so every branch
    answers the same parameter with the same size."""
    if arr.shape[0] == 0:
        return arr
    arr = arr - arr.mean(axis=0)
    span = float(np.abs(arr).max())
    if span > 1e-9:
        arr = arr * (scale / span)
    return arr


def _fa2_edges(G, node_to_idx, edge_weight_influence):
    """``(edges, weights)`` for :class:`ForceSim`, indices in ``G.nodes()``
    order. *weights* is None unless the graph really carries them."""
    if not G.number_of_edges():
        return np.zeros((0, 2), dtype=np.int64), None
    pairs = np.fromiter(
        (node_to_idx[end] for u, v in G.edges() for end in (u, v)),
        dtype=np.int64, count=2 * G.number_of_edges()).reshape(-1, 2)
    if not edge_weight_influence:
        return pairs, None
    weights = np.fromiter(
        (float(data.get("weight", 1.0)) for _u, _v, data in G.edges(data=True)),
        dtype=np.float64, count=pairs.shape[0])
    return pairs, (None if np.all(weights == 1.0) else weights)


def _fa2_report_barnes_hut(n, theta):
    """Say when Barnes-Hut was asked for and something else ran.

    Two ways it does not engage, both silent inside :class:`ForceSim`: under
    DIRECT_MAX the exact all-pairs sum is cheaper as well as exact, and without
    the quadtree that scigraphs-utils 0.2+ compiles out of Graphviz's sfdp the
    tree falls back to the coarse grid, whose force error is 14% against the
    tree's 5% on a settled 2000-node graph."""
    if n <= DIRECT_MAX:
        print(f"  Barnes-Hut not used below {DIRECT_MAX} nodes (n={n}): the "
              f"exact all-pairs repulsion is cheaper there, and exact")
    elif quadtree_repulsion is None or quadtree_repulsion() is None:
        print("  Barnes-Hut unavailable: this scigraphs-utils has no quadtree "
              "(needs 0.2+); repulsion falls back to the coarse grid, so "
              f"theta={theta} does nothing")


def _forceatlas2_forcesim(G, iterations, scale, nodes, node_to_idx,
                          scaling_ratio, gravity, strong_gravity, lin_log_mode,
                          barnes_hut_optimize, barnes_hut_theta,
                          jitter_tolerance, edge_weight_influence, dim):
    """ForceAtlas2 through this repo's own :class:`ForceSim`.

    Same force model the animated preview runs, and the default here because
    ``networkx.forceatlas2_layout`` builds an (n, n, 3) array every iteration.
    That is n squared in both time and memory: 200 iterations of it take 2.8 s
    on a 1024-node grid and 47 s on 2000 Watts-Strogatz nodes, against 0.45 and
    1.6 here, and at 30,000 nodes it wants about 50 minutes and 21.6 GB where
    this path finishes in 14 s.

    A ForceSim iteration is not a NetworkX one, though. It caps a node's move
    at ``k`` and NetworkX caps nothing, so a graph that has to expand spends
    its first iterations travelling rather than arranging. Graph-distance
    correlation at 50 and 200 iterations, NetworkX then here: 32x32 grid
    0.367/0.861 against 0.197/0.712, balanced tree 3^6 0.224/0.377 against
    0.135/0.222, Watts-Strogatz 2000 0.309/0.420 against 0.098/0.385,
    Barabasi-Albert 1000 -0.011/-0.001 against 0.014/0.065, two-community SBM
    0.675/0.673 against 0.675/0.680. So at equal iterations this trails on the
    shapes that expand most and leads on the rest.

    Iterations are the thing to spend, not seconds. At 800 of them the grid
    reaches 0.852 in 1.9 s and the tree 0.398 in 3.1 s, both past NetworkX's
    200-iteration answer and still four to seventeen times faster. Run-to-run
    seed noise on these figures is about 0.05.
    """
    n = len(nodes)
    edges, weights = _fa2_edges(G, node_to_idx, edge_weight_influence)
    seed = _get_layout_rng().randint(0, 2 ** 31 - 1)
    mode = 'TREE' if barnes_hut_optimize else 'GRID'

    print(f"Computing ForceAtlas2 ({dim}D, ForceSim) for {n} nodes...")
    if barnes_hut_optimize:
        _fa2_report_barnes_hut(n, barnes_hut_theta)

    sim = ForceSim(
        random_positions(n, scale=_FA2_SIM_SCALE, seed=seed, dimensions=dim),
        edges, weights=weights, seed=seed, model='FA2',
        repulsion=scaling_ratio,
        gravity=gravity * _FA2_GRAVITY_NORM,
        strong_gravity=strong_gravity,
        lin_log=lin_log_mode,
        edge_weight_influence=edge_weight_influence,
        jitter_tolerance=jitter_tolerance,
        scale=_FA2_SIM_SCALE, dimensions=int(dim),
        repulsion_mode=mode, theta=barnes_hut_theta,
    )
    moved = sim.step(max(1, int(iterations)))

    settled = moved / max(n * sim.k, 1e-12)
    if settled > _FA2_SETTLED:
        print(f"  Still moving {settled:.2f}k per node after {iterations} "
              f"iterations, so this layout is not settled; raise Iterations")
    return _fa2_rescale(np.asarray(sim.positions, dtype=np.float64), scale)


def _forceatlas2_layout(G, iterations, scale, scaling_ratio=2.0, gravity=1.0,
                        strong_gravity=False, lin_log_mode=False, barnes_hut_optimize=True,
                        barnes_hut_theta=DEFAULT_THETA, jitter_tolerance=1.0,
                        edge_weight_influence=1.0, dim=3):
    """ForceAtlas2 (Jacomy et al. 2014), as Gephi uses. Runs on this repo's
    :class:`ForceSim`; NetworkX and then the ``fa2`` package are kept only as
    fallbacks, and last a spring layout that is not it.
    Node labels need not be 0..n-1: positions come back in ``G.nodes()`` order."""
    import time
    start = time.time()
    nodes = list(G.nodes())
    n = len(nodes)
    node_to_idx = {node: i for i, node in enumerate(nodes)}

    if n == 0:
        return np.zeros((0, 3), dtype=np.float64)

    if FORCESIM_AVAILABLE:
        arr = _forceatlas2_forcesim(
            G, iterations, scale, nodes, node_to_idx, scaling_ratio, gravity,
            strong_gravity, lin_log_mode, barnes_hut_optimize,
            barnes_hut_theta, jitter_tolerance, edge_weight_influence, dim)
        print(f"  ForceAtlas2 completed in {time.time() - start:.2f}s")
        return arr

    if NX_FA2 is not None:
        print(f"Computing ForceAtlas2 ({dim}D, networkx) for {n} nodes...")
        pos = NX_FA2(
            G,
            max_iter=max(1, int(iterations)),
            jitter_tolerance=jitter_tolerance,
            scaling_ratio=scaling_ratio,
            gravity=gravity,
            strong_gravity=strong_gravity,
            # NetworkX's name for outboundAttractionDistribution.
            distributed_action=True,
            linlog=lin_log_mode,
            weight="weight" if edge_weight_influence else None,
            dim=int(dim),
            seed=_get_layout_rng().randint(0, 2**31 - 1),
        )
        arr = np.zeros((n, 3), dtype=np.float64)
        for node, coord in pos.items():
            c = np.asarray(coord, dtype=np.float64)
            arr[node_to_idx[node], :c.size] = c
        # NetworkX returns whatever size the simulation reached, so rescale.
        arr = _fa2_rescale(arr, scale)
        print(f"  ForceAtlas2 completed in {time.time() - start:.2f}s")
        return arr

    if FA2_AVAILABLE:
        print(f"Computing ForceAtlas2 (2D, fa2 package) for {n} nodes...")
        forceatlas2 = ForceAtlas2(
            outboundAttractionDistribution=True,
            linLogMode=lin_log_mode,
            adjustSizes=False,
            edgeWeightInfluence=edge_weight_influence,
            jitterTolerance=jitter_tolerance,
            barnesHutOptimize=barnes_hut_optimize,
            barnesHutTheta=barnes_hut_theta,
            scalingRatio=scaling_ratio,
            strongGravityMode=strong_gravity,
            gravity=gravity,
            verbose=False
        )
        positions = forceatlas2.forceatlas2_networkx_layout(
            G, pos=None, iterations=iterations)
        pos_array = np.zeros((n, 3))
        for node, (x, y) in positions.items():
            pos_array[node_to_idx[node]] = [x, y, 0]
        pos_array = _fa2_rescale(pos_array, scale)
        print(f"  ForceAtlas2 completed in {time.time() - start:.2f}s")
        return pos_array

    print("ForceAtlas2 unavailable (needs ForceSim, networkx >= 3.4 or the fa2 "
          "package); falling back to Spring 2D, which is NOT ForceAtlas2")
    return _spring_layout_2d(G, iterations, scale)


def _forceatlas2_fallback():
    """``(missing_library, substitute)`` for FORCEATLAS2, as
    ``common._resolve_fallback`` returns it for everything else.

    That one still answers for the era when NetworkX was the preferred backend,
    so it announced a spring fallback whenever ``networkx.forceatlas2_layout``
    was absent. ForceSim needs neither library, so the announcement would now
    be for a substitution that did not happen, and the dispatcher would log a
    different algorithm than the one that ran."""
    if FORCESIM_AVAILABLE or NX_FA2 is not None:
        return None, None
    if FA2_AVAILABLE:
        return "ForceSim or networkx >= 3.4", "FORCEATLAS2 (2D, fa2 package)"
    return "ForceSim, networkx >= 3.4 or fa2", "SPRING (2D)"


def _forceatlas2_iteration(G, current_pos, scale, props=None):
    """Advance ForceAtlas2 from *current_pos*, returning (positions, energy).
    *current_pos* is in ``G.nodes()`` order, as :func:`_forceatlas2_layout`
    returns it."""
    iters = int(getattr(props, "iterations_per_frame", 5)) if props else 5
    nodes = list(G.nodes())
    node_to_idx = {node: i for i, node in enumerate(nodes)}

    if NX_FA2 is not None:

        current = np.asarray(current_pos, dtype=np.float64)
        pos = {nodes[i]: current[i].copy() for i in range(current.shape[0])}
        out = NX_FA2(
            G, pos=pos, max_iter=max(1, iters),
            jitter_tolerance=getattr(props, "fa2_jitter_tolerance", 1.0) if props else 1.0,
            scaling_ratio=getattr(props, "fa2_scaling_ratio", 2.0) if props else 2.0,
            gravity=getattr(props, "fa2_gravity", 1.0) if props else 1.0,
            strong_gravity=getattr(props, "fa2_strong_gravity", False) if props else False,
            distributed_action=True,
            linlog=getattr(props, "fa2_lin_log_mode", False) if props else False,
            dim=3,
        )
        new_pos = np.zeros_like(current)
        for node, coord in out.items():
            c = np.asarray(coord, dtype=np.float64)
            new_pos[node_to_idx[node], :c.size] = c
        return new_pos, float(np.linalg.norm(new_pos - current))

    if not FA2_AVAILABLE:
        return _spring_layout_2d(G, 50, scale), 0.0

    if props:
        scaling_ratio = props.fa2_scaling_ratio
        gravity = props.fa2_gravity
        strong_gravity = props.fa2_strong_gravity
        lin_log_mode = props.fa2_lin_log_mode
        barnes_hut_optimize = props.fa2_barnes_hut_optimize
        barnes_hut_theta = props.fa2_barnes_hut_theta
        jitter_tolerance = props.fa2_jitter_tolerance
        edge_weight_influence = props.fa2_edge_weight_influence
    else:
        scaling_ratio = 2.0
        gravity = 1.0
        strong_gravity = False
        lin_log_mode = False
        barnes_hut_optimize = True
        barnes_hut_theta = 1.2
        jitter_tolerance = 1.0
        edge_weight_influence = 1.0

    forceatlas2 = ForceAtlas2(
        outboundAttractionDistribution=True,
        linLogMode=lin_log_mode,
        adjustSizes=False,
        edgeWeightInfluence=edge_weight_influence,
        jitterTolerance=jitter_tolerance,
        barnesHutOptimize=barnes_hut_optimize,
        barnesHutTheta=barnes_hut_theta,
        scalingRatio=scaling_ratio,
        strongGravityMode=strong_gravity,
        gravity=gravity,
        verbose=False
    )

    # The fa2 package is 2D only, so Z is set aside and restored below.
    pos_dict = {}
    for i in range(len(current_pos)):
        pos_dict[nodes[i]] = (current_pos[i][0], current_pos[i][1])

    new_pos_dict = forceatlas2.forceatlas2_networkx_layout(
        G, pos=pos_dict, iterations=iters
    )

    new_pos = np.zeros_like(current_pos)
    for node, (x, y) in new_pos_dict.items():
        i = node_to_idx[node]
        new_pos[i] = [x, y, current_pos[i][2]]

    energy = np.linalg.norm(new_pos - current_pos)

    return new_pos, energy

__all__ = [name for name in globals() if not name.startswith('__')]
