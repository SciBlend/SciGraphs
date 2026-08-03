"""ForceAtlas2 layout helpers."""

from .common import *
from .networkx_layouts import _spring_layout_2d

# NetworkX grew a real ForceAtlas2 in 3.4 (``nx.forceatlas2_layout``), and it is
# the one this addon should be using
# NetworkX's version takes a ``dim`` argument, so it is also the first ForceAtlas2
# here that can actually be three-dimensional.
NX_FA2 = getattr(nx, "forceatlas2_layout", None)


def _forceatlas2_layout(G, iterations, scale, scaling_ratio=2.0, gravity=1.0,
                        strong_gravity=False, lin_log_mode=False, barnes_hut_optimize=True,
                        barnes_hut_theta=1.2, jitter_tolerance=1.0, edge_weight_influence=1.0,
                        dim=3):
    """
    ForceAtlas2 (Jacomy et al. 2014), the algorithm Gephi uses.
    """
    import time
    start = time.time()
    n = len(G.nodes())

    if NX_FA2 is not None:
        print(f"Computing ForceAtlas2 ({dim}D, networkx) for {n} nodes...")
        pos = NX_FA2(
            G,
            max_iter=max(1, int(iterations)),
            jitter_tolerance=jitter_tolerance,
            scaling_ratio=scaling_ratio,
            gravity=gravity,
            strong_gravity=strong_gravity,
            # NetworkX's name for outboundAttractionDistribution, which the old
            # call had switched on.
            distributed_action=True,
            linlog=lin_log_mode,
            weight="weight" if edge_weight_influence else None,
            dim=int(dim),
            seed=_get_layout_rng().randint(0, 2**31 - 1),
        )
        arr = np.zeros((n, 3), dtype=np.float64)
        for node, coord in pos.items():
            c = np.asarray(coord, dtype=np.float64)
            arr[node, :c.size] = c
        # Normalise to the requested scale. NetworkX returns whatever size the
        # simulation reached, which for a force layout is arbitrary.
        span = float(np.abs(arr - arr.mean(axis=0)).max())
        if span > 1e-9:
            arr = (arr - arr.mean(axis=0)) * (scale / span)
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
            pos_array[node] = [x * scale, y * scale, 0]
        print(f"  ForceAtlas2 completed in {time.time() - start:.2f}s")
        return pos_array

    print("ForceAtlas2 unavailable (needs networkx >= 3.4 or the fa2 package); "
          "falling back to Spring 2D, which is NOT ForceAtlas2")
    return _spring_layout_2d(G, iterations, scale)

def _forceatlas2_iteration(G, current_pos, scale, props=None):
    """
    Execute a small batch of ForceAtlas2 iterations starting from current positions.
    Returns (new_positions, energy).
    """
    iters = int(getattr(props, "iterations_per_frame", 5)) if props else 5

    if NX_FA2 is not None:

        current = np.asarray(current_pos, dtype=np.float64)
        pos = {i: tuple(current[i]) for i in range(current.shape[0])}
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
            new_pos[node, :c.size] = c
        return new_pos, float(np.linalg.norm(new_pos - current))

    if not FA2_AVAILABLE:
        return _spring_layout_2d(G, 50, scale), 0.0

    # Parameters
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

    # Convert current positions to 2D dict format for ForceAtlas2
    pos_dict = {}
    for i in range(len(current_pos)):
        pos_dict[i] = (current_pos[i][0], current_pos[i][1])

    # Execute small batch (configurable iterations per frame)
    iterations_per_frame = props.iterations_per_frame if props and hasattr(props, 'iterations_per_frame') else 5
    new_pos_dict = forceatlas2.forceatlas2_networkx_layout(
        G, pos=pos_dict, iterations=iterations_per_frame
    )

    # Convert back to 3D array
    new_pos = np.zeros_like(current_pos)
    for node, (x, y) in new_pos_dict.items():
        new_pos[node] = [x, y, current_pos[node][2]]

    # Calculate energy as movement from previous position
    energy = np.linalg.norm(new_pos - current_pos)

    return new_pos, energy

__all__ = [name for name in globals() if not name.startswith('__')]
