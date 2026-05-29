"""ForceAtlas2 layout helpers."""

from .common import *
from .networkx_layouts import _spring_layout_2d

def _forceatlas2_layout(G, iterations, scale, scaling_ratio=2.0, gravity=1.0,
                        strong_gravity=False, lin_log_mode=False, barnes_hut_optimize=True,
                        barnes_hut_theta=1.2, jitter_tolerance=1.0, edge_weight_influence=1.0):
    """
    ForceAtlas2 layout (Gephi's algorithm).
    Very fast for large graphs, better than Spring.
    """
    if not FA2_AVAILABLE:
        print("ForceAtlas2 not available, falling back to Spring 2D")
        return _spring_layout_2d(G, iterations, scale)

    import time
    start = time.time()
    print(f"Computing ForceAtlas2 layout for {len(G.nodes())} nodes...")

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

    # Convert NetworkX graph to list of edges
    positions = forceatlas2.forceatlas2_networkx_layout(G, pos=None, iterations=iterations)

    # Convert to 3D array with z=0
    pos_array = np.zeros((len(G.nodes()), 3))
    for node, (x, y) in positions.items():
        pos_array[node] = [x * scale, y * scale, 0]

    print(f"  ForceAtlas2 completed in {time.time() - start:.2f}s")
    return pos_array

def _forceatlas2_iteration(G, current_pos, scale, props=None):
    """
    Execute a small batch of ForceAtlas2 iterations starting from current positions.
    Returns (new_positions, energy).
    """
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
