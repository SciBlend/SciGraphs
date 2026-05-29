"""Public layout dispatcher."""

from .common import *
from .basic import *
from .networkx_layouts import *
from .forceatlas import *
from .igraph_layouts import *
from .circle_packing import *
from .hierarchical import *
from .yifan_hu import *

def apply_graph_layout(obj, algorithm='SPRING_3D', iterations=50, scale=5.0, props=None):
    """
    Calculates node positions using the specified layout algorithm
    and updates the object's custom property.

    Args:
        props: Blender scene properties (context.scene.scigraphs) for
               algorithm-specific parameters. If None, uses defaults.
    """
    start_time = time.time()
    _reset_layout_rng()

    G, num_nodes = _build_networkx_graph(obj)
    if G is None:
        _log_layout(algorithm, 0, 0, None, start_time, False, "Graph construction failed")
        return False

    num_edges = G.number_of_edges()

    # Build parameters dict for logging
    params = {
        'algorithm': algorithm,
        'iterations': iterations,
        'scale': scale,
        'num_nodes': num_nodes,
        'num_edges': num_edges
    }

    # Track actual algorithm used (for fallback detection)
    actual_algorithm = algorithm

    try:
        if algorithm == 'RANDOM':
            pos = _random_layout(num_nodes, scale)
        elif algorithm == 'GRID':
            pos = _grid_layout(num_nodes, scale)
        elif algorithm == 'SPRING':
            pos = _spring_layout_2d(G, iterations, scale)
        elif algorithm == 'SPRING_3D':
            pos = _spring_layout_3d(G, iterations, scale)
        elif algorithm == 'CIRCLE_PACKING':
            pos, radii = _circle_packing_layout(G, iterations, scale)
            # Store radii as custom property and mesh attribute
            obj["circle_packing_radii"] = radii.tolist()
            _store_radii_as_mesh_attribute(obj, radii)
        elif algorithm == 'FORCEATLAS2':
            if not FA2_AVAILABLE:
                actual_algorithm = 'SPRING (2D fallback)'
            pos = _forceatlas2_layout(G, iterations, scale)
        elif algorithm == 'IGRAPH_FR':
            if not IGRAPH_AVAILABLE:
                actual_algorithm = 'SPRING (2D fallback)'
            pos = _igraph_fruchterman_reingold(G, iterations, scale)
        elif algorithm == 'IGRAPH_KK':
            if not IGRAPH_AVAILABLE:
                actual_algorithm = 'SPRING (2D fallback)'
            if props:
                pos = _igraph_kamada_kawai(
                    G, scale,
                    maxiter=props.igraph_kk_maxiter if props.igraph_kk_maxiter > 0 else None,
                    epsilon=props.igraph_kk_epsilon if props.igraph_kk_epsilon > 0 else None,
                    kkconst=props.igraph_kk_kkconst if props.igraph_kk_kkconst > 0 else None
                )
            else:
                pos = _igraph_kamada_kawai(G, scale)
        elif algorithm == 'IGRAPH_DRL':
            if not IGRAPH_AVAILABLE:
                actual_algorithm = 'SPRING (2D fallback)'
            if props:
                pos = _igraph_drl(G, iterations, scale, **_get_drl_kwargs_from_props(props))
            else:
                pos = _igraph_drl(G, iterations, scale)
        elif algorithm == 'IGRAPH_DRL_2D':
            if not IGRAPH_AVAILABLE:
                actual_algorithm = 'SPRING (2D fallback)'
            if props:
                pos = _igraph_drl_2d(G, iterations, scale, **_get_drl_kwargs_from_props(props))
            else:
                pos = _igraph_drl_2d(G, iterations, scale)
        elif algorithm == 'IGRAPH_LGL':
            if not IGRAPH_AVAILABLE:
                actual_algorithm = 'SPRING (2D fallback)'
            if props:
                pos = _igraph_lgl(
                    G, scale,
                    maxiter=props.igraph_lgl_maxiter,
                    maxdelta=props.igraph_lgl_maxdelta if props.igraph_lgl_maxdelta > 0 else None,
                    area=props.igraph_lgl_area if props.igraph_lgl_area > 0 else None,
                    coolexp=props.igraph_lgl_coolexp,
                    repulserad=props.igraph_lgl_repulserad if props.igraph_lgl_repulserad > 0 else None,
                    cellsize=props.igraph_lgl_cellsize if props.igraph_lgl_cellsize > 0 else None
                )
            else:
                pos = _igraph_lgl(G, scale)
        elif algorithm == 'SPHERE':
            pos = _sphere_layout(num_nodes, scale)
        elif algorithm == 'SPECTRAL_3D':
            pos = _spectral_layout_3d(G, scale)
        elif algorithm == 'SPIRAL_3D':
            pos = _spiral_layout_3d(num_nodes, scale)
        elif algorithm == 'HELIX':
            pos = _helix_layout(num_nodes, scale)
        elif algorithm == 'CUBE':
            pos = _cube_layout(num_nodes, scale)
        elif algorithm == 'HIERARCHICAL_3D':
            pos = _hierarchical_layout_3d(G, scale)
        elif algorithm == 'BIPARTITE_3D':
            pos = _bipartite_layout_3d(G, scale)
        elif algorithm == 'IGRAPH_DH':
            if not IGRAPH_AVAILABLE:
                actual_algorithm = 'SPRING (2D fallback)'
            if props:
                pos = _igraph_davidson_harel(
                    G, iterations, scale,
                    maxiter=props.igraph_dh_maxiter,
                    fineiter=props.igraph_dh_fineiter,
                    cool_fact=props.igraph_dh_cool_fact
                )
            else:
                pos = _igraph_davidson_harel(G, iterations, scale)
        elif algorithm == 'IGRAPH_GRAPHOPT':
            if not IGRAPH_AVAILABLE:
                actual_algorithm = 'SPRING (2D fallback)'
            if props:
                pos = _igraph_graphopt(
                    G, iterations, scale,
                    spring_length=props.igraph_graphopt_spring_length if props.igraph_graphopt_spring_length > 0 else None,
                    node_charge=props.igraph_graphopt_node_charge if props.igraph_graphopt_node_charge != 0 else None,
                    spring_constant=props.igraph_graphopt_spring_constant if props.igraph_graphopt_spring_constant > 0 else None,
                    node_mass=props.igraph_graphopt_node_mass if props.igraph_graphopt_node_mass > 0 else None
                )
            else:
                pos = _igraph_graphopt(G, iterations, scale)
        elif algorithm == 'MDS_3D':
            pos = _mds_layout_3d(G, scale)
        elif algorithm == 'YIFAN_HU':
            pos = _yifan_hu_layout(G, iterations, scale, props=props)
        elif algorithm in GRAPHVIZ_ENGINES:
            pos = _graphviz_engine_layout(G, algorithm, iterations, scale, props=props)
        elif algorithm == 'SUGIYAMA':
            pos = _sugiyama_layout(G, scale)
        elif algorithm == 'CIRCULAR_HIERARCHY':
            pos = _circular_hierarchy_layout(G, scale)
        else:
            pos = _random_layout(num_nodes, scale)

        obj["node_positions"] = pos.flatten().tolist()

        _log_layout(algorithm, num_nodes, num_edges, params, start_time, True, None, actual_algorithm)
        return True

    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        _log_layout(algorithm, num_nodes, num_edges, params, start_time, False, error_msg, actual_algorithm)
        return False

__all__ = [name for name in globals() if not name.startswith('__')]
