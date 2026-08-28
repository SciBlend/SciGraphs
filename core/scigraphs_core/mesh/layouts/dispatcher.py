"""Public layout dispatcher."""

import traceback

from .common import *
from .basic import *
from .networkx_layouts import *
from .forceatlas import *
from .igraph_layouts import *
from .circle_packing import *
from .hierarchical import *
from .yifan_hu import *

def apply_graph_layout(obj, algorithm='SPRING_3D', iterations=50, scale=5.0, props=None,
                       edge_pairs=None):
    """Run a layout and store the result in ``obj["node_positions"]``. *props*
    supplies algorithm-specific parameters and may be None. *edge_pairs* carries
    a mesh-native object's topology, since this package never reads meshes; an
    object with neither it nor ``edges_data`` is refused, not laid out flat.
    An unknown *algorithm* is refused too, rather than laid out at random."""
    start_time = time.time()
    _reset_layout_rng()

    num_nodes = 0
    num_edges = 0
    params = None
    radii = None

    # Diverges from `algorithm` when a library is missing and we fall back.
    missing_library, actual_algorithm = _resolve_fallback(algorithm)
    if algorithm == 'FORCEATLAS2':
        missing_library, actual_algorithm = _forceatlas2_fallback()
    if actual_algorithm is None:
        actual_algorithm = algorithm

    try:
        G, num_nodes = _build_networkx_graph(obj, edge_pairs)
        if G is None:
            _log_layout(algorithm, 0, 0, None, start_time, False, "Graph construction failed")
            return False

        num_edges = G.number_of_edges()

        params = {
            'iterations': iterations,
            'scale': scale,
            'num_nodes': num_nodes,
            'num_edges': num_edges
        }

        if algorithm == 'RANDOM':
            pos = _call_with_props(_random_layout, num_nodes, scale, props=props)
        elif algorithm == 'GRID':
            pos = _call_with_props(_grid_layout, num_nodes, scale, props=props)
        elif algorithm == 'SPRING':
            pos = _call_with_props(_spring_layout_2d, G, iterations, scale, props=props)
        elif algorithm == 'SPRING_3D':
            pos = _call_with_props(_spring_layout_3d, G, iterations, scale, props=props)
        elif algorithm == 'CIRCLE_PACKING':
            pos, radii = _call_with_props(_circle_packing_layout, G, iterations, scale,
                                          props=props)
        elif algorithm == 'FORCEATLAS2':
            pos = _forceatlas2_layout(G, iterations, scale,
                                      **_fa2_kwargs_from_props(props))
        elif algorithm == 'IGRAPH_FR':
            pos = _call_with_props(_igraph_fruchterman_reingold, G, iterations, scale,
                                   props=props)
        elif algorithm == 'IGRAPH_KK':
            if props:
                pos = _igraph_kamada_kawai(
                    G, scale,
                    maxiter=_positive_prop(props, 'igraph_kk_maxiter'),
                    epsilon=_positive_prop(props, 'igraph_kk_epsilon'),
                    kkconst=_positive_prop(props, 'igraph_kk_kkconst')
                )
            else:
                pos = _igraph_kamada_kawai(G, scale)
        elif algorithm == 'IGRAPH_DRL':
            if props:
                pos = _igraph_drl(G, iterations, scale, **_get_drl_kwargs_from_props(props))
            else:
                pos = _igraph_drl(G, iterations, scale)
        elif algorithm == 'IGRAPH_DRL_2D':
            if props:
                pos = _igraph_drl_2d(G, iterations, scale, **_get_drl_kwargs_from_props(props))
            else:
                pos = _igraph_drl_2d(G, iterations, scale)
        elif algorithm == 'IGRAPH_LGL':
            if props:
                pos = _igraph_lgl(
                    G, scale,
                    maxiter=props.igraph_lgl_maxiter,
                    maxdelta=_positive_prop(props, 'igraph_lgl_maxdelta'),
                    area=_positive_prop(props, 'igraph_lgl_area'),
                    coolexp=props.igraph_lgl_coolexp,
                    repulserad=_positive_prop(props, 'igraph_lgl_repulserad'),
                    cellsize=_positive_prop(props, 'igraph_lgl_cellsize')
                )
            else:
                pos = _igraph_lgl(G, scale)
        elif algorithm == 'SPHERE':
            pos = _call_with_props(_sphere_layout, num_nodes, scale, props=props)
        elif algorithm == 'SPECTRAL_3D':
            pos = _call_with_props(_spectral_layout_3d, G, scale, props=props)
        elif algorithm == 'SPIRAL_3D':
            pos = _call_with_props(_spiral_layout_3d, num_nodes, scale, props=props)
        elif algorithm == 'HELIX':
            pos = _call_with_props(_helix_layout, num_nodes, scale, props=props)
        elif algorithm == 'CUBE':
            pos = _call_with_props(_cube_layout, num_nodes, scale, props=props)
        elif algorithm == 'HIERARCHICAL_3D':
            pos = _call_with_props(_hierarchical_layout_3d, G, scale, props=props)
        elif algorithm == 'BIPARTITE_3D':
            pos = _call_with_props(_bipartite_layout_3d, G, scale, props=props)
        elif algorithm == 'IGRAPH_DH':
            if props:
                pos = _igraph_davidson_harel(
                    G, iterations, scale,
                    maxiter=props.igraph_dh_maxiter,
                    fineiter=props.igraph_dh_fineiter,
                    cool_fact=props.igraph_dh_cool_fact,
                    weight_node_dist=props.igraph_dh_weight_node_dist,
                    weight_border=props.igraph_dh_weight_border,
                    weight_edge_lengths=props.igraph_dh_weight_edge_lengths,
                    weight_edge_crossings=props.igraph_dh_weight_edge_crossings,
                    weight_node_edge_dist=props.igraph_dh_weight_node_edge_dist
                )
            else:
                pos = _igraph_davidson_harel(G, iterations, scale)
        elif algorithm == 'IGRAPH_GRAPHOPT':
            if props:
                pos = _igraph_graphopt(G, iterations, scale,
                                       **_graphopt_kwargs_from_props(props))
            else:
                pos = _igraph_graphopt(G, iterations, scale)
        elif algorithm == 'MDS_3D':
            pos = _call_with_props(_mds_layout_3d, G, scale, props=props)
        elif algorithm == 'YIFAN_HU':
            pos = _yifan_hu_layout(G, iterations, scale, props=props)
        elif algorithm in GRAPHVIZ_ENGINES:
            pos = _graphviz_engine_layout(G, algorithm, iterations, scale, props=props)
        elif algorithm == 'SUGIYAMA':
            pos = _call_with_props(_sugiyama_layout, G, scale, props=props)
        elif algorithm == 'CIRCULAR_HIERARCHY':
            pos = _call_with_props(_circular_hierarchy_layout, G, scale, props=props)
        else:
            raise ValueError("unknown layout algorithm %r" % (algorithm,))

        pos = _check_positions(pos, num_nodes, actual_algorithm)

        obj["node_positions"] = pos.flatten().tolist()
        if actual_algorithm != algorithm:
            obj["layout_substituted"] = "%s (no %s)" % (actual_algorithm,
                                                        missing_library)
        elif "layout_substituted" in obj:
            del obj["layout_substituted"]
        if radii is not None:
            obj["circle_packing_radii"] = np.asarray(radii, dtype=float).tolist()
            _store_radii_as_mesh_attribute(obj, radii)

        _log_layout(algorithm, num_nodes, num_edges, params, start_time, True, None,
                    actual_algorithm, missing_library)
        return True

    except MemoryError:
        _log_layout(algorithm, num_nodes, num_edges, params, start_time, False,
                    "MemoryError: out of memory", actual_algorithm, missing_library)
        raise
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        traceback.print_exc()
        _log_layout(algorithm, num_nodes, num_edges, params, start_time, False, error_msg,
                    actual_algorithm, missing_library)
        return False

__all__ = [name for name in globals() if not name.startswith('__')]
