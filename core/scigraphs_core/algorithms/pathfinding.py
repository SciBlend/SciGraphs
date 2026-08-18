# Pathfinding algorithms. Without networkx each logs the missing extra and
# returns None, or `[]` where the caller iterates the result.

import numpy as np

from scigraphs_core.logger import log


try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
    NETWORKX_REASON = None
except ImportError:
    nx = None
    NETWORKX_AVAILABLE = False
    NETWORKX_REASON = ("networkx is not installed; install the 'networkx' "
                       "extra to enable the pathfinding algorithms")


def _networkx_missing(feature, empty=None):
    """Log why `feature` did not run and return `empty`. k_shortest_paths needs
    `[]`, which is already its "no path found", not a None nobody handles."""
    log(f"{feature} unavailable: {NETWORKX_REASON}")
    return empty


def dijkstra_shortest_path(graph_data, source, target=None):
    """Shortest path from ``source`` by Dijkstra. With a ``target``: 'path',
    'distance', 'exists'. Without: 'distances' and 'paths' per reachable node."""
    if not NETWORKX_AVAILABLE:
        return _networkx_missing("Dijkstra shortest path")

    G = nx.DiGraph() if hasattr(graph_data, 'is_directed') and graph_data.is_directed else nx.Graph()
    
    for i, node in enumerate(graph_data.nodes):
        G.add_node(i)
    
    for edge in graph_data.edges:
        src_idx = graph_data.node_to_index.get(edge[0], None)
        tgt_idx = graph_data.node_to_index.get(edge[1], None)
        if src_idx is not None and tgt_idx is not None:
            weight = 1.0
            if hasattr(graph_data, 'edge_weights') and graph_data.edge_weights:
                weight = graph_data.edge_weights.get((edge[0], edge[1]), 1.0)
            G.add_edge(src_idx, tgt_idx, weight=weight)
    
    try:
        if target is not None:
            path = nx.shortest_path(G, source, target, weight='weight')
            distance = nx.shortest_path_length(G, source, target, weight='weight')
            return {
                'path': path,
                'distance': distance,
                'exists': True
            }
        else:
            lengths = nx.single_source_dijkstra_path_length(G, source, weight='weight')
            paths = nx.single_source_dijkstra_path(G, source, weight='weight')
            return {
                'distances': lengths,
                'paths': paths
            }
    except nx.NetworkXNoPath:
        return {
            'path': None,
            'distance': float('inf'),
            'exists': False
        }


def a_star_path(graph_data, source, target, positions):
    """Shortest path by A*, with Euclidean distance over ``positions``, an
    (N, 3) array indexed by node, as heuristic. Returns 'path', 'distance' and
    'exists'. The heuristic stays admissible only while edge weights are no
    smaller than the straight-line distance between their endpoints."""
    if not NETWORKX_AVAILABLE:
        return _networkx_missing("A* shortest path")

    G = nx.DiGraph() if hasattr(graph_data, 'is_directed') and graph_data.is_directed else nx.Graph()
    
    for i, node in enumerate(graph_data.nodes):
        G.add_node(i)
    
    for edge in graph_data.edges:
        src_idx = graph_data.node_to_index.get(edge[0], None)
        tgt_idx = graph_data.node_to_index.get(edge[1], None)
        if src_idx is not None and tgt_idx is not None:
            weight = 1.0
            if hasattr(graph_data, 'edge_weights') and graph_data.edge_weights:
                weight = graph_data.edge_weights.get((edge[0], edge[1]), 1.0)
            G.add_edge(src_idx, tgt_idx, weight=weight)
    
    def heuristic(u, v):
        pos_u = positions[u]
        pos_v = positions[v]
        return np.linalg.norm(pos_u - pos_v)
    
    try:
        path = nx.astar_path(G, source, target, heuristic=heuristic, weight='weight')
        distance = nx.astar_path_length(G, source, target, heuristic=heuristic, weight='weight')
        return {
            'path': path,
            'distance': distance,
            'exists': True
        }
    except nx.NetworkXNoPath:
        return {
            'path': None,
            'distance': float('inf'),
            'exists': False
        }


def bellman_ford_path(graph_data, source):
    """Shortest paths from ``source`` by Bellman-Ford, negative weights allowed:
    'distances', 'paths' and 'has_negative_cycle'. A negative cycle empties the
    first two rather than raising."""
    if not NETWORKX_AVAILABLE:
        return _networkx_missing("Bellman-Ford shortest paths")

    G = nx.DiGraph() if hasattr(graph_data, 'is_directed') and graph_data.is_directed else nx.Graph()
    
    for i, node in enumerate(graph_data.nodes):
        G.add_node(i)
    
    for edge in graph_data.edges:
        src_idx = graph_data.node_to_index.get(edge[0], None)
        tgt_idx = graph_data.node_to_index.get(edge[1], None)
        if src_idx is not None and tgt_idx is not None:
            weight = 1.0
            if hasattr(graph_data, 'edge_weights') and graph_data.edge_weights:
                weight = graph_data.edge_weights.get((edge[0], edge[1]), 1.0)
            G.add_edge(src_idx, tgt_idx, weight=weight)
    
    try:
        lengths, paths = nx.single_source_bellman_ford(G, source, weight='weight')
        return {
            'distances': lengths,
            'paths': paths,
            'has_negative_cycle': False
        }
    except nx.NetworkXUnbounded:
        return {
            'distances': {},
            'paths': {},
            'has_negative_cycle': True
        }


def all_pairs_shortest_path(graph_data):
    """An (N, N) 'distance_matrix' of shortest paths, unreachable pairs left at
    inf. One Dijkstra run per node, so cost grows with nodes times edges."""
    if not NETWORKX_AVAILABLE:
        return _networkx_missing("All-pairs shortest paths")

    G = nx.DiGraph() if hasattr(graph_data, 'is_directed') and graph_data.is_directed else nx.Graph()
    
    for i, node in enumerate(graph_data.nodes):
        G.add_node(i)
    
    for edge in graph_data.edges:
        src_idx = graph_data.node_to_index.get(edge[0], None)
        tgt_idx = graph_data.node_to_index.get(edge[1], None)
        if src_idx is not None and tgt_idx is not None:
            weight = 1.0
            if hasattr(graph_data, 'edge_weights') and graph_data.edge_weights:
                weight = graph_data.edge_weights.get((edge[0], edge[1]), 1.0)
            G.add_edge(src_idx, tgt_idx, weight=weight)
    
    n = len(graph_data.nodes)
    distance_matrix = np.full((n, n), float('inf'))
    
    for source in range(n):
        lengths = nx.single_source_dijkstra_path_length(G, source, weight='weight')
        for target, dist in lengths.items():
            distance_matrix[source, target] = dist
    
    return {
        'distance_matrix': distance_matrix
    }


def k_shortest_paths(graph_data, source, target, k=3):
    """Up to ``k`` shortest simple paths, shortest first, as dicts of 'path' and
    'length'. Empty when no path exists or ``k`` is not positive."""
    if not NETWORKX_AVAILABLE:
        return _networkx_missing("k shortest paths", empty=[])

    G = nx.DiGraph() if hasattr(graph_data, 'is_directed') and graph_data.is_directed else nx.Graph()
    
    for i, node in enumerate(graph_data.nodes):
        G.add_node(i)
    
    for edge in graph_data.edges:
        src_idx = graph_data.node_to_index.get(edge[0], None)
        tgt_idx = graph_data.node_to_index.get(edge[1], None)
        if src_idx is not None and tgt_idx is not None:
            weight = 1.0
            if hasattr(graph_data, 'edge_weights') and graph_data.edge_weights:
                weight = graph_data.edge_weights.get((edge[0], edge[1]), 1.0)
            G.add_edge(src_idx, tgt_idx, weight=weight)
    
    if k <= 0:
        return []
    
    paths_found = []
    try:
        for path in nx.shortest_simple_paths(G, source, target, weight='weight'):
            length = sum(G[path[i]][path[i+1]]['weight'] for i in range(len(path)-1))
            paths_found.append({
                'path': path,
                'length': length
            })
            if len(paths_found) >= k:
                break
    except nx.NetworkXNoPath:
        pass
    
    return paths_found

