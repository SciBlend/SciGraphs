# Spanning tree algorithms. Without networkx each returns None, kept distinct
# from the empty tree an edgeless graph genuinely has.

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
                       "extra to enable the spanning tree algorithms")


def _networkx_missing(feature):
    log(f"{feature} unavailable: {NETWORKX_REASON}")
    return None


def minimum_spanning_tree_kruskal(graph_data):
    """Minimum spanning tree by Kruskal: 'edges', 'total_weight', 'num_edges'
    and 'edge_in_mst', a 0/1 array parallel to ``graph_data.edges``. Weights
    default to 1.0; a disconnected graph yields a forest, not a tree."""
    if not NETWORKX_AVAILABLE:
        return _networkx_missing("Minimum spanning tree (Kruskal)")

    G = nx.Graph()  # MST only for undirected graphs
    
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
    
    mst = nx.minimum_spanning_tree(G, algorithm='kruskal')

    mst_edges = list(mst.edges())
    total_weight = sum(G[u][v]['weight'] for u, v in mst_edges)
    
    num_edges = len(graph_data.edges)
    edge_in_mst = np.zeros(num_edges, dtype=np.float32)
    
    for i, edge in enumerate(graph_data.edges):
        src_idx = graph_data.node_to_index.get(edge[0])
        tgt_idx = graph_data.node_to_index.get(edge[1])
        if (src_idx, tgt_idx) in mst_edges or (tgt_idx, src_idx) in mst_edges:
            edge_in_mst[i] = 1.0
    
    return {
        'edges': mst_edges,
        'total_weight': total_weight,
        'edge_in_mst': edge_in_mst,
        'num_edges': len(mst_edges)
    }


def minimum_spanning_tree_prim(graph_data, start_node=0):
    """Minimum spanning tree by Prim, same return shape as
    :func:`minimum_spanning_tree_kruskal`. ``start_node`` is accepted for
    symmetry but not passed on, so distinct weights give Kruskal's tree."""
    if not NETWORKX_AVAILABLE:
        return _networkx_missing("Minimum spanning tree (Prim)")

    G = nx.Graph()

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

    mst = nx.minimum_spanning_tree(G, algorithm='prim')
    
    mst_edges = list(mst.edges())
    total_weight = sum(G[u][v]['weight'] for u, v in mst_edges)
    
    num_edges = len(graph_data.edges)
    edge_in_mst = np.zeros(num_edges, dtype=np.float32)
    
    for i, edge in enumerate(graph_data.edges):
        src_idx = graph_data.node_to_index.get(edge[0])
        tgt_idx = graph_data.node_to_index.get(edge[1])
        if (src_idx, tgt_idx) in mst_edges or (tgt_idx, src_idx) in mst_edges:
            edge_in_mst[i] = 1.0
    
    return {
        'edges': mst_edges,
        'total_weight': total_weight,
        'edge_in_mst': edge_in_mst,
        'num_edges': len(mst_edges)
    }


def maximum_spanning_tree(graph_data):
    """Maximum spanning tree, shaped like :func:`minimum_spanning_tree_kruskal`."""
    if not NETWORKX_AVAILABLE:
        return _networkx_missing("Maximum spanning tree")

    G = nx.Graph()
    
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
    
    mst = nx.maximum_spanning_tree(G)
    
    mst_edges = list(mst.edges())
    total_weight = sum(G[u][v]['weight'] for u, v in mst_edges)
    
    num_edges = len(graph_data.edges)
    edge_in_mst = np.zeros(num_edges, dtype=np.float32)
    
    for i, edge in enumerate(graph_data.edges):
        src_idx = graph_data.node_to_index.get(edge[0])
        tgt_idx = graph_data.node_to_index.get(edge[1])
        if (src_idx, tgt_idx) in mst_edges or (tgt_idx, src_idx) in mst_edges:
            edge_in_mst[i] = 1.0
    
    return {
        'edges': mst_edges,
        'total_weight': total_weight,
        'edge_in_mst': edge_in_mst,
        'num_edges': len(mst_edges)
    }


def steiner_tree(graph_data, terminal_nodes):
    """Approximate the smallest tree connecting every node in ``terminal_nodes``,
    as 'edges', 'total_weight' and 'num_edges'. Exact Steiner trees are NP-hard,
    so this is networkx's approximation and the weight is an upper bound."""
    if not NETWORKX_AVAILABLE:
        return _networkx_missing("Steiner tree")

    G = nx.Graph()
    
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
    
    steiner = nx.approximation.steiner_tree(G, terminal_nodes, weight='weight')

    steiner_edges = list(steiner.edges())
    total_weight = sum(G[u][v]['weight'] for u, v in steiner_edges)
    
    return {
        'edges': steiner_edges,
        'total_weight': total_weight,
        'num_edges': len(steiner_edges)
    }

