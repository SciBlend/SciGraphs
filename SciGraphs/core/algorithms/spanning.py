# Spanning tree algorithms

import networkx as nx
import numpy as np


def minimum_spanning_tree_kruskal(graph_data):
    """
    Compute Minimum Spanning Tree using Kruskal's algorithm.
    
    Args:
        graph_data: GraphData object
    
    Returns:
        dict with 'edges', 'total_weight', 'edge_indices'
    """
    G = nx.Graph()  # MST only for undirected graphs
    
    # Add nodes
    for i, node in enumerate(graph_data.nodes):
        G.add_node(i)
    
    # Add edges with weights
    for edge in graph_data.edges:
        src_idx = graph_data.node_to_index.get(edge[0], None)
        tgt_idx = graph_data.node_to_index.get(edge[1], None)
        if src_idx is not None and tgt_idx is not None:
            weight = 1.0
            if hasattr(graph_data, 'edge_weights') and graph_data.edge_weights:
                weight = graph_data.edge_weights.get((edge[0], edge[1]), 1.0)
            G.add_edge(src_idx, tgt_idx, weight=weight)
    
    # Compute MST
    mst = nx.minimum_spanning_tree(G, algorithm='kruskal')
    
    # Extract edges and total weight
    mst_edges = list(mst.edges())
    total_weight = sum(G[u][v]['weight'] for u, v in mst_edges)
    
    # Create binary array indicating which edges are in MST
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
    """
    Compute Minimum Spanning Tree using Prim's algorithm.
    
    Args:
        graph_data: GraphData object
        start_node: Node to start from
    
    Returns:
        dict with 'edges', 'total_weight', 'edge_indices'
    """
    G = nx.Graph()
    
    # Add nodes
    for i, node in enumerate(graph_data.nodes):
        G.add_node(i)
    
    # Add edges with weights
    for edge in graph_data.edges:
        src_idx = graph_data.node_to_index.get(edge[0], None)
        tgt_idx = graph_data.node_to_index.get(edge[1], None)
        if src_idx is not None and tgt_idx is not None:
            weight = 1.0
            if hasattr(graph_data, 'edge_weights') and graph_data.edge_weights:
                weight = graph_data.edge_weights.get((edge[0], edge[1]), 1.0)
            G.add_edge(src_idx, tgt_idx, weight=weight)
    
    # Compute MST using Prim's
    mst = nx.minimum_spanning_tree(G, algorithm='prim')
    
    # Extract edges and total weight
    mst_edges = list(mst.edges())
    total_weight = sum(G[u][v]['weight'] for u, v in mst_edges)
    
    # Create binary array
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
    """
    Compute Maximum Spanning Tree.
    
    Args:
        graph_data: GraphData object
    
    Returns:
        dict with 'edges', 'total_weight', 'edge_indices'
    """
    G = nx.Graph()
    
    # Add nodes
    for i, node in enumerate(graph_data.nodes):
        G.add_node(i)
    
    # Add edges with weights
    for edge in graph_data.edges:
        src_idx = graph_data.node_to_index.get(edge[0], None)
        tgt_idx = graph_data.node_to_index.get(edge[1], None)
        if src_idx is not None and tgt_idx is not None:
            weight = 1.0
            if hasattr(graph_data, 'edge_weights') and graph_data.edge_weights:
                weight = graph_data.edge_weights.get((edge[0], edge[1]), 1.0)
            G.add_edge(src_idx, tgt_idx, weight=weight)
    
    # Compute maximum spanning tree
    mst = nx.maximum_spanning_tree(G)
    
    # Extract edges and total weight
    mst_edges = list(mst.edges())
    total_weight = sum(G[u][v]['weight'] for u, v in mst_edges)
    
    # Create binary array
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
    """
    Compute approximate Steiner tree for a set of terminal nodes.
    
    Args:
        graph_data: GraphData object
        terminal_nodes: List of node indices that must be connected
    
    Returns:
        dict with 'edges', 'total_weight'
    """
    G = nx.Graph()
    
    # Add nodes
    for i, node in enumerate(graph_data.nodes):
        G.add_node(i)
    
    # Add edges with weights
    for edge in graph_data.edges:
        src_idx = graph_data.node_to_index.get(edge[0], None)
        tgt_idx = graph_data.node_to_index.get(edge[1], None)
        if src_idx is not None and tgt_idx is not None:
            weight = 1.0
            if hasattr(graph_data, 'edge_weights') and graph_data.edge_weights:
                weight = graph_data.edge_weights.get((edge[0], edge[1]), 1.0)
            G.add_edge(src_idx, tgt_idx, weight=weight)
    
    # Compute approximate Steiner tree
    steiner = nx.approximation.steiner_tree(G, terminal_nodes, weight='weight')
    
    # Extract edges and total weight
    steiner_edges = list(steiner.edges())
    total_weight = sum(G[u][v]['weight'] for u, v in steiner_edges)
    
    return {
        'edges': steiner_edges,
        'total_weight': total_weight,
        'num_edges': len(steiner_edges)
    }

