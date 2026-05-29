# Network flow algorithms

import networkx as nx
import numpy as np


def maximum_flow_ford_fulkerson(graph_data, source, sink):
    """
    Compute maximum flow using Ford-Fulkerson algorithm.
    
    Args:
        graph_data: GraphData object
        source: Source node index
        sink: Sink node index
    
    Returns:
        dict with 'max_flow', 'flow_dict', 'flow_per_edge'
    """
    G = nx.DiGraph()
    
    # Add nodes
    for i, node in enumerate(graph_data.nodes):
        G.add_node(i)
    
    # Add edges with capacity
    for edge in graph_data.edges:
        src_idx = graph_data.node_to_index.get(edge[0], None)
        tgt_idx = graph_data.node_to_index.get(edge[1], None)
        if src_idx is not None and tgt_idx is not None:
            capacity = 1.0
            if hasattr(graph_data, 'edge_weights') and graph_data.edge_weights:
                capacity = graph_data.edge_weights.get((edge[0], edge[1]), 1.0)
            G.add_edge(src_idx, tgt_idx, capacity=capacity)
    
    # Compute max flow
    try:
        flow_value, flow_dict = nx.maximum_flow(G, source, sink, capacity='capacity')
        
        # Create edge flow array
        num_edges = len(graph_data.edges)
        edge_flow = np.zeros(num_edges, dtype=np.float32)
        
        for i, edge in enumerate(graph_data.edges):
            src_idx = graph_data.node_to_index.get(edge[0])
            tgt_idx = graph_data.node_to_index.get(edge[1])
            if src_idx in flow_dict and tgt_idx in flow_dict[src_idx]:
                edge_flow[i] = flow_dict[src_idx][tgt_idx]
        
        return {
            'max_flow': flow_value,
            'flow_dict': flow_dict,
            'edge_flow': edge_flow
        }
    except nx.NetworkXError:
        return {
            'max_flow': 0.0,
            'flow_dict': {},
            'edge_flow': np.zeros(len(graph_data.edges))
        }


def minimum_cut(graph_data, source, sink):
    """
    Find minimum cut in flow network.
    
    Args:
        graph_data: GraphData object
        source: Source node index
        sink: Sink node index
    
    Returns:
        dict with 'cut_value', 'partition', 'cut_edges'
    """
    G = nx.DiGraph()
    
    # Add nodes
    for i, node in enumerate(graph_data.nodes):
        G.add_node(i)
    
    # Add edges with capacity
    for edge in graph_data.edges:
        src_idx = graph_data.node_to_index.get(edge[0], None)
        tgt_idx = graph_data.node_to_index.get(edge[1], None)
        if src_idx is not None and tgt_idx is not None:
            capacity = 1.0
            if hasattr(graph_data, 'edge_weights') and graph_data.edge_weights:
                capacity = graph_data.edge_weights.get((edge[0], edge[1]), 1.0)
            G.add_edge(src_idx, tgt_idx, capacity=capacity)
    
    try:
        cut_value, partition = nx.minimum_cut(G, source, sink, capacity='capacity')
        
        # Find cut edges
        reachable, non_reachable = partition
        cut_edges = []
        
        for u in reachable:
            for v in non_reachable:
                if G.has_edge(u, v):
                    cut_edges.append((u, v))
        
        # Create binary array for edges in cut
        num_edges = len(graph_data.edges)
        edge_in_cut = np.zeros(num_edges, dtype=np.float32)
        
        for i, edge in enumerate(graph_data.edges):
            src_idx = graph_data.node_to_index.get(edge[0])
            tgt_idx = graph_data.node_to_index.get(edge[1])
            if (src_idx, tgt_idx) in cut_edges:
                edge_in_cut[i] = 1.0
        
        return {
            'cut_value': cut_value,
            'reachable': list(reachable),
            'non_reachable': list(non_reachable),
            'cut_edges': cut_edges,
            'edge_in_cut': edge_in_cut
        }
    except nx.NetworkXError:
        return {
            'cut_value': 0.0,
            'reachable': [],
            'non_reachable': [],
            'cut_edges': [],
            'edge_in_cut': np.zeros(len(graph_data.edges))
        }


def min_cost_flow(graph_data, demands):
    """
    Compute minimum cost flow.
    
    Args:
        graph_data: GraphData object
        demands: Dict mapping node indices to demand (positive=sink, negative=source)
    
    Returns:
        dict with 'cost', 'flow_dict'
    """
    G = nx.DiGraph()
    
    # Add nodes with demand
    for i, node in enumerate(graph_data.nodes):
        demand = demands.get(i, 0)
        G.add_node(i, demand=demand)
    
    # Add edges with capacity and cost
    for edge in graph_data.edges:
        src_idx = graph_data.node_to_index.get(edge[0], None)
        tgt_idx = graph_data.node_to_index.get(edge[1], None)
        if src_idx is not None and tgt_idx is not None:
            capacity = 1.0
            cost = 1.0
            if hasattr(graph_data, 'edge_weights') and graph_data.edge_weights:
                # Use weight as cost
                cost = graph_data.edge_weights.get((edge[0], edge[1]), 1.0)
            G.add_edge(src_idx, tgt_idx, capacity=capacity, weight=cost)
    
    try:
        flow_dict = nx.min_cost_flow(G)
        flow_cost = nx.cost_of_flow(G, flow_dict)
        
        return {
            'cost': flow_cost,
            'flow_dict': flow_dict
        }
    except (nx.NetworkXError, nx.NetworkXUnfeasible):
        return {
            'cost': float('inf'),
            'flow_dict': {}
        }


def edge_connectivity(graph_data):
    """
    Compute edge connectivity (minimum number of edges to disconnect graph).
    
    Args:
        graph_data: GraphData object
    
    Returns:
        int: edge connectivity
    """
    G = nx.Graph()
    
    # Add nodes
    for i, node in enumerate(graph_data.nodes):
        G.add_node(i)
    
    # Add edges
    for edge in graph_data.edges:
        src_idx = graph_data.node_to_index.get(edge[0], None)
        tgt_idx = graph_data.node_to_index.get(edge[1], None)
        if src_idx is not None and tgt_idx is not None:
            G.add_edge(src_idx, tgt_idx)
    
    try:
        return nx.edge_connectivity(G)
    except nx.NetworkXError:
        return 0


def node_connectivity(graph_data):
    """
    Compute node connectivity (minimum number of nodes to disconnect graph).
    
    Args:
        graph_data: GraphData object
    
    Returns:
        int: node connectivity
    """
    G = nx.Graph()
    
    # Add nodes
    for i, node in enumerate(graph_data.nodes):
        G.add_node(i)
    
    # Add edges
    for edge in graph_data.edges:
        src_idx = graph_data.node_to_index.get(edge[0], None)
        tgt_idx = graph_data.node_to_index.get(edge[1], None)
        if src_idx is not None and tgt_idx is not None:
            G.add_edge(src_idx, tgt_idx)
    
    try:
        return nx.node_connectivity(G)
    except nx.NetworkXError:
        return 0

