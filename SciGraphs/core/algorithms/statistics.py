# Global graph statistics and metrics

import networkx as nx
import numpy as np
from scipy import stats


def calculate_degree_distribution(graph_data):
    """
    Calculate degree distribution statistics.
    
    Args:
        graph_data: GraphData object
    
    Returns:
        dict with distribution stats
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
    
    degrees = [d for n, d in G.degree()]
    
    if not degrees:
        return {
            'mean': 0.0,
            'median': 0.0,
            'std': 0.0,
            'min': 0,
            'max': 0,
            'histogram': [],
            'bins': []
        }
    
    # Calculate statistics
    mean_degree = np.mean(degrees)
    median_degree = np.median(degrees)
    std_degree = np.std(degrees)
    min_degree = np.min(degrees)
    max_degree = np.max(degrees)
    
    # Create histogram
    if max_degree > 0:
        num_bins = min(20, max_degree + 1)
        hist, bins = np.histogram(degrees, bins=num_bins)
    else:
        hist, bins = np.array([]), np.array([])
    
    return {
        'mean': float(mean_degree),
        'median': float(median_degree),
        'std': float(std_degree),
        'min': int(min_degree),
        'max': int(max_degree),
        'degrees': degrees,
        'histogram': hist.tolist(),
        'bins': bins.tolist()
    }


def calculate_global_clustering(graph_data):
    """
    Calculate global clustering coefficient (transitivity).
    
    Args:
        graph_data: GraphData object
    
    Returns:
        float: global clustering coefficient
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
    
    return nx.transitivity(G)


def calculate_density(graph_data):
    """
    Calculate graph density.
    
    Args:
        graph_data: GraphData object
    
    Returns:
        float: density (0 to 1)
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
    
    return nx.density(G)


def calculate_diameter(graph_data):
    """
    Calculate graph diameter (longest shortest path).
    
    Args:
        graph_data: GraphData object
    
    Returns:
        int: diameter (or inf if disconnected)
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
    
    # Handle empty graph or single node
    if len(G.nodes()) <= 1:
        return float('inf')
    
    # Handle graph with no edges
    if len(G.edges()) == 0:
        return float('inf')
    
    try:
        if nx.is_connected(G):
            return nx.diameter(G)
        else:
            # For disconnected graph, return diameter of largest component
            largest_cc = max(nx.connected_components(G), key=len)
            subgraph = G.subgraph(largest_cc)
            if len(subgraph) <= 1:
                return float('inf')
            return nx.diameter(subgraph)
    except (nx.NetworkXError, nx.NetworkXPointlessConcept):
        return float('inf')


def calculate_average_path_length(graph_data):
    """
    Calculate average shortest path length.
    
    Args:
        graph_data: GraphData object
    
    Returns:
        float: average path length
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
    
    # Handle empty graph or single node
    if len(G.nodes()) <= 1:
        return float('inf')
    
    # Handle graph with no edges
    if len(G.edges()) == 0:
        return float('inf')
    
    try:
        if nx.is_connected(G):
            return nx.average_shortest_path_length(G)
        else:
            # For disconnected graph, average over largest component
            largest_cc = max(nx.connected_components(G), key=len)
            subgraph = G.subgraph(largest_cc)
            if len(subgraph) <= 1:
                return float('inf')
            return nx.average_shortest_path_length(subgraph)
    except (nx.NetworkXError, nx.NetworkXPointlessConcept):
        return float('inf')


def calculate_assortativity(graph_data):
    """
    Calculate degree assortativity coefficient.
    
    Args:
        graph_data: GraphData object
    
    Returns:
        float: assortativity coefficient (-1 to 1)
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
    
    # Handle special cases
    if len(G.nodes()) == 0 or len(G.edges()) == 0:
        return 0.0
    
    try:
        result = nx.degree_assortativity_coefficient(G)
        # Handle NaN (returned when variance is zero or graph is too simple)
        if result is None or np.isnan(result):
            return 0.0
        return result
    except (nx.NetworkXError, ZeroDivisionError):
        return 0.0


def calculate_all_statistics(graph_data):
    """
    Calculate all graph statistics at once.
    
    Args:
        graph_data: GraphData object
    
    Returns:
        dict with all statistics
    """
    return {
        'num_nodes': len(graph_data.nodes),
        'num_edges': len(graph_data.edges),
        'density': calculate_density(graph_data),
        'degree_distribution': calculate_degree_distribution(graph_data),
        'global_clustering': calculate_global_clustering(graph_data),
        'diameter': calculate_diameter(graph_data),
        'average_path_length': calculate_average_path_length(graph_data),
        'assortativity': calculate_assortativity(graph_data)
    }


def power_law_fit(degrees):
    """
    Fit power law distribution to degree sequence.
    
    Args:
        degrees: List of node degrees
    
    Returns:
        dict with 'alpha' (exponent) and 'fit_quality'
    """
    if not degrees or len(degrees) < 10:
        return {
            'alpha': None,
            'fit_quality': 0.0
        }
    
    # Filter out zero degrees for log-log fit
    degrees_nonzero = [d for d in degrees if d > 0]
    
    if len(degrees_nonzero) < 10:
        return {
            'alpha': None,
            'fit_quality': 0.0
        }
    
    # Create histogram
    unique_degrees = sorted(set(degrees_nonzero))
    degree_count = {d: degrees_nonzero.count(d) for d in unique_degrees}
    
    x = np.array(unique_degrees)
    y = np.array([degree_count[d] for d in unique_degrees])
    
    # Log-log fit
    log_x = np.log(x)
    log_y = np.log(y)
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_x, log_y)
    
    return {
        'alpha': -slope,  # Power law exponent
        'fit_quality': r_value**2  # R-squared
    }

