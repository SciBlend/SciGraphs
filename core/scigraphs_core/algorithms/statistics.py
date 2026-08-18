# Global graph statistics and metrics

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
                       "extra to enable the graph statistics")

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
    SCIPY_REASON = None
except ImportError:
    stats = None
    SCIPY_AVAILABLE = False
    SCIPY_REASON = ("scipy is not installed; install the 'scipy' extra to "
                    "enable the power-law fit of the degree distribution")


def _networkx_missing(feature):
    """Log why `feature` did not run and return None.

    None rather than 0.0, because functions below already return 0.0 for a real
    measurement of zero density or assortativity. Sharing the value would let a
    caption read "clustering: 0.00" for a library that was never installed.
    """
    log(f"{feature} unavailable: {NETWORKX_REASON}")
    return None


def calculate_degree_distribution(graph_data):
    """Return mean, median, std, min, max, the degrees and a histogram of them.

    At most 20 histogram bins. None without networkx.
    """
    if not NETWORKX_AVAILABLE:
        return _networkx_missing("Degree distribution")

    G = nx.Graph()
    
    for i, node in enumerate(graph_data.nodes):
        G.add_node(i)
    
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
    
    mean_degree = np.mean(degrees)
    median_degree = np.median(degrees)
    std_degree = np.std(degrees)
    min_degree = np.min(degrees)
    max_degree = np.max(degrees)
    
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
    """Return the global clustering coefficient (transitivity); None without networkx."""
    if not NETWORKX_AVAILABLE:
        return _networkx_missing("Global clustering coefficient")

    G = nx.Graph()
    
    for i, node in enumerate(graph_data.nodes):
        G.add_node(i)
    
    for edge in graph_data.edges:
        src_idx = graph_data.node_to_index.get(edge[0], None)
        tgt_idx = graph_data.node_to_index.get(edge[1], None)
        if src_idx is not None and tgt_idx is not None:
            G.add_edge(src_idx, tgt_idx)
    
    return nx.transitivity(G)


def calculate_density(graph_data):
    """Return graph density from 0 to 1; None without networkx."""
    if not NETWORKX_AVAILABLE:
        return _networkx_missing("Density")

    G = nx.Graph()
    
    for i, node in enumerate(graph_data.nodes):
        G.add_node(i)
    
    for edge in graph_data.edges:
        src_idx = graph_data.node_to_index.get(edge[0], None)
        tgt_idx = graph_data.node_to_index.get(edge[1], None)
        if src_idx is not None and tgt_idx is not None:
            G.add_edge(src_idx, tgt_idx)
    
    return nx.density(G)


def calculate_diameter(graph_data):
    """Return the diameter, the longest shortest path; None without networkx.

    A disconnected graph is measured on its largest component. inf when the
    graph has fewer than two nodes, no edges, or that component is a single
    node.
    """
    if not NETWORKX_AVAILABLE:
        return _networkx_missing("Diameter")

    G = nx.Graph()
    
    for i, node in enumerate(graph_data.nodes):
        G.add_node(i)
    
    for edge in graph_data.edges:
        src_idx = graph_data.node_to_index.get(edge[0], None)
        tgt_idx = graph_data.node_to_index.get(edge[1], None)
        if src_idx is not None and tgt_idx is not None:
            G.add_edge(src_idx, tgt_idx)
    
    if len(G.nodes()) <= 1:
        return float('inf')

    if len(G.edges()) == 0:
        return float('inf')
    
    try:
        if nx.is_connected(G):
            return nx.diameter(G)
        else:
            largest_cc = max(nx.connected_components(G), key=len)
            subgraph = G.subgraph(largest_cc)
            if len(subgraph) <= 1:
                return float('inf')
            return nx.diameter(subgraph)
    except (nx.NetworkXError, nx.NetworkXPointlessConcept):
        return float('inf')


def calculate_average_path_length(graph_data):
    """Return the average shortest path length; None without networkx.

    A disconnected graph is measured on its largest component. inf when the
    graph has fewer than two nodes, no edges, or that component is a single
    node.
    """
    if not NETWORKX_AVAILABLE:
        return _networkx_missing("Average path length")

    G = nx.Graph()
    
    for i, node in enumerate(graph_data.nodes):
        G.add_node(i)
    
    for edge in graph_data.edges:
        src_idx = graph_data.node_to_index.get(edge[0], None)
        tgt_idx = graph_data.node_to_index.get(edge[1], None)
        if src_idx is not None and tgt_idx is not None:
            G.add_edge(src_idx, tgt_idx)
    
    if len(G.nodes()) <= 1:
        return float('inf')

    if len(G.edges()) == 0:
        return float('inf')
    
    try:
        if nx.is_connected(G):
            return nx.average_shortest_path_length(G)
        else:
            largest_cc = max(nx.connected_components(G), key=len)
            subgraph = G.subgraph(largest_cc)
            if len(subgraph) <= 1:
                return float('inf')
            return nx.average_shortest_path_length(subgraph)
    except (nx.NetworkXError, nx.NetworkXPointlessConcept):
        return float('inf')


def calculate_assortativity(graph_data):
    """Return the degree assortativity coefficient, -1 to 1; None without networkx.

    0.0 stands in for an empty graph and for the NaN networkx returns when the
    degree variance is zero.
    """
    if not NETWORKX_AVAILABLE:
        return _networkx_missing("Degree assortativity")

    G = nx.Graph()
    
    for i, node in enumerate(graph_data.nodes):
        G.add_node(i)
    
    for edge in graph_data.edges:
        src_idx = graph_data.node_to_index.get(edge[0], None)
        tgt_idx = graph_data.node_to_index.get(edge[1], None)
        if src_idx is not None and tgt_idx is not None:
            G.add_edge(src_idx, tgt_idx)
    
    if len(G.nodes()) == 0 or len(G.edges()) == 0:
        return 0.0

    try:
        result = nx.degree_assortativity_coefficient(G)
        if result is None or np.isnan(result):
            return 0.0
        return result
    except (nx.NetworkXError, ZeroDivisionError):
        return 0.0


def calculate_all_statistics(graph_data):
    """Run every statistic in this module and return them in one dict.

    Rebuilds the networkx graph once per metric, so this is slower than calling
    the one metric you need.
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
    """Fit a power law to a degree sequence by least squares on the log-log counts.

    Returns 'alpha' (the exponent) and 'fit_quality' (R-squared). `alpha` is
    None below 10 nonzero degrees and when scipy is missing, which is logged.
    """
    if not SCIPY_AVAILABLE:
        log(f"Power-law fit unavailable: {SCIPY_REASON}")
        return {
            'alpha': None,
            'fit_quality': 0.0
        }

    if not degrees or len(degrees) < 10:
        return {
            'alpha': None,
            'fit_quality': 0.0
        }
    
    # log(0) is undefined, so isolated nodes cannot take part in the fit.
    degrees_nonzero = [d for d in degrees if d > 0]
    
    if len(degrees_nonzero) < 10:
        return {
            'alpha': None,
            'fit_quality': 0.0
        }
    
    unique_degrees = sorted(set(degrees_nonzero))
    degree_count = {d: degrees_nonzero.count(d) for d in unique_degrees}
    
    x = np.array(unique_degrees)
    y = np.array([degree_count[d] for d in unique_degrees])
    
    log_x = np.log(x)
    log_y = np.log(y)
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_x, log_y)
    
    return {
        'alpha': -slope,  # negated: the fitted slope runs downward
        'fit_quality': r_value**2
    }

