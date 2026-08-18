# Network flow algorithms. Without networkx each returns None, kept distinct
# from a genuine zero flow or empty cut.

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
                       "extra to enable the network flow algorithms")


def _networkx_missing(feature):
    log(f"{feature} unavailable: {NETWORKX_REASON}")
    return None


def maximum_flow_ford_fulkerson(graph_data, source, sink):
    """Maximum flow from ``source`` to ``sink`` by Ford-Fulkerson: 'max_flow',
    'flow_dict' and 'edge_flow' parallel to ``graph_data.edges``. Capacities are
    edge weights, default 1.0."""
    if not NETWORKX_AVAILABLE:
        return _networkx_missing("Maximum flow (Ford-Fulkerson)")

    G = nx.DiGraph()
    
    for i, node in enumerate(graph_data.nodes):
        G.add_node(i)
    
    for edge in graph_data.edges:
        src_idx = graph_data.node_to_index.get(edge[0], None)
        tgt_idx = graph_data.node_to_index.get(edge[1], None)
        if src_idx is not None and tgt_idx is not None:
            capacity = 1.0
            if hasattr(graph_data, 'edge_weights') and graph_data.edge_weights:
                capacity = graph_data.edge_weights.get((edge[0], edge[1]), 1.0)
            G.add_edge(src_idx, tgt_idx, capacity=capacity)
    
    try:
        flow_value, flow_dict = nx.maximum_flow(G, source, sink, capacity='capacity')

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
    """Minimum cut separating ``source`` from ``sink``: 'cut_value', the
    'reachable' and 'non_reachable' node lists, 'cut_edges', and 'edge_in_cut',
    a 0/1 array parallel to ``graph_data.edges``."""
    if not NETWORKX_AVAILABLE:
        return _networkx_missing("Minimum cut")

    G = nx.DiGraph()
    
    for i, node in enumerate(graph_data.nodes):
        G.add_node(i)
    
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
        
        reachable, non_reachable = partition
        cut_edges = []

        for u in reachable:
            for v in non_reachable:
                if G.has_edge(u, v):
                    cut_edges.append((u, v))

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
    """Minimum cost flow satisfying ``demands``, a node index to demand map,
    positive for a sink and negative for a source, omitted nodes at 0. Edge
    weights are cost, every capacity 1.0. Returns 'cost' (inf when the demands
    cannot be met) and 'flow_dict'."""
    if not NETWORKX_AVAILABLE:
        return _networkx_missing("Minimum cost flow")

    G = nx.DiGraph()
    
    for i, node in enumerate(graph_data.nodes):
        demand = demands.get(i, 0)
        G.add_node(i, demand=demand)

    for edge in graph_data.edges:
        src_idx = graph_data.node_to_index.get(edge[0], None)
        tgt_idx = graph_data.node_to_index.get(edge[1], None)
        if src_idx is not None and tgt_idx is not None:
            capacity = 1.0
            cost = 1.0
            if hasattr(graph_data, 'edge_weights') and graph_data.edge_weights:
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
    """The fewest edges whose removal disconnects the graph."""
    if not NETWORKX_AVAILABLE:
        return _networkx_missing("Edge connectivity")

    G = nx.Graph()
    
    for i, node in enumerate(graph_data.nodes):
        G.add_node(i)
    
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
    """The fewest nodes whose removal disconnects the graph."""
    if not NETWORKX_AVAILABLE:
        return _networkx_missing("Node connectivity")

    G = nx.Graph()
    
    for i, node in enumerate(graph_data.nodes):
        G.add_node(i)
    
    for edge in graph_data.edges:
        src_idx = graph_data.node_to_index.get(edge[0], None)
        tgt_idx = graph_data.node_to_index.get(edge[1], None)
        if src_idx is not None and tgt_idx is not None:
            G.add_edge(src_idx, tgt_idx)
    
    try:
        return nx.node_connectivity(G)
    except nx.NetworkXError:
        return 0

