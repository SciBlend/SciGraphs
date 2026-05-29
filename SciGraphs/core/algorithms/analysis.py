import numpy as np
import networkx as nx

def calculate_centrality(graph_data, method='degree'):
    """
    Calculates node centrality using various methods.
    Returns a list of centrality values for each node.
    """
    G = nx.Graph()
    G.add_nodes_from(range(len(graph_data.nodes)))
    
    node_to_idx = {node: i for i, node in enumerate(graph_data.nodes)}
    
    edge_indices = []
    for src, tgt in graph_data.edges:
        if src in node_to_idx and tgt in node_to_idx:
            edge_indices.append((node_to_idx[src], node_to_idx[tgt]))
    
    G.add_edges_from(edge_indices)
    
    if method == 'degree':
        centrality = nx.degree_centrality(G)
    elif method == 'betweenness':
        centrality = nx.betweenness_centrality(G)
    elif method == 'closeness':
        centrality = nx.closeness_centrality(G)
    elif method == 'eigenvector':
        try:
            centrality = nx.eigenvector_centrality(G, max_iter=1000)
        except nx.PowerIterationFailedConvergence:
            try:
                centrality = nx.eigenvector_centrality_numpy(G)
            except Exception:
                print("Eigenvector centrality failed to converge; falling back to degree centrality.")
                centrality = nx.degree_centrality(G)
    else:
        centrality = {i: 0.0 for i in range(len(graph_data.nodes))}
    
    return [centrality[i] for i in range(len(graph_data.nodes))]

def calculate_clustering(graph_data):
    """
    Calculates clustering coefficient for each node.
    """
    G = nx.Graph()
    G.add_nodes_from(range(len(graph_data.nodes)))
    
    node_to_idx = {node: i for i, node in enumerate(graph_data.nodes)}
    
    edge_indices = []
    for src, tgt in graph_data.edges:
        if src in node_to_idx and tgt in node_to_idx:
            edge_indices.append((node_to_idx[src], node_to_idx[tgt]))
    
    G.add_edges_from(edge_indices)
    
    clustering = nx.clustering(G)
    
    return [clustering[i] for i in range(len(graph_data.nodes))]

def _build_edge_list(graph_data):
    """
    Build an integer edge list and index mappings from graph_data.
    Returns (edges_int, node_to_idx, idx_to_node).
    """
    node_to_idx = {node: i for i, node in enumerate(graph_data.nodes)}
    idx_to_node = {i: node for node, i in node_to_idx.items()}
    edges_int = []
    for src, tgt in graph_data.edges:
        if src in node_to_idx and tgt in node_to_idx:
            si, ti = node_to_idx[src], node_to_idx[tgt]
            if si != ti:
                edges_int.append((si, ti))
    return edges_int, node_to_idx, idx_to_node


def _pysurprise_available():
    """Check whether the pysurprise package can be imported."""
    try:
        import pysurprise  # noqa: F401
        return True
    except ImportError:
        return False


def compute_surprise(graph_data, partition_list):
    """
    Compute the Surprise quality metric for a given partition.
    Returns the Surprise value (float) or 0.0 if pysurprise is unavailable.
    """
    if not _pysurprise_available():
        return 0.0
    import pysurprise
    edges_int, _, _ = _build_edge_list(graph_data)
    if not edges_int or len(set(partition_list)) < 2:
        return 0.0
    try:
        return pysurprise.surprise(edges_int, partition_list)
    except Exception as e:
        print(f"Surprise computation failed: {e}")
        return 0.0


_BIN_PERMISSIONS_FIXED = False


def _ensure_pysurprise_bin_permissions(bin_dir):
    """Ensure pySurprise compiled binaries have execute permission.

    Blender's extension installer extracts wheel contents without preserving
    the UNIX execute bit, so the SurpriseMe binaries need ``chmod +x`` at
    runtime the first time they are used.
    """
    global _BIN_PERMISSIONS_FIXED
    if _BIN_PERMISSIONS_FIXED:
        return
    import os, stat
    if not bin_dir.is_dir():
        return
    for entry in bin_dir.iterdir():
        if entry.is_file():
            mode = entry.stat().st_mode
            if not (mode & stat.S_IXUSR):
                entry.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    _BIN_PERMISSIONS_FIXED = True


def detect_communities(graph_data, algorithm='rn'):
    """
    Detects communities using a pySurprise algorithm.
    Falls back to networkx greedy_modularity if pysurprise is unavailable.
    Returns a list of community IDs for each node.
    """
    num_nodes = len(graph_data.nodes)
    edges_int, node_to_idx, idx_to_node = _build_edge_list(graph_data)

    if not edges_int:
        return [0] * num_nodes

    if _pysurprise_available():
        try:
            from pysurprise import algorithms as ps_algo
            _ensure_pysurprise_bin_permissions(ps_algo._PKG_BIN_DIR)
            algo_map = {
                'cpm': ps_algo.cpm,
                'infomap': ps_algo.infomap,
                'rb': ps_algo.rb,
                'rn': ps_algo.rn,
                'rnsc': ps_algo.rnsc,
                'scluster': ps_algo.scluster,
                'uvcluster': ps_algo.uvcluster,
            }
            func = algo_map.get(algorithm)
            if func is None:
                func = ps_algo.rn

            str_edges = [(str(u), str(v)) for u, v in edges_int]
            partition_dict = func(str_edges, timeout=300)

            if not partition_dict:
                raise RuntimeError(f"{algorithm} returned empty partition")

            cluster_ids = [0] * num_nodes
            for str_node, comm_id in partition_dict.items():
                idx = int(str_node)
                if 0 <= idx < num_nodes:
                    cluster_ids[idx] = comm_id

            # Remap arbitrary community IDs to contiguous 0..N-1
            unique_ids = sorted(set(cluster_ids))
            remap = {old: new for new, old in enumerate(unique_ids)}
            cluster_ids = [remap[c] for c in cluster_ids]
            return cluster_ids
        except Exception as e:
            print(f"pySurprise {algorithm} failed: {e}; falling back to networkx")

    # Fallback: networkx greedy_modularity
    try:
        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))
        G.add_edges_from(edges_int)
        from networkx.algorithms import community
        communities = community.greedy_modularity_communities(G)
        node_community = {}
        for comm_id, comm in enumerate(communities):
            for node in comm:
                node_community[node] = comm_id
        return [node_community.get(i, 0) for i in range(num_nodes)]
    except Exception as e:
        print(f"Community detection failed: {e}")
        return [0] * num_nodes

def calculate_shortest_paths(graph_data, source_idx=0):
    """
    Calculates shortest path lengths from a source node to all other nodes.
    """
    G = nx.Graph()
    G.add_nodes_from(range(len(graph_data.nodes)))
    
    node_to_idx = {node: i for i, node in enumerate(graph_data.nodes)}
    
    edge_indices = []
    for src, tgt in graph_data.edges:
        if src in node_to_idx and tgt in node_to_idx:
            edge_indices.append((node_to_idx[src], node_to_idx[tgt]))
    
    G.add_edges_from(edge_indices)
    
    try:
        lengths = nx.single_source_shortest_path_length(G, source_idx)
        max_length = max(lengths.values()) if lengths else 1
        
        return [lengths.get(i, max_length + 1) for i in range(len(graph_data.nodes))]
    except Exception as e:
        print(f"Shortest path calculation failed: {e}")
        return [0] * len(graph_data.nodes)

def apply_advanced_clustering(graph_data, algorithm='rn', resolution=1.0, seed=0, threshold=1e-7):
    """
    Applies community detection via pySurprise algorithms (CPM, Infomap, RB,
    RN, RNSC, SCluster, UVCluster) and computes the Surprise quality metric.
    Returns dict with: cluster_ids, cluster_sizes, surprise, modularity,
                       clustering_coefficients, num_clusters
    """
    num_nodes = len(graph_data.nodes)
    edges_int, node_to_idx, idx_to_node = _build_edge_list(graph_data)

    # Detect communities
    cluster_ids = detect_communities(graph_data, algorithm=algorithm)

    # Build networkx graph for modularity and clustering coefficient
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from(edges_int)

    # Compute Surprise
    surprise_val = compute_surprise(graph_data, cluster_ids)

    # Compute modularity via networkx
    communities_sets = {}
    for node_idx, cid in enumerate(cluster_ids):
        communities_sets.setdefault(cid, set()).add(node_idx)
    communities_list = list(communities_sets.values())

    if len(communities_list) > 0 and len(G.edges()) > 0:
        modularity = nx.algorithms.community.modularity(G, communities_list, resolution=resolution)
    else:
        modularity = 0.0

    # Cluster sizes
    cluster_sizes_dict = {}
    for cid in cluster_ids:
        cluster_sizes_dict[cid] = cluster_sizes_dict.get(cid, 0) + 1
    cluster_sizes = [cluster_sizes_dict[cid] for cid in cluster_ids]

    # Local clustering coefficient
    local_clustering = nx.clustering(G)
    clustering_coefficients = [local_clustering.get(i, 0.0) for i in range(num_nodes)]

    return {
        'cluster_ids': cluster_ids,
        'cluster_sizes': cluster_sizes,
        'surprise': surprise_val,
        'modularity': modularity,
        'clustering_coefficients': clustering_coefficients,
        'num_clusters': len(communities_list)
    }

def calculate_directed_centrality(graph_data, method='pagerank'):
    """
    Calculates centrality metrics specific to directed graphs.
    Returns a list of centrality values for each node.
    """
    G = nx.DiGraph()
    G.add_nodes_from(range(len(graph_data.nodes)))
    
    node_to_idx = {node: i for i, node in enumerate(graph_data.nodes)}
    
    edge_indices = []
    for src, tgt in graph_data.edges:
        if src in node_to_idx and tgt in node_to_idx:
            edge_indices.append((node_to_idx[src], node_to_idx[tgt]))
    
    G.add_edges_from(edge_indices)
    
    centrality = {}
    
    if method == 'pagerank':
        # PageRank - web-style importance
        centrality = nx.pagerank(G, alpha=0.85, max_iter=1000)
    elif method == 'hub_score':
        # HITS algorithm - Hub scores (good pointers to authorities)
        try:
            hits = nx.hits(G, max_iter=1000)
            centrality = hits[0]  # hub scores
        except:
            # Fallback if HITS doesn't converge
            centrality = {i: 0.0 for i in range(len(graph_data.nodes))}
    elif method == 'authority_score':
        # HITS algorithm - Authority scores (good destinations)
        try:
            hits = nx.hits(G, max_iter=1000)
            centrality = hits[1]  # authority scores
        except:
            centrality = {i: 0.0 for i in range(len(graph_data.nodes))}
    elif method == 'in_degree':
        # In-degree centrality (popularity - how many point to this node)
        in_deg = dict(G.in_degree())
        max_in = max(in_deg.values()) if in_deg.values() else 1
        centrality = {k: v / max_in for k, v in in_deg.items()}
    elif method == 'out_degree':
        # Out-degree centrality (influence - how many this node points to)
        out_deg = dict(G.out_degree())
        max_out = max(out_deg.values()) if out_deg.values() else 1
        centrality = {k: v / max_out for k, v in out_deg.items()}
    elif method == 'katz':
        # Katz centrality - variant of eigenvector for directed graphs
        try:
            centrality = nx.katz_centrality(G, max_iter=1000)
        except:
            # Fallback if doesn't converge
            centrality = {i: 0.0 for i in range(len(graph_data.nodes))}
    else:
        centrality = {i: 0.0 for i in range(len(graph_data.nodes))}
    
    return [centrality.get(i, 0.0) for i in range(len(graph_data.nodes))]

def detect_graph_patterns(graph_data):
    """
    Detects structural patterns in directed graphs.
    Returns a dictionary with pattern information.
    """
    G = nx.DiGraph()
    G.add_nodes_from(range(len(graph_data.nodes)))
    
    node_to_idx = {node: i for i, node in enumerate(graph_data.nodes)}
    
    edge_indices = []
    for src, tgt in graph_data.edges:
        if src in node_to_idx and tgt in node_to_idx:
            edge_indices.append((node_to_idx[src], node_to_idx[tgt]))
    
    G.add_edges_from(edge_indices)
    
    # Detect various patterns
    patterns = {
        'is_dag': nx.is_directed_acyclic_graph(G),
        'is_tree': nx.is_tree(G),
        'is_forest': nx.is_forest(G),
        'is_strongly_connected': nx.is_strongly_connected(G),
        'is_weakly_connected': nx.is_weakly_connected(G),
        'num_strongly_connected_components': nx.number_strongly_connected_components(G),
        'num_weakly_connected_components': nx.number_weakly_connected_components(G),
    }
    
    # Count cycles (limit to avoid performance issues)
    try:
        cycles = list(nx.simple_cycles(G))
        patterns['num_cycles'] = len(cycles[:1000])  # Limit to 1000
        patterns['has_cycles'] = len(cycles) > 0
    except:
        patterns['num_cycles'] = 0
        patterns['has_cycles'] = False
    
    return patterns

def analyze_flow_structure(graph_data):
    """
    Analyzes flow patterns: sources, sinks, and bottlenecks.
    Returns dictionary with flow information and node classifications.
    """
    G = nx.DiGraph()
    G.add_nodes_from(range(len(graph_data.nodes)))
    
    node_to_idx = {node: i for i, node in enumerate(graph_data.nodes)}
    
    edge_indices = []
    for src, tgt in graph_data.edges:
        if src in node_to_idx and tgt in node_to_idx:
            edge_indices.append((node_to_idx[src], node_to_idx[tgt]))
    
    G.add_edges_from(edge_indices)
    
    # Identify sources (no incoming edges)
    sources = [n for n in G.nodes() if G.in_degree(n) == 0]
    
    # Identify sinks (no outgoing edges)
    sinks = [n for n in G.nodes() if G.out_degree(n) == 0]
    
    # Identify intermediaries (both in and out edges)
    intermediaries = [n for n in G.nodes() if G.in_degree(n) > 0 and G.out_degree(n) > 0]
    
    # Calculate betweenness to find bottlenecks
    betweenness = nx.betweenness_centrality(G)
    
    # Find top bottlenecks (high betweenness)
    sorted_betweenness = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)
    top_bottlenecks = [node for node, score in sorted_betweenness[:10] if score > 0]
    
    # Create node classification array
    node_types = []
    for i in range(len(graph_data.nodes)):
        if i in sources:
            node_types.append(1)  # Source
        elif i in sinks:
            node_types.append(2)  # Sink
        elif i in top_bottlenecks:
            node_types.append(3)  # Bottleneck
        elif i in intermediaries:
            node_types.append(4)  # Intermediary
        else:
            node_types.append(0)  # Isolated
    
    return {
        'sources': sources,
        'sinks': sinks,
        'intermediaries': intermediaries,
        'bottlenecks': top_bottlenecks,
        'node_types': node_types,
        'betweenness': [betweenness.get(i, 0.0) for i in range(len(graph_data.nodes))],
        'num_sources': len(sources),
        'num_sinks': len(sinks),
        'num_intermediaries': len(intermediaries),
    }

def find_strongly_connected_components(graph_data):
    """
    Finds strongly connected components (SCCs) in the directed graph.
    Returns component IDs for each node.
    """
    G = nx.DiGraph()
    G.add_nodes_from(range(len(graph_data.nodes)))
    
    node_to_idx = {node: i for i, node in enumerate(graph_data.nodes)}
    
    edge_indices = []
    for src, tgt in graph_data.edges:
        if src in node_to_idx and tgt in node_to_idx:
            edge_indices.append((node_to_idx[src], node_to_idx[tgt]))
    
    G.add_edges_from(edge_indices)
    
    # Find SCCs
    sccs = list(nx.strongly_connected_components(G))
    
    # Assign component ID to each node
    component_ids = [0] * len(graph_data.nodes)
    for comp_id, component in enumerate(sccs):
        for node in component:
            component_ids[node] = comp_id
    
    # Calculate component sizes
    component_sizes = [len(comp) for comp in sccs]
    
    return {
        'component_ids': component_ids,
        'component_sizes': component_sizes,
        'num_components': len(sccs),
        'largest_component_size': max(component_sizes) if component_sizes else 0,
    }


def calculate_flow_distances(graph_data):
    """
    Calculate the minimum distance from any source node for flow animation.
    Returns distances (in propagation steps) for each node.
    Sources get distance 0, their neighbors get 1, etc.
    """
    G = nx.DiGraph()
    
    # Add nodes
    for i, node in enumerate(graph_data.nodes):
        G.add_node(i)
    
    # Add edges
    for source, target in graph_data.edges:
        source_idx = graph_data.node_to_index.get(source)
        target_idx = graph_data.node_to_index.get(target)
        if source_idx is not None and target_idx is not None:
            G.add_edge(source_idx, target_idx)
    
    # Identify source nodes (no incoming edges)
    sources = [node for node in G.nodes() if G.in_degree(node) == 0]
    
    if not sources:
        # If no pure sources, use nodes with highest out-degree
        out_degrees = [(node, G.out_degree(node)) for node in G.nodes()]
        out_degrees.sort(key=lambda x: x[1], reverse=True)
        num_sources = max(1, len(G.nodes()) // 10)
        sources = [node for node, _ in out_degrees[:num_sources]]
    
    # BFS from all sources simultaneously to find minimum distance
    distances = {node: float('inf') for node in G.nodes()}
    queue = []
    
    # Initialize sources
    for source in sources:
        distances[source] = 0
        queue.append((source, 0))
    
    # BFS
    visited = set()
    while queue:
        node, dist = queue.pop(0)
        if node in visited:
            continue
        visited.add(node)
        
        # Visit all successors (nodes that this node points to)
        for neighbor in G.successors(node):
            if distances[neighbor] > dist + 1:
                distances[neighbor] = dist + 1
                queue.append((neighbor, dist + 1))
    
    # Convert to list in node order
    result = [distances.get(i, -1) for i in range(len(graph_data.nodes))]
    
    # Normalize unreachable nodes to max distance + 1
    max_dist = max([d for d in result if d >= 0], default=0)
    result = [d if d >= 0 else max_dist + 1 for d in result]
    
    return {
        'distances': result,
        'max_distance': max_dist,
        'sources': sources,
    }


def calculate_bfs_traversal(graph_data, start_nodes=None, is_directed=False):
    """
    Perform Breadth-First Search traversal from specified start nodes.
    Returns visit order, depth, and parent information for each node.
    
    Args:
        graph_data: GraphData object containing nodes and edges
        start_nodes: List of node indices to start from (None = auto-select)
        is_directed: Whether to treat graph as directed
    
    Returns:
        dict with 'order', 'depth', 'parent', 'max_order', 'visited_count'
    """
    if is_directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()
    
    # Add nodes
    for i, node in enumerate(graph_data.nodes):
        G.add_node(i)
    
    # Add edges
    for source, target in graph_data.edges:
        source_idx = graph_data.node_to_index.get(source)
        target_idx = graph_data.node_to_index.get(target)
        if source_idx is not None and target_idx is not None:
            G.add_edge(source_idx, target_idx)
    
    # Determine start nodes
    if start_nodes is None or len(start_nodes) == 0:
        # Auto-select: use node(s) with highest degree
        degrees = [(node, G.degree(node)) for node in G.nodes()]
        if degrees:
            degrees.sort(key=lambda x: x[1], reverse=True)
            start_nodes = [degrees[0][0]]
        else:
            start_nodes = [0] if len(graph_data.nodes) > 0 else []
    
    # Initialize tracking
    order = {node: -1 for node in G.nodes()}
    depth = {node: -1 for node in G.nodes()}
    parent = {node: -1 for node in G.nodes()}
    
    # BFS from all start nodes
    queue = []
    current_order = 0
    
    for start in start_nodes:
        if start in G.nodes() and order[start] == -1:
            queue.append((start, 0, -1))
            order[start] = current_order
            depth[start] = 0
            parent[start] = -1
            current_order += 1
    
    # Process queue
    while queue:
        node, node_depth, node_parent = queue.pop(0)
        
        # Visit all neighbors
        for neighbor in G.neighbors(node):
            if order[neighbor] == -1:
                order[neighbor] = current_order
                depth[neighbor] = node_depth + 1
                parent[neighbor] = node
                queue.append((neighbor, node_depth + 1, node))
                current_order += 1
    
    # Convert to lists
    num_nodes = len(graph_data.nodes)
    order_list = [order.get(i, -1) for i in range(num_nodes)]
    depth_list = [depth.get(i, -1) for i in range(num_nodes)]
    parent_list = [parent.get(i, -1) for i in range(num_nodes)]
    
    # Calculate max order (excluding unvisited)
    visited_orders = [o for o in order_list if o >= 0]
    max_order = max(visited_orders) if visited_orders else 0
    visited_count = len(visited_orders)
    
    return {
        'order': order_list,
        'depth': depth_list,
        'parent': parent_list,
        'max_order': max_order,
        'visited_count': visited_count,
        'start_nodes': start_nodes,
    }


def calculate_dfs_traversal(graph_data, start_nodes=None, is_directed=False):
    """
    Perform Depth-First Search traversal from specified start nodes.
    Returns visit order, depth, and parent information for each node.
    
    Args:
        graph_data: GraphData object containing nodes and edges
        start_nodes: List of node indices to start from (None = auto-select)
        is_directed: Whether to treat graph as directed
    
    Returns:
        dict with 'order', 'depth', 'parent', 'max_order', 'visited_count'
    """
    if is_directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()
    
    # Add nodes
    for i, node in enumerate(graph_data.nodes):
        G.add_node(i)
    
    # Add edges
    for source, target in graph_data.edges:
        source_idx = graph_data.node_to_index.get(source)
        target_idx = graph_data.node_to_index.get(target)
        if source_idx is not None and target_idx is not None:
            G.add_edge(source_idx, target_idx)
    
    # Determine start nodes
    if start_nodes is None or len(start_nodes) == 0:
        # Auto-select: use node(s) with highest degree
        degrees = [(node, G.degree(node)) for node in G.nodes()]
        if degrees:
            degrees.sort(key=lambda x: x[1], reverse=True)
            start_nodes = [degrees[0][0]]
        else:
            start_nodes = [0] if len(graph_data.nodes) > 0 else []
    
    # Initialize tracking
    order = {node: -1 for node in G.nodes()}
    depth = {node: -1 for node in G.nodes()}
    parent = {node: -1 for node in G.nodes()}
    
    # DFS helper function
    current_order = [0]  # Use list to allow modification in nested function
    
    def dfs_visit(node, node_depth, node_parent):
        order[node] = current_order[0]
        depth[node] = node_depth
        parent[node] = node_parent
        current_order[0] += 1
        
        # Visit all neighbors
        for neighbor in G.neighbors(node):
            if order[neighbor] == -1:
                dfs_visit(neighbor, node_depth + 1, node)
    
    # Start DFS from all start nodes
    for start in start_nodes:
        if start in G.nodes() and order[start] == -1:
            dfs_visit(start, 0, -1)
    
    # Convert to lists
    num_nodes = len(graph_data.nodes)
    order_list = [order.get(i, -1) for i in range(num_nodes)]
    depth_list = [depth.get(i, -1) for i in range(num_nodes)]
    parent_list = [parent.get(i, -1) for i in range(num_nodes)]
    
    # Calculate max order (excluding unvisited)
    visited_orders = [o for o in order_list if o >= 0]
    max_order = max(visited_orders) if visited_orders else 0
    visited_count = len(visited_orders)
    
    return {
        'order': order_list,
        'depth': depth_list,
        'parent': parent_list,
        'max_order': max_order,
        'visited_count': visited_count,
        'start_nodes': start_nodes,
    }

