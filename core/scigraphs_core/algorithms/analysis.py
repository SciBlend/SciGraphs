import contextlib
import random

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
                       "extra to enable centrality, clustering, traversal and "
                       "flow analysis")

# Fixed seed for the randomized community algorithms; see _seeded_igraph.
COMMUNITY_SEED = 20240517


def _networkx_missing(feature, empty=None):
    """Log why `feature` did not run and return `empty`, its no-result value.
    Pass a zero-filled list of the right length wherever the caller writes a
    mesh attribute, since a wrong-length write corrupts the mesh."""
    log(f"{feature} unavailable: {NETWORKX_REASON}")
    return empty

def calculate_centrality(graph_data, method='degree'):
    """One centrality value per node by degree, betweenness, closeness or
    eigenvector; eigenvector falls back to the numpy solver, then to degree."""
    if not NETWORKX_AVAILABLE:
        return _networkx_missing("Centrality",
                                 empty=[0.0] * len(graph_data.nodes))

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
    """Return the local clustering coefficient per node, zeros without networkx."""
    if not NETWORKX_AVAILABLE:
        return _networkx_missing("Clustering coefficient",
                                 empty=[0.0] * len(graph_data.nodes))

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
    """Return (edges_int, node_to_idx, idx_to_node); self-loops are dropped."""
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
    try:
        import pysurprise  # noqa: F401
        return True
    except ImportError:
        return False


def compute_surprise(graph_data, partition_list):
    """Return the Surprise quality metric of a partition, 0.0 without pysurprise."""
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
    """Chmod +x the pySurprise binaries once per session: Blender's extension
    installer unpacks wheels without preserving the UNIX execute bit."""
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


@contextlib.contextmanager
def _seeded_igraph(seed):
    """Pin igraph's random source so community detection is reproducible. It
    wants a generator object rather than a seed, so this installs a private
    ``Random`` and restores the default after. No-op without igraph."""
    try:
        import igraph
    except Exception:
        yield
        return
    try:
        igraph.set_random_number_generator(random.Random(seed))
        yield
    finally:
        igraph.set_random_number_generator(random)


def communities_from_edges(edges_int, num_nodes, algorithm='rn', timeout=300,
                           seed=COMMUNITY_SEED):
    """A contiguous community id per node, or None if every backend failed.
    ``edges_int`` is (u, v) pairs with 0 <= u, v < num_nodes; kept separate from
    ``detect_communities`` so callers holding an edge list skip graph_data."""
    if not edges_int or num_nodes <= 0:
        return [0] * max(num_nodes, 0)

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
            func = algo_map.get(algorithm, ps_algo.rn)

            str_edges = [(str(u), str(v)) for u, v in edges_int]
            with _seeded_igraph(seed):
                partition_dict = func(str_edges, timeout=timeout)

            if not partition_dict:
                raise RuntimeError(f"{algorithm} returned empty partition")

            cluster_ids = [0] * num_nodes
            for str_node, comm_id in partition_dict.items():
                idx = int(str_node)
                if 0 <= idx < num_nodes:
                    cluster_ids[idx] = comm_id

            unique_ids = sorted(set(cluster_ids))
            remap = {old: new for new, old in enumerate(unique_ids)}
            return [remap[c] for c in cluster_ids]
        except Exception as e:
            print(f"pySurprise {algorithm} failed: {e}; falling back to networkx")

    if not NETWORKX_AVAILABLE:
        # Last backend: None is the documented "everything failed" answer.
        return _networkx_missing("NetworkX community detection")

    try:
        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))
        G.add_edges_from(edges_int)
        from networkx.algorithms import community
        found = community.greedy_modularity_communities(G)
        node_community = {}
        for comm_id, comm in enumerate(found):
            for node in comm:
                node_community[node] = comm_id
        return [node_community.get(i, 0) for i in range(num_nodes)]
    except Exception as e:
        print(f"networkx community detection failed: {e}")
        return None


def detect_communities(graph_data, algorithm='rn'):
    """Return a community id per node, via pySurprise or networkx greedy modularity."""
    num_nodes = len(graph_data.nodes)
    edges_int, node_to_idx, idx_to_node = _build_edge_list(graph_data)

    if not edges_int:
        return [0] * num_nodes

    result = communities_from_edges(edges_int, num_nodes, algorithm)
    return result if result is not None else [0] * num_nodes

def calculate_shortest_paths(graph_data, source_idx=0):
    """Hop counts from ``source_idx`` to every node, zeros without networkx.
    Unreachable nodes get the largest reachable distance plus one."""
    if not NETWORKX_AVAILABLE:
        return _networkx_missing("Shortest path lengths",
                                 empty=[0] * len(graph_data.nodes))

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
    """Detect communities (CPM, Infomap, RB, RN, RNSC, SCluster, UVCluster) and
    score them: cluster_ids, cluster_sizes, surprise, modularity,
    clustering_coefficients, num_clusters. None without networkx."""
    if not NETWORKX_AVAILABLE:
        return _networkx_missing("Advanced clustering")

    num_nodes = len(graph_data.nodes)
    edges_int, node_to_idx, idx_to_node = _build_edge_list(graph_data)

    cluster_ids = detect_communities(graph_data, algorithm=algorithm)

    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from(edges_int)

    surprise_val = compute_surprise(graph_data, cluster_ids)

    communities_sets = {}
    for node_idx, cid in enumerate(cluster_ids):
        communities_sets.setdefault(cid, set()).add(node_idx)
    communities_list = list(communities_sets.values())

    if len(communities_list) > 0 and len(G.edges()) > 0:
        modularity = nx.algorithms.community.modularity(G, communities_list, resolution=resolution)
    else:
        modularity = 0.0

    # Per-node, so every node carries the size of the cluster it landed in.
    cluster_sizes_dict = {}
    for cid in cluster_ids:
        cluster_sizes_dict[cid] = cluster_sizes_dict.get(cid, 0) + 1
    cluster_sizes = [cluster_sizes_dict[cid] for cid in cluster_ids]

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
    """One directed-graph centrality value per node, zeros without networkx.
    `method` is pagerank, hub_score, authority_score (HITS), in_degree,
    out_degree or katz; non-converging methods come back as zeros."""
    if not NETWORKX_AVAILABLE:
        return _networkx_missing("Directed centrality",
                                 empty=[0.0] * len(graph_data.nodes))

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
        centrality = nx.pagerank(G, alpha=0.85, max_iter=1000)
    elif method == 'hub_score':
        try:
            hits = nx.hits(G, max_iter=1000)
            centrality = hits[0]  # hub scores
        except:
            centrality = {i: 0.0 for i in range(len(graph_data.nodes))}
    elif method == 'authority_score':
        try:
            hits = nx.hits(G, max_iter=1000)
            centrality = hits[1]  # authority scores
        except:
            centrality = {i: 0.0 for i in range(len(graph_data.nodes))}
    elif method == 'in_degree':
        in_deg = dict(G.in_degree())
        max_in = max(in_deg.values()) if in_deg.values() else 1
        centrality = {k: v / max_in for k, v in in_deg.items()}
    elif method == 'out_degree':
        out_deg = dict(G.out_degree())
        max_out = max(out_deg.values()) if out_deg.values() else 1
        centrality = {k: v / max_out for k, v in out_deg.items()}
    elif method == 'katz':
        try:
            centrality = nx.katz_centrality(G, max_iter=1000)
        except:
            centrality = {i: 0.0 for i in range(len(graph_data.nodes))}
    else:
        centrality = {i: 0.0 for i in range(len(graph_data.nodes))}
    
    return [centrality.get(i, 0.0) for i in range(len(graph_data.nodes))]

def detect_graph_patterns(graph_data):
    """Structural flags for a directed graph: DAG, tree, connectivity, cycles.
    None without networkx, every entry being a networkx predicate."""
    if not NETWORKX_AVAILABLE:
        return _networkx_missing("Graph pattern detection")

    G = nx.DiGraph()
    G.add_nodes_from(range(len(graph_data.nodes)))
    
    node_to_idx = {node: i for i, node in enumerate(graph_data.nodes)}
    
    edge_indices = []
    for src, tgt in graph_data.edges:
        if src in node_to_idx and tgt in node_to_idx:
            edge_indices.append((node_to_idx[src], node_to_idx[tgt]))
    
    G.add_edges_from(edge_indices)
    
    patterns = {
        'is_dag': nx.is_directed_acyclic_graph(G),
        'is_tree': nx.is_tree(G),
        'is_forest': nx.is_forest(G),
        'is_strongly_connected': nx.is_strongly_connected(G),
        'is_weakly_connected': nx.is_weakly_connected(G),
        'num_strongly_connected_components': nx.number_strongly_connected_components(G),
        'num_weakly_connected_components': nx.number_weakly_connected_components(G),
    }
    
    try:
        cycles = list(nx.simple_cycles(G))
        patterns['num_cycles'] = len(cycles[:1000])  # capped, dense graphs blow up
        patterns['has_cycles'] = len(cycles) > 0
    except:
        patterns['num_cycles'] = 0
        patterns['has_cycles'] = False
    
    return patterns

def analyze_flow_structure(graph_data):
    """Classify nodes as sources, sinks, bottlenecks or intermediaries, with
    per-node ``node_types`` codes and betweenness. Bottlenecks are the ten
    highest betweenness scores above zero. None without networkx."""
    if not NETWORKX_AVAILABLE:
        return _networkx_missing("Flow structure analysis")

    G = nx.DiGraph()
    G.add_nodes_from(range(len(graph_data.nodes)))
    
    node_to_idx = {node: i for i, node in enumerate(graph_data.nodes)}
    
    edge_indices = []
    for src, tgt in graph_data.edges:
        if src in node_to_idx and tgt in node_to_idx:
            edge_indices.append((node_to_idx[src], node_to_idx[tgt]))
    
    G.add_edges_from(edge_indices)
    
    sources = [n for n in G.nodes() if G.in_degree(n) == 0]

    sinks = [n for n in G.nodes() if G.out_degree(n) == 0]

    intermediaries = [n for n in G.nodes() if G.in_degree(n) > 0 and G.out_degree(n) > 0]

    betweenness = nx.betweenness_centrality(G)

    sorted_betweenness = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)
    top_bottlenecks = [node for node, score in sorted_betweenness[:10] if score > 0]

    # Codes 1 source, 2 sink, 3 bottleneck, 4 intermediary: return contract,
    # do not renumber.
    node_types = []
    for i in range(len(graph_data.nodes)):
        if i in sources:
            node_types.append(1)
        elif i in sinks:
            node_types.append(2)
        elif i in top_bottlenecks:
            node_types.append(3)
        elif i in intermediaries:
            node_types.append(4)
        else:
            node_types.append(0)  # isolated
    
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
    """Find strongly connected components; returns ids, sizes and counts per node."""
    if not NETWORKX_AVAILABLE:
        return _networkx_missing("Strongly connected components",
                                 empty=[0] * len(graph_data.nodes))

    G = nx.DiGraph()
    G.add_nodes_from(range(len(graph_data.nodes)))
    
    node_to_idx = {node: i for i, node in enumerate(graph_data.nodes)}
    
    edge_indices = []
    for src, tgt in graph_data.edges:
        if src in node_to_idx and tgt in node_to_idx:
            edge_indices.append((node_to_idx[src], node_to_idx[tgt]))
    
    G.add_edges_from(edge_indices)
    
    sccs = list(nx.strongly_connected_components(G))

    component_ids = [0] * len(graph_data.nodes)
    for comp_id, component in enumerate(sccs):
        for node in component:
            component_ids[node] = comp_id
    
    component_sizes = [len(comp) for comp in sccs]
    
    return {
        'component_ids': component_ids,
        'component_sizes': component_sizes,
        'num_components': len(sccs),
        'largest_component_size': max(component_sizes) if component_sizes else 0,
    }


def calculate_flow_distances(graph_data):
    """Distance in propagation steps from the nearest source. Sources have no
    incoming edges and sit at 0; with no such node the top decile by out-degree
    stands in. Unreachable nodes get the largest distance plus one."""
    if not NETWORKX_AVAILABLE:
        return _networkx_missing("Flow distances",
                                 empty=[0] * len(graph_data.nodes))

    G = nx.DiGraph()

    for i, node in enumerate(graph_data.nodes):
        G.add_node(i)

    for source, target in graph_data.edges:
        source_idx = graph_data.node_to_index.get(source)
        target_idx = graph_data.node_to_index.get(target)
        if source_idx is not None and target_idx is not None:
            G.add_edge(source_idx, target_idx)

    sources = [node for node in G.nodes() if G.in_degree(node) == 0]

    if not sources:
        out_degrees = [(node, G.out_degree(node)) for node in G.nodes()]
        out_degrees.sort(key=lambda x: x[1], reverse=True)
        num_sources = max(1, len(G.nodes()) // 10)
        sources = [node for node, _ in out_degrees[:num_sources]]

    # Multi-source BFS: seed every source at 0 and the first visit wins.
    distances = {node: float('inf') for node in G.nodes()}
    queue = []

    for source in sources:
        distances[source] = 0
        queue.append((source, 0))

    visited = set()
    while queue:
        node, dist = queue.pop(0)
        if node in visited:
            continue
        visited.add(node)

        for neighbor in G.successors(node):
            if distances[neighbor] > dist + 1:
                distances[neighbor] = dist + 1
                queue.append((neighbor, dist + 1))

    result = [distances.get(i, -1) for i in range(len(graph_data.nodes))]

    max_dist = max([d for d in result if d >= 0], default=0)
    result = [d if d >= 0 else max_dist + 1 for d in result]
    
    return {
        'distances': result,
        'max_distance': max_dist,
        'sources': sources,
    }


def calculate_bfs_traversal(graph_data, start_nodes=None, is_directed=False):
    """Breadth-first traversal returning per-node order, depth and parent.
    ``start_nodes`` of None picks the highest-degree node; nodes never reached
    keep -1 in all three lists. None without networkx."""
    if not NETWORKX_AVAILABLE:
        return _networkx_missing("BFS traversal")

    if is_directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()
    
    for i, node in enumerate(graph_data.nodes):
        G.add_node(i)
    
    for source, target in graph_data.edges:
        source_idx = graph_data.node_to_index.get(source)
        target_idx = graph_data.node_to_index.get(target)
        if source_idx is not None and target_idx is not None:
            G.add_edge(source_idx, target_idx)
    
    if start_nodes is None or len(start_nodes) == 0:
        degrees =[(node, G.degree(node)) for node in G.nodes()]
        if degrees:
            degrees.sort(key=lambda x: x[1], reverse=True)
            start_nodes = [degrees[0][0]]
        else:
            start_nodes = [0] if len(graph_data.nodes) > 0 else []
    
    order = {node: -1 for node in G.nodes()}
    depth = {node: -1 for node in G.nodes()}
    parent = {node: -1 for node in G.nodes()}
    
    queue = []
    current_order = 0
    
    for start in start_nodes:
        if start in G.nodes() and order[start] == -1:
            queue.append((start, 0, -1))
            order[start] = current_order
            depth[start] = 0
            parent[start] = -1
            current_order += 1
    
    while queue:
        node, node_depth, node_parent = queue.pop(0)
        
        for neighbor in G.neighbors(node):
            if order[neighbor] == -1:
                order[neighbor] = current_order
                depth[neighbor] = node_depth + 1
                parent[neighbor] = node
                queue.append((neighbor, node_depth + 1, node))
                current_order += 1
    
    num_nodes = len(graph_data.nodes)
    order_list = [order.get(i, -1) for i in range(num_nodes)]
    depth_list = [depth.get(i, -1) for i in range(num_nodes)]
    parent_list = [parent.get(i, -1) for i in range(num_nodes)]
    
    visited_orders =[o for o in order_list if o >= 0]
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
    """Depth-first traversal returning per-node order, depth and parent.
    ``start_nodes`` of None picks the highest-degree node; nodes never reached
    keep -1 in all three lists. Recursive, so very deep graphs can hit the
    interpreter recursion limit."""
    if not NETWORKX_AVAILABLE:
        return _networkx_missing("DFS traversal")

    if is_directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()
    
    for i, node in enumerate(graph_data.nodes):
        G.add_node(i)
    
    for source, target in graph_data.edges:
        source_idx = graph_data.node_to_index.get(source)
        target_idx = graph_data.node_to_index.get(target)
        if source_idx is not None and target_idx is not None:
            G.add_edge(source_idx, target_idx)
    
    if start_nodes is None or len(start_nodes) == 0:
        degrees =[(node, G.degree(node)) for node in G.nodes()]
        if degrees:
            degrees.sort(key=lambda x: x[1], reverse=True)
            start_nodes = [degrees[0][0]]
        else:
            start_nodes = [0] if len(graph_data.nodes) > 0 else []
    
    order = {node: -1 for node in G.nodes()}
    depth = {node: -1 for node in G.nodes()}
    parent = {node: -1 for node in G.nodes()}
    
    current_order = [0]  # boxed so the nested dfs_visit can bump it
    
    def dfs_visit(node, node_depth, node_parent):
        order[node] = current_order[0]
        depth[node] = node_depth
        parent[node] = node_parent
        current_order[0] += 1
        
        for neighbor in G.neighbors(node):
            if order[neighbor] == -1:
                dfs_visit(neighbor, node_depth + 1, node)
    
    for start in start_nodes:
        if start in G.nodes() and order[start] == -1:
            dfs_visit(start, 0, -1)
    
    num_nodes = len(graph_data.nodes)
    order_list = [order.get(i, -1) for i in range(num_nodes)]
    depth_list = [depth.get(i, -1) for i in range(num_nodes)]
    parent_list = [parent.get(i, -1) for i in range(num_nodes)]
    
    visited_orders =[o for o in order_list if o >= 0]
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

