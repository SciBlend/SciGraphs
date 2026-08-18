# Surface embeddings: planarity, genus, Euler characteristic, faces. Planarity,
# the Kuratowski subgraph, Chrobak-Payne drawing and spectral layout come from
# networkx; without it every entry point returns None or a failure dict.

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
                       "extra to enable planarity testing, genus and the "
                       "planar layouts")


def _networkx_missing(feature, empty=None):
    log(f"{feature} unavailable: {NETWORKX_REASON}")
    return empty


def _layout_unavailable():
    """Failure dict for the planar layouts. Reuses the existing 'error' slot so
    operators already reporting `layout_result['error']` need no change."""
    log(f"Planar layout unavailable: {NETWORKX_REASON}")
    return {
        'positions': None,
        'success': False,
        'error': NETWORKX_REASON
    }


def build_networkx_graph(graph_data, directed=False):
    """Build a NetworkX Graph or DiGraph from GraphData."""
    if not NETWORKX_AVAILABLE:
        return _networkx_missing("NetworkX graph construction")

    if directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()
    
    G.add_nodes_from(range(len(graph_data.nodes)))
    
    node_to_idx = graph_data.node_to_index
    
    for src, tgt in graph_data.edges:
        if src in node_to_idx and tgt in node_to_idx:
            G.add_edge(node_to_idx[src], node_to_idx[tgt])
    
    return G


def check_planarity_nx(graph_data):
    """Boyer-Myrvold planarity test. Returns ``(is_planar, embedding)``, the
    combinatorial embedding being None for a non-planar graph."""
    if not NETWORKX_AVAILABLE:
        return _networkx_missing("Planarity test", empty=(None, None))

    G = build_networkx_graph(graph_data)
    
    if G.number_of_nodes() == 0:
        return True, None
    
    is_planar, embedding = nx.check_planarity(G)
    
    return is_planar, embedding


def get_euler_characteristic(graph_data, embedding=None):
    """Euler characteristic chi = V - E + F, as a dict of V, E, F, chi and
    faces_computed. Without an ``embedding`` F is the planar estimate 2 - V + E
    and faces_computed is False, so chi comes back 2 whatever the graph is."""
    if not NETWORKX_AVAILABLE:
        return _networkx_missing("Euler characteristic")

    G = build_networkx_graph(graph_data)
    
    V = G.number_of_nodes()
    E = G.number_of_edges()
    
    if embedding is not None:
        faces = compute_faces_from_embedding(embedding)
        F = len(faces)
        faces_computed = True
    else:
        F = 2 - V + E
        faces_computed = False
    
    chi = V - E + F
    
    return {
        'V': V,
        'E': E,
        'F': F,
        'chi': chi,
        'faces_computed': faces_computed
    }


def compute_faces_from_embedding(embedding):
    """Every face of a planar embedding as a list of vertex indices. Neighbors
    are stored clockwise, so a face is traced by always turning right."""
    if embedding is None:
        return []
    
    faces = []
    visited_edges = set()
    
    for v in embedding.nodes():
        for w in embedding.neighbors_cw_order(v):
            if (v, w) in visited_edges:
                continue

            face = []
            current_v = v
            current_w = w
            
            while (current_v, current_w) not in visited_edges:
                visited_edges.add((current_v, current_w))
                face.append(current_v)
                
                # Turn right at current_w: the next neighbor clockwise from current_v.
                next_v = embedding.next_face_half_edge(current_v, current_w)[1]
                current_v = current_w
                current_w = next_v
            
            if len(face) > 0:
                faces.append(face)
    
    return faces


def compute_geometric_dual_3d(graph_data, positions, embedding=None):
    """Geometric 3D dual graph G*, after Mohar and Thomassen 2.6: dual vertices
    are face centroids of G, joined where their faces share an edge.
    ``positions`` is (N, 3); ``embedding`` is computed here when omitted, which
    requires a planar graph. Returns nodes, edges, positions and face_to_nodes."""

    if embedding is None:
        if not NETWORKX_AVAILABLE:
            log(f"Geometric dual unavailable: {NETWORKX_REASON}")
            return {
                'nodes': [],
                'edges': [],
                'positions': np.array([]),
                'face_to_nodes': {},
                'success': False,
                'error': NETWORKX_REASON
            }
        is_planar, embedding = check_planarity_nx(graph_data)
        if not is_planar:
            return {
                'nodes': [],
                'edges': [],
                'positions': np.array([]),
                'face_to_nodes': {},
                'success': False,
                'error': 'Graph is not planar - cannot compute dual'
            }
    
    faces = compute_faces_from_embedding(embedding)
    
    if not faces:
        return {
            'nodes': [],
            'edges': [],
            'positions': np.array([]),
            'face_to_nodes': {},
            'success': False,
            'error': 'No faces found in embedding'
        }
    
    num_faces = len(faces)
    dual_positions = np.zeros((num_faces, 3))
    
    edge_to_faces = {}  # edge (u, v) -> indices of the faces containing it

    for face_idx, face_nodes in enumerate(faces):
        face_coords = []
        for node_idx in face_nodes:
            if node_idx < len(positions):
                face_coords.append(positions[node_idx])
        
        if face_coords:
            centroid = np.mean(face_coords, axis=0)
            dual_positions[face_idx] = centroid
        else:
            dual_positions[face_idx] = np.array([0, 0, 0])

        n = len(face_nodes)
        for i in range(n):
            u = face_nodes[i]
            v = face_nodes[(i + 1) % n]
            # Smaller index first, so both traversal directions hash alike.
            edge_key = (min(u, v), max(u, v))
            
            if edge_key not in edge_to_faces:
                edge_to_faces[edge_key] = []
            edge_to_faces[edge_key].append(face_idx)
    
    dual_edges = []
    for edge, adjacent_faces in edge_to_faces.items():
        if len(adjacent_faces) == 2:
            dual_edges.append((adjacent_faces[0], adjacent_faces[1]))
        elif len(adjacent_faces) > 2:
            # Impossible on a simple planar graph; pair them all rather than drop.
            for i in range(len(adjacent_faces)):
                for j in range(i + 1, len(adjacent_faces)):
                    dual_edges.append((adjacent_faces[i], adjacent_faces[j]))

    dual_edges = list(set(dual_edges))

    face_to_nodes = {i: faces[i] for i in range(num_faces)}
    
    return {
        'nodes': list(range(num_faces)),
        'edges': dual_edges,
        'positions': dual_positions,
        'face_to_nodes': face_to_nodes,
        'num_faces': num_faces,
        'num_dual_edges': len(dual_edges),
        'success': True,
        'error': None
    }


def calculate_genus(graph_data):
    """Genus of the minimal surface embedding the graph without crossings. Exact
    only for planar graphs, where it is 0; otherwise genus_exact is None and
    genus_lower_bound is ceil((E - 3V + 6) / 6) from Euler's formula."""
    if not NETWORKX_AVAILABLE:
        return _networkx_missing("Genus")

    G = build_networkx_graph(graph_data)
    V = G.number_of_nodes()
    E = G.number_of_edges()
    
    is_planar, embedding = check_planarity_nx(graph_data)
    
    if is_planar:
        euler_data = get_euler_characteristic(graph_data, embedding)
        return {
            'is_planar': True,
            'genus_lower_bound': 0,
            'genus_exact': 0,
            'euler_data': euler_data
        }
    
    # The bound holds for connected graphs.
    if V <= 2:
        genus_lower = 0
    else:
        genus_lower = max(0, int(np.ceil((E - 3 * V + 6) / 6)))

    euler_data = {
        'V': V,
        'E': E,
        'F': None,
        'chi': None,
        'faces_computed': False
    }
    
    return {
        'is_planar': False,
        'genus_lower_bound': genus_lower,
        'genus_exact': None,
        'euler_data': euler_data
    }


def detect_kuratowski_subgraph(graph_data):
    """Find a Kuratowski subgraph, a K5 or K3,3 subdivision, in a non-planar
    graph; by Kuratowski's theorem a graph is planar exactly when it has
    neither. Returns is_planar, kuratowski_type and subgraph_nodes."""
    if not NETWORKX_AVAILABLE:
        return _networkx_missing("Kuratowski subgraph")

    G = build_networkx_graph(graph_data)
    
    is_planar, _ = nx.check_planarity(G)
    
    if is_planar:
        return {
            'is_planar': True,
            'kuratowski_type': None,
            'subgraph_nodes': None
        }
    
    try:
        kuratowski = nx.algorithms.planarity.kuratowski_subgraph(G)

        nodes = list(kuratowski.nodes())
        edges = kuratowski.number_of_edges()

        # A K5 subdivision carries at least 10 edges, a K3,3 subdivision 9.
        if len(nodes) >= 5 and edges >= 10:
            k_type = 'K5'
        else:
            k_type = 'K3,3'
        
        return {
            'is_planar': False,
            'kuratowski_type': k_type,
            'subgraph_nodes': nodes
        }
    except Exception:
        return {
            'is_planar': False,
            'kuratowski_type': 'unknown',
            'subgraph_nodes': None
        }


def get_face_node_assignments(graph_data, embedding=None):
    """Assign each node the lowest id among the faces containing it; nodes in
    no face keep -1. Returns node_face_ids, faces and num_faces. The embedding
    is computed here when not supplied, which needs a planar graph."""

    if embedding is None:
        if not NETWORKX_AVAILABLE:
            log(f"Face assignment unavailable: {NETWORKX_REASON}")
            return {
                'node_face_ids': None,
                'faces': None,
                'num_faces': 0,
                'error': NETWORKX_REASON
            }
        is_planar, embedding = check_planarity_nx(graph_data)
        if not is_planar:
            return {
                'node_face_ids': None,
                'faces': None,
                'num_faces': 0,
                'error': 'Graph is not planar'
            }
    
    faces = compute_faces_from_embedding(embedding)
    num_nodes = len(graph_data.nodes)
    
    node_face_ids = [-1] * num_nodes

    for face_id, face in enumerate(faces):
        for node in face:
            if node < num_nodes and node_face_ids[node] == -1:
                node_face_ids[node] = face_id
    
    return {
        'node_face_ids': node_face_ids,
        'faces': faces,
        'num_faces': len(faces)
    }


def compute_graph_connectivity(graph_data):
    """Return is_connected, num_components and vertex/edge connectivity. The
    two connectivity numbers are 0 for a disconnected or single-node graph,
    where they are undefined."""
    if not NETWORKX_AVAILABLE:
        return _networkx_missing("Connectivity metrics")

    G = build_networkx_graph(graph_data)
    
    if G.number_of_nodes() == 0:
        return {
            'is_connected': True,
            'num_components': 0,
            'vertex_connectivity': 0,
            'edge_connectivity': 0
        }
    
    is_connected = nx.is_connected(G)
    num_components = nx.number_connected_components(G)
    
    if is_connected and G.number_of_nodes() > 1:
        try:
            vertex_conn = nx.node_connectivity(G)
            edge_conn = nx.edge_connectivity(G)
        except Exception:
            vertex_conn = 0
            edge_conn = 0
    else:
        vertex_conn = 0
        edge_conn = 0
    
    return {
        'is_connected': is_connected,
        'num_components': num_components,
        'vertex_connectivity': vertex_conn,
        'edge_connectivity': edge_conn
    }


def compute_planar_layout(graph_data, scale=5.0):
    """Crossing-free layout via Chrobak-Payne straight-line drawing. Returns
    positions (an (N, 3) array with Z=0), success and error. Only planar graphs
    can be drawn this way; anything else fails with a message."""
    if not NETWORKX_AVAILABLE:
        return _layout_unavailable()

    G = build_networkx_graph(graph_data)
    
    is_planar, embedding = nx.check_planarity(G)
    
    if not is_planar:
        return {
            'positions': None,
            'success': False,
            'error': 'Graph is not planar - cannot compute crossing-free layout'
        }
    
    try:
        pos_dict = nx.planar_layout(G, scale=scale)

        num_nodes = len(graph_data.nodes)
        positions = np.zeros((num_nodes, 3))
        
        for node_idx, (x, y) in pos_dict.items():
            if node_idx < num_nodes:
                positions[node_idx] = [x, y, 0.0]
        
        return {
            'positions': positions,
            'success': True,
            'error': None
        }
    except Exception as e:
        return {
            'positions': None,
            'success': False,
            'error': str(e)
        }


def compute_tutte_layout(graph_data, scale=5.0):
    """Barycentric layout approximating a Tutte embedding; needs 3 nodes. Not a
    real Tutte embedding: that needs an outer face pinned to a convex polygon,
    which nothing here picks, so this substitutes a 2D spectral layout, reports
    ``method: 'spectral_2d'``, and may cross where Tutte would not."""
    if not NETWORKX_AVAILABLE:
        return _layout_unavailable()

    G = build_networkx_graph(graph_data)
    
    if G.number_of_nodes() < 3:
        return {
            'positions': None,
            'success': False,
            'error': 'Graph needs at least 3 nodes for Tutte embedding'
        }
    
    try:
        pos_dict = nx.spectral_layout(G, scale=scale, dim=2)
        
        num_nodes = len(graph_data.nodes)
        positions = np.zeros((num_nodes, 3))
        
        for node_idx, coords in pos_dict.items():
            if node_idx < num_nodes:
                positions[node_idx] = [coords[0], coords[1], 0.0]
        
        return {
            'positions': positions,
            'success': True,
            'error': None,
            'method': 'spectral_2d'
        }
    except Exception as e:
        return {
            'positions': None,
            'success': False,
            'error': str(e)
        }


def detect_edge_crossings_3d(vertices, edges, tolerance=1e-6):
    """Find edges that cross in 3D by testing every pair of segments, quadratic
    in the edge count. ``tolerance`` is the distance in scene units below which
    two segments cross; edges sharing a vertex are skipped."""
    crossing_pairs = []
    
    num_edges = len(edges)
    
    for i in range(num_edges):
        e1_start, e1_end = edges[i]
        p1 = np.array(vertices[e1_start])
        p2 = np.array(vertices[e1_end])
        
        for j in range(i + 1, num_edges):
            e2_start, e2_end = edges[j]
            
            if e1_start in (e2_start, e2_end) or e1_end in (e2_start, e2_end):
                continue

            p3 = np.array(vertices[e2_start])
            p4 = np.array(vertices[e2_end])

            if _segments_intersect_3d(p1, p2, p3, p4, tolerance):
                crossing_pairs.append(((e1_start, e1_end), (e2_start, e2_end)))
    
    return {
        'has_crossings': len(crossing_pairs) > 0,
        'num_crossings': len(crossing_pairs),
        'crossing_pairs': crossing_pairs
    }


def _segments_intersect_3d(p1, p2, p3, p4, tolerance=1e-6):
    """True when two 3D segments pass within ``tolerance``, from the closest
    point on each infinite line. Parallel lines always return False, so a
    genuine overlap along the same line is missed."""
    d1 = p2 - p1
    d2 = p4 - p3
    d3 = p1 - p3
    
    a = np.dot(d1, d1)
    b = np.dot(d1, d2)
    c = np.dot(d2, d2)
    d = np.dot(d1, d3)
    e = np.dot(d2, d3)
    
    denom = a * c - b * b
    
    if abs(denom) < 1e-10:
        return False

    s = (b * e - c * d) / denom
    t = (a * e - b * d) / denom

    if s < 0 or s > 1 or t < 0 or t > 1:
        return False

    closest1 = p1 + s * d1
    closest2 = p3 + t * d2

    distance = np.linalg.norm(closest1 - closest2)
    
    return distance < tolerance
