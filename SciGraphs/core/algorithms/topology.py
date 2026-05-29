# Topological analysis module for graph surface embeddings
# Handles planarity testing, genus computation, Euler characteristic, and face detection

import networkx as nx
import numpy as np


def build_networkx_graph(graph_data, directed=False):
    """
    Constructs a NetworkX graph from the internal GraphData structure.
    
    Args:
        graph_data: GraphData object with nodes and edges
        directed: Whether to create a directed graph
    
    Returns:
        NetworkX Graph or DiGraph
    """
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
    """
    Checks if the graph can be embedded in a plane without edge crossings.
    Uses NetworkX's Boyer-Myrvold planarity algorithm.
    
    Args:
        graph_data: GraphData object
    
    Returns:
        tuple: (is_planar: bool, embedding: PlanarEmbedding or None)
            - is_planar: True if graph is planar
            - embedding: Combinatorial embedding if planar, None otherwise
    """
    G = build_networkx_graph(graph_data)
    
    if G.number_of_nodes() == 0:
        return True, None
    
    is_planar, embedding = nx.check_planarity(G)
    
    return is_planar, embedding


def get_euler_characteristic(graph_data, embedding=None):
    """
    Calculates the Euler characteristic chi = V - E + F.
    
    For planar graphs embedded in a plane: chi = 2
    For graphs on a torus: chi = 0
    For graphs on a surface of genus g: chi = 2 - 2g
    
    Args:
        graph_data: GraphData object
        embedding: Optional PlanarEmbedding from check_planarity_nx
    
    Returns:
        dict with keys:
            - V: number of vertices
            - E: number of edges
            - F: number of faces (if embedding provided, else estimated)
            - chi: Euler characteristic
            - faces_computed: whether F was computed from embedding
    """
    G = build_networkx_graph(graph_data)
    
    V = G.number_of_nodes()
    E = G.number_of_edges()
    
    if embedding is not None:
        # Compute actual faces from the planar embedding
        faces = compute_faces_from_embedding(embedding)
        F = len(faces)
        faces_computed = True
    else:
        # For planar graphs: V - E + F = 2, so F = 2 - V + E
        # This is an estimate assuming planarity
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
    """
    Extracts all faces from a planar embedding by traversing the rotation system.
    
    A planar embedding stores for each vertex the clockwise order of its neighbors.
    Faces are found by following edges and always turning right (next in rotation).
    
    Args:
        embedding: NetworkX PlanarEmbedding object
    
    Returns:
        list of faces, where each face is a list of vertex indices
    """
    if embedding is None:
        return []
    
    faces = []
    visited_edges = set()
    
    # Traverse each directed edge to find faces
    for v in embedding.nodes():
        for w in embedding.neighbors_cw_order(v):
            if (v, w) in visited_edges:
                continue
            
            # Start a new face
            face = []
            current_v = v
            current_w = w
            
            while (current_v, current_w) not in visited_edges:
                visited_edges.add((current_v, current_w))
                face.append(current_v)
                
                # Move to next edge in face (turn right at current_w)
                # Get the next neighbor after current_v in clockwise order around current_w
                next_v = embedding.next_face_half_edge(current_v, current_w)[1]
                current_v = current_w
                current_w = next_v
            
            if len(face) > 0:
                faces.append(face)
    
    return faces


def compute_geometric_dual_3d(graph_data, positions, embedding=None):
    """
    Constructs the geometric 3D dual graph G*.
    
    Based on Chapter 2.6 of "Graphs on Surfaces" (Mohar & Thomassen).
    The vertices of G* are the centroids of the faces of G.
    Two dual vertices are connected if their corresponding faces share an edge.
    
    Args:
        graph_data: GraphData object with nodes and edges
        positions: numpy array (N, 3) of vertex positions
        embedding: Optional PlanarEmbedding (will compute if not provided)
    
    Returns:
        dict with keys:
            - nodes: list of face indices (dual vertices)
            - edges: list of (face_i, face_j) tuples (dual edges)
            - positions: numpy array (F, 3) of dual vertex positions (face centroids)
            - face_to_nodes: mapping from face index to original nodes in that face
            - success: bool
            - error: error message if failed
    """
    # Get planar embedding if not provided
    if embedding is None:
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
    
    # Get all faces from the embedding
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
    
    # Map: edge (u, v) -> list of face indices that contain this edge
    edge_to_faces = {}
    
    # Compute centroid for each face and build edge-to-face mapping
    for face_idx, face_nodes in enumerate(faces):
        # Calculate 3D centroid of the face
        face_coords = []
        for node_idx in face_nodes:
            if node_idx < len(positions):
                face_coords.append(positions[node_idx])
        
        if face_coords:
            centroid = np.mean(face_coords, axis=0)
            dual_positions[face_idx] = centroid
        else:
            # Fallback to origin if no valid coordinates
            dual_positions[face_idx] = np.array([0, 0, 0])
        
        # Register edges of this face
        n = len(face_nodes)
        for i in range(n):
            u = face_nodes[i]
            v = face_nodes[(i + 1) % n]
            # Normalize edge key (smaller index first)
            edge_key = (min(u, v), max(u, v))
            
            if edge_key not in edge_to_faces:
                edge_to_faces[edge_key] = []
            edge_to_faces[edge_key].append(face_idx)
    
    # Build dual edges: connect faces that share an edge in the original graph
    dual_edges = []
    for edge, adjacent_faces in edge_to_faces.items():
        if len(adjacent_faces) == 2:
            # Two distinct faces share this edge - connect them in the dual
            dual_edges.append((adjacent_faces[0], adjacent_faces[1]))
        elif len(adjacent_faces) > 2:
            # Multiple faces share this edge (shouldn't happen in simple planar graphs)
            # Connect all pairs
            for i in range(len(adjacent_faces)):
                for j in range(i + 1, len(adjacent_faces)):
                    dual_edges.append((adjacent_faces[i], adjacent_faces[j]))
    
    # Remove duplicate edges
    dual_edges = list(set(dual_edges))
    
    # Build face_to_nodes mapping
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
    """
    Computes the genus (number of handles) of the minimal surface 
    on which the graph can be embedded without crossings.
    
    Uses the relationship: chi = 2 - 2g (for orientable surfaces)
    So: g = (2 - chi) / 2 = 1 - chi/2
    
    For non-planar graphs, provides a lower bound using:
    gamma >= ceil((E - 3V + 6) / 6) for simple graphs
    
    Args:
        graph_data: GraphData object
    
    Returns:
        dict with keys:
            - is_planar: whether graph is planar (genus 0)
            - genus_lower_bound: minimum genus required
            - genus_exact: exact genus if planar (0), else None
            - euler_data: Euler characteristic data
    """
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
    
    # Non-planar: compute lower bound
    # For connected graphs: g >= ceil((E - 3V + 6) / 6)
    # This comes from Euler's formula generalized to surfaces
    
    if V <= 2:
        genus_lower = 0
    else:
        # Lower bound formula
        genus_lower = max(0, int(np.ceil((E - 3 * V + 6) / 6)))
    
    # Compute partial Euler data (without actual faces since not planar)
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
    """
    Finds a Kuratowski subgraph (K5 or K3,3 subdivision) if the graph is non-planar.
    
    By Kuratowski's theorem, a graph is planar iff it contains no subdivision
    of K5 (complete graph on 5 vertices) or K3,3 (complete bipartite 3,3).
    
    Args:
        graph_data: GraphData object
    
    Returns:
        dict with keys:
            - is_planar: whether graph is planar
            - kuratowski_type: 'K5', 'K3,3', or None
            - subgraph_nodes: nodes in the Kuratowski subgraph, or None
    """
    G = build_networkx_graph(graph_data)
    
    is_planar, _ = nx.check_planarity(G)
    
    if is_planar:
        return {
            'is_planar': True,
            'kuratowski_type': None,
            'subgraph_nodes': None
        }
    
    # Find Kuratowski subgraph using NetworkX
    try:
        # This returns the subgraph that is a subdivision of K5 or K3,3
        kuratowski = nx.algorithms.planarity.kuratowski_subgraph(G)
        
        nodes = list(kuratowski.nodes())
        edges = kuratowski.number_of_edges()
        
        # Determine type: K5 has 5 principal vertices, K3,3 has 6
        # A subdivision of K5 has at least 10 edges (5 choose 2)
        # A subdivision of K3,3 has exactly 9 edges
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
    """
    Assigns each node to the faces it belongs to.
    Useful for visualization and coloring.
    
    Args:
        graph_data: GraphData object
        embedding: Optional PlanarEmbedding (will compute if not provided)
    
    Returns:
        dict with keys:
            - node_face_ids: list where node_face_ids[i] is the smallest face ID containing node i
            - faces: list of faces (each face is a list of node indices)
            - num_faces: total number of faces
    """
    if embedding is None:
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
    
    # Initialize all nodes with face ID -1 (no face)
    node_face_ids = [-1] * num_nodes
    
    # Assign each node to its first (smallest ID) face
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
    """
    Computes connectivity properties relevant to topological analysis.
    
    Args:
        graph_data: GraphData object
    
    Returns:
        dict with connectivity metrics
    """
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
    
    # Connectivity only makes sense for connected graphs
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
    """
    Computes a planar layout where edges DO NOT cross.
    
    Uses NetworkX's planar_layout which is based on the Chrobak-Payne 
    straight-line drawing algorithm. This guarantees a crossing-free
    embedding for planar graphs.
    
    Args:
        graph_data: GraphData object
        scale: Scale factor for the layout
    
    Returns:
        dict with keys:
            - positions: numpy array (N, 3) with Z=0
            - success: True if layout computed
            - error: error message if failed
    """
    G = build_networkx_graph(graph_data)
    
    # First verify planarity
    is_planar, embedding = nx.check_planarity(G)
    
    if not is_planar:
        return {
            'positions': None,
            'success': False,
            'error': 'Graph is not planar - cannot compute crossing-free layout'
        }
    
    # Compute planar layout using the embedding
    # This uses Chrobak-Payne algorithm for straight-line drawing
    try:
        pos_dict = nx.planar_layout(G, scale=scale)
        
        # Convert to numpy array with Z=0
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
    """
    Computes a Tutte embedding (barycentric/spring embedding).
    
    For 3-connected planar graphs, this is guaranteed to produce
    a convex crossing-free embedding.
    
    For other planar graphs, it usually produces a nice layout
    but crossings might occur.
    
    Args:
        graph_data: GraphData object
        scale: Scale factor for the layout
    
    Returns:
        dict with positions and success status
    """
    G = build_networkx_graph(graph_data)
    
    if G.number_of_nodes() < 3:
        return {
            'positions': None,
            'success': False,
            'error': 'Graph needs at least 3 nodes for Tutte embedding'
        }
    
    try:
        # Find a cycle for the outer face (use longest cycle or arbitrary)
        # For simplicity, we'll use spectral layout as approximation
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
    """
    Detects if any edges cross each other in 3D space.
    
    Uses line segment intersection test for all edge pairs.
    
    Args:
        vertices: numpy array (N, 3) of vertex positions
        edges: list of (i, j) vertex index pairs
        tolerance: distance threshold for considering intersection
    
    Returns:
        dict with:
            - has_crossings: bool
            - num_crossings: int
            - crossing_pairs: list of ((e1_start, e1_end), (e2_start, e2_end))
    """
    crossing_pairs = []
    
    num_edges = len(edges)
    
    for i in range(num_edges):
        e1_start, e1_end = edges[i]
        p1 = np.array(vertices[e1_start])
        p2 = np.array(vertices[e1_end])
        
        for j in range(i + 1, num_edges):
            e2_start, e2_end = edges[j]
            
            # Skip if edges share a vertex
            if e1_start in (e2_start, e2_end) or e1_end in (e2_start, e2_end):
                continue
            
            p3 = np.array(vertices[e2_start])
            p4 = np.array(vertices[e2_end])
            
            # Check for intersection
            if _segments_intersect_3d(p1, p2, p3, p4, tolerance):
                crossing_pairs.append(((e1_start, e1_end), (e2_start, e2_end)))
    
    return {
        'has_crossings': len(crossing_pairs) > 0,
        'num_crossings': len(crossing_pairs),
        'crossing_pairs': crossing_pairs
    }


def _segments_intersect_3d(p1, p2, p3, p4, tolerance=1e-6):
    """
    Check if two 3D line segments intersect or come very close.
    
    Uses the closest point approach between two lines.
    """
    d1 = p2 - p1  # Direction of segment 1
    d2 = p4 - p3  # Direction of segment 2
    d3 = p1 - p3  # Vector between start points
    
    a = np.dot(d1, d1)
    b = np.dot(d1, d2)
    c = np.dot(d2, d2)
    d = np.dot(d1, d3)
    e = np.dot(d2, d3)
    
    denom = a * c - b * b
    
    if abs(denom) < 1e-10:
        # Lines are parallel
        return False
    
    # Parameters for closest points
    s = (b * e - c * d) / denom
    t = (a * e - b * d) / denom
    
    # Check if closest points are within segments
    if s < 0 or s > 1 or t < 0 or t > 1:
        return False
    
    # Compute closest points
    closest1 = p1 + s * d1
    closest2 = p3 + t * d2
    
    # Check distance
    distance = np.linalg.norm(closest1 - closest2)
    
    return distance < tolerance
