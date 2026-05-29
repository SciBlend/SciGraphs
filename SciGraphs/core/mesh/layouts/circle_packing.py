"""Circle-packing layout algorithms."""

from .common import *
from .basic import _random_layout

def _store_radii_as_mesh_attribute(obj, radii):
    """
    Store circle packing radii as a mesh attribute for Geometry Nodes.

    Args:
        obj: Blender mesh object
        radii: numpy array or list of radii values
    """
    if obj is None or obj.data is None:
        return

    mesh = obj.data
    attr_name = "circle_radius"

    # Remove existing attribute if present
    if attr_name in mesh.attributes:
        mesh.attributes.remove(mesh.attributes[attr_name])

    # Create new float attribute on vertices
    num_verts = len(mesh.vertices)
    num_radii = len(radii)

    # Expand or truncate radii to match vertex count
    if num_radii < num_verts:
        # Extend with default value (average radius)
        avg_radius = np.mean(radii) if num_radii > 0 else 0.1
        extended_radii = list(radii) + [avg_radius] * (num_verts - num_radii)
    else:
        extended_radii = list(radii[:num_verts])

    attr = mesh.attributes.new(name=attr_name, type='FLOAT', domain='POINT')
    attr.data.foreach_set("value", extended_radii)

    mesh.update()
    print(f"  Mesh attribute 'circle_radius' created on {num_verts} vertices")

def _circle_packing_layout(G, iterations=500, scale=5.0):
    """
    Circle Packing layout based on Collins-Stephenson algorithm (Koebe theorem).

    CRITICAL: This implementation uses Delaunay triangulation to complete the graph
    into a maximal planar triangulation before computing radii. This is the
    "Graph Completer" step required by the theorem.

    Args:
        G: NetworkX graph (will be triangulated internally)
        iterations: number of iterations for radius computation
        scale: overall scale of the layout

    Returns:
        tuple: (positions, radii)
            - positions: numpy array (N, 3) with Z=0
            - radii: numpy array (N,) with circle radii for each node
    """
    import time
    import math
    import cmath
    from collections import deque
    from scipy.spatial import Delaunay

    start = time.time()

    num_nodes = len(G.nodes())
    num_edges = G.number_of_edges()
    print(f"Computing Circle Packing layout for {num_nodes} nodes, {num_edges} edges...")

    if num_nodes == 0:
        return np.zeros((0, 3)), np.zeros(0)

    if num_nodes == 1:
        return np.array([[0.0, 0.0, 0.0]]), np.array([scale * 0.5])

    if num_nodes == 2:
        r = scale * 0.25
        return np.array([[-r, 0.0, 0.0], [r, 0.0, 0.0]]), np.array([r, r])

    if num_nodes < 3:
        return _circle_packing_force_directed(G, iterations, scale)

    # STEP 1: Node mapping and initial layout for triangulation
    nodes_list = list(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(nodes_list)}
    idx_to_node = {i: n for n, i in node_to_idx.items()}

    # STEP 2: Check if graph is already a triangulation
    # If all faces are triangles, use the graph's own structure
    # Otherwise, use Delaunay to complete it

    # First, check if graph is planar and get its embedding
    is_planar, embedding = nx.check_planarity(G)

    if not is_planar:
        print("  Graph is NOT planar - using force-directed fallback")
        return _circle_packing_force_directed(G, iterations, scale)

    # Extract faces from planar embedding
    all_faces = []
    visited_half_edges = set()

    for start_node in embedding:
        for neighbor in embedding.neighbors_cw_order(start_node):
            if (start_node, neighbor) in visited_half_edges:
                continue

            face = []
            current = start_node
            next_node = neighbor

            for _ in range(num_nodes + 10):
                face.append(current)
                visited_half_edges.add((current, next_node))

                neighbors_cw = list(embedding.neighbors_cw_order(next_node))
                if current not in neighbors_cw:
                    break

                idx = neighbors_cw.index(current)
                prev_idx = (idx - 1) % len(neighbors_cw)
                new_next = neighbors_cw[prev_idx]

                current = next_node
                next_node = new_next

                if current == start_node and next_node == neighbor:
                    break

            if len(face) >= 3:
                all_faces.append(face)

    # Convert faces to index-based
    faces_idx = []
    for face in all_faces:
        face_idx = tuple(node_to_idx[n] for n in face if n in node_to_idx)
        if len(face_idx) >= 3:
            faces_idx.append(face_idx)

    # Find outer face (largest)
    outer_face = max(faces_idx, key=len) if faces_idx else ()

    # Check if graph is already a triangulation (all internal faces are triangles)
    internal_faces = [f for f in faces_idx if f != outer_face]
    non_triangular = [f for f in internal_faces if len(f) != 3]

    use_graph_faces = len(non_triangular) == 0 and len(internal_faces) > 0

    if use_graph_faces:
        # Graph is already triangulated - use its own faces
        print(f"  Graph is already triangulated: {len(internal_faces)} triangular faces")
        triangles = [tuple(f) for f in internal_faces if len(f) == 3]
        boundary_nodes = set(outer_face)

        # Also need positions for layout - use planar layout
        try:
            init_pos = nx.planar_layout(G)
        except:
            init_pos = nx.spring_layout(G, seed=42)
        points = np.array([init_pos[n] for n in nodes_list])

    else:
        # Graph needs triangulation - use Delaunay
        print(f"  Graph has {len(non_triangular)} non-triangular faces - using Delaunay")

        try:
            init_pos = nx.planar_layout(G)
        except:
            init_pos = nx.spring_layout(G, seed=42, iterations=100)

        points = np.array([init_pos[n] for n in nodes_list])
        tri = Delaunay(points)

        triangles = [tuple(simplex) for simplex in tri.simplices]
        boundary_nodes = set(np.unique(tri.convex_hull))

    # Collins-Stephenson algorithm:
    # 1. Fix ALL boundary node radii (they define the outer shape)
    # 2. Iterate ONLY on internal nodes to achieve angle sum = 2*pi
    # 3. When angle_sum > 2*pi, the radius is too SMALL (increase it)
    # 4. When angle_sum < 2*pi, the radius is too LARGE (decrease it)

    internal_nodes = [i for i in range(num_nodes) if i not in boundary_nodes]

    # Only update internal nodes - boundary radii are fixed
    nodes_to_update = internal_nodes.copy()

    print(f"  Internal vertices: {len(internal_nodes)}, Boundary: {len(boundary_nodes)}")
    print(f"  Boundary nodes (fixed): {list(boundary_nodes)[:5]}...")
    print(f"  Total triangles: {len(triangles)}")

    # Build triangles-at-vertex structure
    triangles_at_vertex = [[] for _ in range(num_nodes)]
    for triangle in triangles:
        u, v, w = triangle[0], triangle[1], triangle[2]
        triangles_at_vertex[u].append((v, w))
        triangles_at_vertex[v].append((u, w))
        triangles_at_vertex[w].append((u, v))

    # Log triangle structure (brief)
    int_degrees = [len(triangles_at_vertex[i]) for i in internal_nodes]
    if int_degrees:
        print(f"  Internal node degrees: min={min(int_degrees)}, max={max(int_degrees)}, avg={sum(int_degrees)/len(int_degrees):.1f}")

    # STEP 4: Compute radii (Stephenson's algorithm)
    def angle_at_vertex(r_i, r_j, r_k):
        """Angle at vertex i in triangle (i,j,k) with tangent circles."""
        a = r_i + r_j
        b = r_i + r_k
        c = r_j + r_k

        denom = 2.0 * a * b
        if denom < 1e-12:
            return math.pi / 3.0

        cos_val = (a*a + b*b - c*c) / denom
        cos_val = max(-1.0, min(1.0, cos_val))
        return math.acos(cos_val)

    # Initialize radii uniformly - the algorithm will find correct proportions
    # All radii start equal; internal nodes will be adjusted to satisfy angle constraint
    radii = np.ones(num_nodes, dtype=float)

    # Count triangles at each internal node to estimate initial radii
    # More triangles = smaller radius needed to fit all angles in 2*pi
    for i in internal_nodes:
        k = len(triangles_at_vertex[i])
        if k > 0:
            # Estimate: if k triangles must sum to 2*pi, avg angle approx 2*pi/k
            # For equilateral configuration, this gives a rough starting point
            radii[i] = 1.0 / (1.0 + 0.1 * max(0, k - 6))

    print(f"  Initial radii range: {radii.min():.4f} to {radii.max():.4f}")

    tolerance = 1e-6
    actual_iterations = max(iterations, 1000)

    print(f"  Computing radii (max {actual_iterations} iterations)...")

    for it in range(actual_iterations):
        max_error = 0.0
        worst_node = -1
        new_radii = radii.copy()

        # Only update internal nodes; boundary radii stay fixed
        for i in nodes_to_update:
            neighbors_pairs = triangles_at_vertex[i]
            k = len(neighbors_pairs)

            if k == 0:
                continue

            # Compute angle sum at this vertex
            angle_sum = 0.0
            for v, w in neighbors_pairs:
                angle = angle_at_vertex(radii[i], radii[v], radii[w])
                angle_sum += angle

            # Target is always 2*pi for internal nodes
            target = 2.0 * math.pi

            error = abs(angle_sum - target)
            if error > max_error:
                max_error = error
                worst_node = i

            # Update formula (Stephenson's method):
            # - If angle_sum > 2*pi, radius is too SMALL -> INCREASE
            # - If angle_sum < 2*pi, radius is too LARGE -> DECREASE
            # Using multiplicative update: r_new = r_old * (angle_sum / 2*pi)
            if angle_sum > 1e-9:
                ratio = angle_sum / target
                # Adaptive damping: more aggressive when far, gentler when close
                if error > 1.0:
                    damping = 0.7
                elif error > 0.1:
                    damping = 0.5
                else:
                    damping = 0.3
                new_radii[i] = radii[i] * (1.0 - damping + damping * ratio)
                # Wide bounds for complex graphs (Apollonian can have 1000:1 ratios)
                new_radii[i] = max(1e-6, min(1e6, new_radii[i]))

        radii = new_radii

        # Log progress periodically
        if it % 200 == 0:
            at_lower = sum(1 for r in radii if r <= 1e-5)
            at_upper = sum(1 for r in radii if r >= 1e5)
            bounds_info = f", at_bounds={at_lower}L/{at_upper}U" if (at_lower + at_upper) > 0 else ""
            print(f"  Iter {it}: error={max_error:.4f}, radii=[{radii.min():.2e}, {radii.max():.2e}]{bounds_info}")

        if max_error < tolerance:
            print(f"  Radii converged at iteration {it}, error={max_error:.2e}")
            break

    if it == actual_iterations - 1 and max_error > tolerance:
        print(f"  Warning: iteration limit reached, final error={max_error:.4f}")
        # Show which nodes are problematic
        for i in internal_nodes:
            pairs = triangles_at_vertex[i]
            if pairs:
                angles = [angle_at_vertex(radii[i], radii[v], radii[w]) for v, w in pairs]
                node_error = abs(sum(angles) - 2.0 * math.pi)
                if node_error > 0.1:
                    print(f"    Node {i}: {len(pairs)} triangles, sum={sum(angles):.4f}, error={node_error:.4f}")

    # Normalize radii
    radii = radii / radii.mean()

    # STEP 5: Build oriented half-edge map for deterministic layout

    def is_ccw(a, b, c):
        """Check if triangle a->b->c is counter-clockwise using cross product."""
        A, B, C = points[a], points[b], points[c]
        return (B[0] - A[0]) * (C[1] - A[1]) - (B[1] - A[1]) * (C[0] - A[0]) > 0

    # Build half-edge map: (u, v) -> w means w is to the LEFT of directed edge u->v
    half_edge_map = {}

    for triangle in triangles:
        u, v, w = triangle[0], triangle[1], triangle[2]

        # Ensure CCW orientation
        if not is_ccw(u, v, w):
            u, v, w = u, w, v  # Flip to make CCW

        # Register oriented edges: each edge points to the vertex on its left
        half_edge_map[(u, v)] = w
        half_edge_map[(v, w)] = u
        half_edge_map[(w, u)] = v

    # STEP 6: Deterministic layout using oriented edge BFS
    positions_complex = {}
    placed = [False] * num_nodes

    # Get a starting edge from the half-edge map
    first_edge = next(iter(half_edge_map.keys()))
    u, v = first_edge
    w = half_edge_map[(u, v)]

    # Place first triangle (u, v, w is CCW)
    positions_complex[u] = 0j
    placed[u] = True

    positions_complex[v] = complex(radii[u] + radii[v], 0)
    placed[v] = True

    # w is to the left of u->v, so rotate CCW (positive angle)
    angle_u = angle_at_vertex(radii[u], radii[v], radii[w])
    positions_complex[w] = cmath.rect(radii[u] + radii[w], angle_u)
    placed[w] = True

    # Edge queue: oriented edges whose opposite vertex might need placement
    # For the initial triangle (u,v,w) CCW, the external edges are (v,u), (w,v), (u,w)
    edge_queue = deque()
    edge_queue.append((v, u))  # Opposite direction to look for adjacent triangle
    edge_queue.append((w, v))
    edge_queue.append((u, w))

    processed_edges = set()

    while edge_queue:
        a, b = edge_queue.popleft()

        if (a, b) in processed_edges:
            continue
        processed_edges.add((a, b))

        # Look up who is to the left of a->b
        if (a, b) not in half_edge_map:
            continue  # Convex hull boundary edge

        c = half_edge_map[(a, b)]

        if placed[c]:
            continue

        # Calculate position of c DETERMINISTICALLY
        # c is to the LEFT of a->b, so we rotate CCW (add angle)
        r_a, r_b, r_c = radii[a], radii[b], radii[c]
        d_ac = r_a + r_c
        d_bc = r_b + r_c

        # CRITICAL: Use ACTUAL distance between placed nodes, not theoretical
        # This allows the layout to close geometrically even with numerical errors
        vec_ab = positions_complex[b] - positions_complex[a]
        dist_ab = abs(vec_ab)  # Real distance, may differ from r_a + r_b

        if dist_ab < 1e-10:
            continue

        # Angle at vertex a using law of cosines with REAL base distance
        # cos(theta) = (d_ac^2 + dist_ab^2 - d_bc^2) / (2 * d_ac * dist_ab)
        denom = 2.0 * d_ac * dist_ab
        if denom < 1e-12:
            theta = math.pi / 3.0
        else:
            cos_val = (d_ac * d_ac + dist_ab * dist_ab - d_bc * d_bc) / denom
            cos_val = max(-1.0, min(1.0, cos_val))
            theta = math.acos(cos_val)

        # Base direction: a -> b
        base_angle = cmath.phase(vec_ab)

        # c is to the LEFT, so rotate CCW (add the angle)
        new_pos = positions_complex[a] + cmath.rect(d_ac, base_angle + theta)
        positions_complex[c] = new_pos
        placed[c] = True

        # Add new boundary edges (opposite orientation to expand)
        edge_queue.append((b, c))
        edge_queue.append((c, a))

    # Handle any remaining unplaced nodes (shouldn't happen with valid Delaunay)
    for i in range(num_nodes):
        if not placed[i]:
            positions_complex[i] = complex(
                np.random.uniform(-scale, scale),
                np.random.uniform(-scale, scale)
            )
            placed[i] = True

    # Convert to numpy array
    positions = np.zeros((num_nodes, 2), dtype=float)
    for i in range(num_nodes):
        p = positions_complex.get(i, 0j)
        positions[i] = [p.real, p.imag]

    # STEP 7: Position refinement to reduce tangency error
    # The BFS placement accumulates numerical errors; refine with
    # gradient descent to make adjacent circles truly tangent

    # Build edge list for original graph
    edges_idx = [(node_to_idx[u], node_to_idx[v]) for u, v in G.edges()]

    # Compute initial tangency error
    def compute_tangency_error(pos, rad, edges):
        total_err = 0.0
        max_err = 0.0
        for ui, vi in edges:
            dist = np.linalg.norm(pos[ui] - pos[vi])
            target = rad[ui] + rad[vi]
            err = abs(dist - target)
            total_err += err * err
            max_err = max(max_err, err)
        return total_err, max_err

    initial_error, initial_max = compute_tangency_error(positions, radii, edges_idx)

    # Only refine if there's significant error
    if initial_max > 0.01:
        print(f"  Refining positions (initial tangency error: {initial_max:.4f})...")

        # Gradient descent refinement
        learning_rate = 0.1
        refinement_iters = 200

        for ref_it in range(refinement_iters):
            # Compute gradient for each node
            gradients = np.zeros_like(positions)

            for ui, vi in edges_idx:
                diff = positions[ui] - positions[vi]
                dist = np.linalg.norm(diff)
                if dist < 1e-10:
                    continue

                target = radii[ui] + radii[vi]
                error = dist - target

                # Gradient: push apart if too close, pull together if too far
                direction = diff / dist
                grad = error * direction

                gradients[ui] += grad
                gradients[vi] -= grad

            # Apply gradient with adaptive learning rate
            grad_norm = np.linalg.norm(gradients)
            if grad_norm > 1e-10:
                # Normalize and apply
                step = learning_rate * gradients / (1.0 + 0.1 * grad_norm)
                positions -= step

            # Check convergence
            _, current_max = compute_tangency_error(positions, radii, edges_idx)
            if current_max < 0.01:
                print(f"  Position refinement converged at iteration {ref_it}")
                break

            # Adaptive learning rate
            if ref_it > 0 and ref_it % 50 == 0:
                learning_rate *= 0.8

        _, final_max = compute_tangency_error(positions, radii, edges_idx)
        if final_max < initial_max:
            print(f"  Position refinement: {initial_max:.4f} -> {final_max:.4f}")
        else:
            print(f"  Position refinement did not improve (kept original)")

    # Center and scale
    center = positions.mean(axis=0)
    positions -= center

    max_extent = np.max(np.abs(positions)) + np.max(radii)
    if max_extent > 1e-6:
        scale_factor = (scale * 0.45) / max_extent
        positions *= scale_factor
        radii *= scale_factor

    # Compute final tangency error for ORIGINAL graph edges
    max_error = 0.0
    for u, v in G.edges():
        ui, vi = node_to_idx[u], node_to_idx[v]
        dist = np.linalg.norm(positions[ui] - positions[vi])
        target = radii[ui] + radii[vi]
        error = abs(dist - target)
        max_error = max(max_error, error)

    # Convert to 3D
    positions_3d = np.zeros((num_nodes, 3))
    positions_3d[:, :2] = positions

    elapsed = time.time() - start
    print(f"  Circle Packing completed in {elapsed:.2f}s")
    print(f"  Radii range: {radii.min():.4f} to {radii.max():.4f}")
    print(f"  Final max tangency error: {max_error:.4f}")

    return positions_3d, radii

def _circle_packing_force_directed(G, iterations=500, scale=5.0):
    """
    Force-directed circle packing for non-planar graphs.

    Uses a force simulation where:
    - Adjacent circles want to be tangent (attraction to ideal distance)
    - Non-adjacent circles repel to avoid overlap
    - Radii are based on node degree

    This is a fallback for graphs that cannot use the Collins-Stephenson algorithm.
    """
    import time
    import math

    start = time.time()
    num_nodes = len(G.nodes())

    # Initialize radii based on degree
    degrees = np.array([G.degree(n) for n in G.nodes()], dtype=float)
    max_degree = max(degrees.max(), 1.0)

    # Higher degree = larger circle
    radii = 0.3 + 0.7 * (degrees / max_degree)
    radii = radii * scale * 0.1

    # Initialize positions using spring layout as starting point
    pos_dict = nx.spring_layout(G, dim=2, scale=scale * 0.3, iterations=50)
    nodes_list = list(G.nodes())
    positions = np.array([pos_dict[n] for n in nodes_list])

    # Build adjacency for efficient lookup
    node_to_idx = {n: i for i, n in enumerate(nodes_list)}
    adj_set = [set() for _ in range(num_nodes)]
    for u, v in G.edges():
        ui, vi = node_to_idx[u], node_to_idx[v]
        adj_set[ui].add(vi)
        adj_set[vi].add(ui)

    # Force-directed iteration
    temperature = scale * 0.2
    cooling_rate = 0.98

    for it in range(iterations):
        forces = np.zeros((num_nodes, 2))

        for i in range(num_nodes):
            # Attractive forces to neighbors (achieve tangency)
            for j in adj_set[i]:
                delta = positions[j] - positions[i]
                dist = np.linalg.norm(delta)

                if dist < 1e-6:
                    delta = np.random.rand(2) - 0.5
                    dist = np.linalg.norm(delta)

                # Target: sum of radii (tangent)
                target_dist = radii[i] + radii[j]

                # Force proportional to difference from target
                diff = dist - target_dist
                force_mag = diff * 0.3
                forces[i] += force_mag * (delta / dist)

            # Repulsive forces from ALL other nodes (prevent overlap)
            for j in range(num_nodes):
                if i == j:
                    continue

                delta = positions[j] - positions[i]
                dist = np.linalg.norm(delta)

                if dist < 1e-6:
                    delta = np.random.rand(2) - 0.5
                    dist = np.linalg.norm(delta)

                # Minimum distance: sum of radii
                min_dist = radii[i] + radii[j]

                if dist < min_dist:
                    # Push apart
                    overlap = min_dist - dist
                    forces[i] -= overlap * 0.5 * (delta / dist)
                elif j not in adj_set[i]:
                    # Small repulsion for non-neighbors to spread out
                    repulsion = 0.1 * (radii[i] + radii[j]) / (dist * dist + 0.1)
                    forces[i] -= repulsion * (delta / dist)

        # Apply forces with temperature limiting
        for i in range(num_nodes):
            force_mag = np.linalg.norm(forces[i])
            if force_mag > temperature:
                forces[i] = forces[i] / force_mag * temperature
            positions[i] += forces[i]

        # Cool down
        temperature *= cooling_rate

        if temperature < 1e-4:
            break

    # Center the layout
    center = positions.mean(axis=0)
    positions -= center

    # Scale to desired size
    max_extent = np.max(np.abs(positions)) + np.max(radii)
    if max_extent > 1e-6:
        scale_factor = (scale * 0.45) / max_extent
        positions *= scale_factor
        radii *= scale_factor

    # Compute tangency error
    max_error = 0.0
    for u, v in G.edges():
        ui, vi = node_to_idx[u], node_to_idx[v]
        dist = np.linalg.norm(positions[ui] - positions[vi])
        target = radii[ui] + radii[vi]
        error = abs(dist - target)
        max_error = max(max_error, error)

    # Convert to 3D
    positions_3d = np.zeros((num_nodes, 3))
    positions_3d[:, :2] = positions

    elapsed = time.time() - start
    print(f"  Force-directed packing completed in {elapsed:.2f}s")
    print(f"  Radii range: {radii.min():.4f} to {radii.max():.4f}")
    print(f"  Final max tangency error: {max_error:.4f}")

    return positions_3d, radii

__all__ = [name for name in globals() if not name.startswith('__')]
