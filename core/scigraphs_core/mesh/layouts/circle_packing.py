"""Circle-packing layout algorithms."""

from .common import *
from .basic import _random_layout

def _store_radii_as_mesh_attribute(obj, radii):
    """Write the packing radii to a ``circle_radius`` point attribute for
    Geometry Nodes. Short lists are padded with the mean radius.
    """
    if obj is None or obj.data is None:
        return

    mesh = obj.data
    attr_name = "circle_radius"

    if attr_name in mesh.attributes:
        mesh.attributes.remove(mesh.attributes[attr_name])

    num_verts = len(mesh.vertices)
    num_radii = len(radii)

    if num_radii < num_verts:
        avg_radius = np.mean(radii) if num_radii > 0 else 0.1
        extended_radii = list(radii) + [avg_radius] * (num_verts - num_radii)
    else:
        extended_radii = list(radii[:num_verts])

    attr = mesh.attributes.new(name=attr_name, type='FLOAT', domain='POINT')
    attr.data.foreach_set("value", extended_radii)

    mesh.update()
    print(f"  Mesh attribute 'circle_radius' created on {num_verts} vertices")

def _circle_packing_layout(G, iterations=500, scale=5.0):
    """Circle packing by the Collins-Stephenson algorithm (Koebe's theorem).

    Koebe requires a maximal planar triangulation, so the graph is completed
    with Delaunay first unless its own faces are already all triangles. A
    non-planar graph falls through to :func:`_circle_packing_force_directed`.

    Returns ``(positions, radii)``: an (N, 3) array with Z = 0, and one circle
    radius per node.
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

    nodes_list = list(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(nodes_list)}
    idx_to_node = {i: n for n, i in node_to_idx.items()}

    is_planar, embedding = nx.check_planarity(G)

    if not is_planar:
        print("  Graph is NOT planar - using force-directed fallback")
        return _circle_packing_force_directed(G, iterations, scale)

    # Walk the planar embedding's half-edges to recover its faces.
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

    faces_idx = []
    for face in all_faces:
        face_idx = tuple(node_to_idx[n] for n in face if n in node_to_idx)
        if len(face_idx) >= 3:
            faces_idx.append(face_idx)

    outer_face = max(faces_idx, key=len) if faces_idx else ()

    internal_faces = [f for f in faces_idx if f != outer_face]
    non_triangular = [f for f in internal_faces if len(f) != 3]

    use_graph_faces = len(non_triangular) == 0 and len(internal_faces) > 0

    if use_graph_faces:
        print(f"  Graph is already triangulated: {len(internal_faces)} triangular faces")
        triangles = [tuple(f) for f in internal_faces if len(f) == 3]
        boundary_nodes = set(outer_face)

        # The orientation test below needs coordinates, not just faces.
        try:
            init_pos = nx.planar_layout(G)
        except:
            init_pos = nx.spring_layout(G, seed=42)
        points = np.array([init_pos[n] for n in nodes_list])

    else:
        print(f"  Graph has {len(non_triangular)} non-triangular faces - using Delaunay")

        try:
            init_pos = nx.planar_layout(G)
        except:
            init_pos = nx.spring_layout(G, seed=42, iterations=100)

        points = np.array([init_pos[n] for n in nodes_list])
        tri = Delaunay(points)

        triangles = [tuple(simplex) for simplex in tri.simplices]
        boundary_nodes = set(np.unique(tri.convex_hull))

    # Collins-Stephenson: boundary radii stay fixed and define the outer shape,
    # internal radii iterate until the angles around each one sum to 2*pi. The
    # sign is the part that is easy to get backwards: angle_sum above 2*pi means
    # the radius is too small.
    internal_nodes = [i for i in range(num_nodes) if i not in boundary_nodes]
    nodes_to_update = internal_nodes.copy()

    print(f"  Internal vertices: {len(internal_nodes)}, Boundary: {len(boundary_nodes)}")
    print(f"  Boundary nodes (fixed): {list(boundary_nodes)[:5]}...")
    print(f"  Total triangles: {len(triangles)}")

    triangles_at_vertex = [[] for _ in range(num_nodes)]
    for triangle in triangles:
        u, v, w = triangle[0], triangle[1], triangle[2]
        triangles_at_vertex[u].append((v, w))
        triangles_at_vertex[v].append((u, w))
        triangles_at_vertex[w].append((u, v))

    int_degrees = [len(triangles_at_vertex[i]) for i in internal_nodes]
    if int_degrees:
        print(f"  Internal node degrees: min={min(int_degrees)}, max={max(int_degrees)}, avg={sum(int_degrees)/len(int_degrees):.1f}")

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

    radii = np.ones(num_nodes, dtype=float)

    # Rough starting guess: the more triangles meet at a node, the smaller its
    # radius has to be for their angles to fit into 2*pi. Six is the equilateral
    # case, so nodes below that keep radius 1.
    for i in internal_nodes:
        k = len(triangles_at_vertex[i])
        if k > 0:
            radii[i] = 1.0 / (1.0 + 0.1 * max(0, k - 6))

    print(f"  Initial radii range: {radii.min():.4f} to {radii.max():.4f}")

    tolerance = 1e-6
    actual_iterations = max(iterations, 1000)

    print(f"  Computing radii (max {actual_iterations} iterations)...")

    for it in range(actual_iterations):
        max_error = 0.0
        worst_node = -1
        new_radii = radii.copy()

        for i in nodes_to_update:
            neighbors_pairs = triangles_at_vertex[i]
            k = len(neighbors_pairs)

            if k == 0:
                continue

            angle_sum = 0.0
            for v, w in neighbors_pairs:
                angle = angle_at_vertex(radii[i], radii[v], radii[w])
                angle_sum += angle

            target = 2.0 * math.pi

            error = abs(angle_sum - target)
            if error > max_error:
                max_error = error
                worst_node = i

            # Stephenson's multiplicative update, damped harder while the error
            # is large so it does not overshoot on the way in.
            if angle_sum > 1e-9:
                ratio = angle_sum / target
                if error > 1.0:
                    damping = 0.7
                elif error > 0.1:
                    damping = 0.5
                else:
                    damping = 0.3
                new_radii[i] = radii[i] * (1.0 - damping + damping * ratio)
                # Bounds this wide because an Apollonian packing genuinely
                # reaches radius ratios around 1000:1.
                new_radii[i] = max(1e-6, min(1e6, new_radii[i]))

        radii = new_radii

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
        for i in internal_nodes:
            pairs = triangles_at_vertex[i]
            if pairs:
                angles = [angle_at_vertex(radii[i], radii[v], radii[w]) for v, w in pairs]
                node_error = abs(sum(angles) - 2.0 * math.pi)
                if node_error > 0.1:
                    print(f"    Node {i}: {len(pairs)} triangles, sum={sum(angles):.4f}, error={node_error:.4f}")

    radii = radii / radii.mean()

    def is_ccw(a, b, c):
        """True when triangle a->b->c winds counter-clockwise."""
        A, B, C = points[a], points[b], points[c]
        return (B[0] - A[0]) * (C[1] - A[1]) - (B[1] - A[1]) * (C[0] - A[0]) > 0

    # Half-edge map: (u, v) -> w means w lies to the left of directed edge u->v.
    # Consistent winding is what makes the placement below deterministic.
    half_edge_map = {}

    for triangle in triangles:
        u, v, w = triangle[0], triangle[1], triangle[2]

        if not is_ccw(u, v, w):
            u, v, w = u, w, v

        half_edge_map[(u, v)] = w
        half_edge_map[(v, w)] = u
        half_edge_map[(w, u)] = v

    # Place nodes by breadth-first walk over the oriented edges.
    positions_complex = {}
    placed = [False] * num_nodes

    first_edge = next(iter(half_edge_map.keys()))
    u, v = first_edge
    w = half_edge_map[(u, v)]

    positions_complex[u] = 0j
    placed[u] = True

    positions_complex[v] = complex(radii[u] + radii[v], 0)
    placed[v] = True

    angle_u = angle_at_vertex(radii[u], radii[v], radii[w])
    positions_complex[w] = cmath.rect(radii[u] + radii[w], angle_u)
    placed[w] = True

    # Queue the seed triangle's edges reversed, since the adjacent triangle sits
    # on the other side of each one.
    edge_queue = deque()
    edge_queue.append((v, u))
    edge_queue.append((w, v))
    edge_queue.append((u, w))

    processed_edges = set()

    while edge_queue:
        a, b = edge_queue.popleft()

        if (a, b) in processed_edges:
            continue
        processed_edges.add((a, b))

        if (a, b) not in half_edge_map:
            continue  # Convex hull boundary edge

        c = half_edge_map[(a, b)]

        if placed[c]:
            continue

        r_a, r_b, r_c = radii[a], radii[b], radii[c]
        d_ac = r_a + r_c
        d_bc = r_b + r_c

        # Measure a to b rather than assuming r_a + r_b. The two drift apart as
        # rounding accumulates, and using the measured base is what lets the
        # packing still close up geometrically.
        vec_ab = positions_complex[b] - positions_complex[a]
        dist_ab = abs(vec_ab)

        if dist_ab < 1e-10:
            continue

        # Law of cosines at a:
        # cos(theta) = (d_ac^2 + dist_ab^2 - d_bc^2) / (2 * d_ac * dist_ab)
        denom = 2.0 * d_ac * dist_ab
        if denom < 1e-12:
            theta = math.pi / 3.0
        else:
            cos_val = (d_ac * d_ac + dist_ab * dist_ab - d_bc * d_bc) / denom
            cos_val = max(-1.0, min(1.0, cos_val))
            theta = math.acos(cos_val)

        base_angle = cmath.phase(vec_ab)

        new_pos = positions_complex[a] + cmath.rect(d_ac, base_angle + theta)
        positions_complex[c] = new_pos
        placed[c] = True

        edge_queue.append((b, c))
        edge_queue.append((c, a))

    # Scatter anything the walk missed. A valid Delaunay leaves nothing here.
    for i in range(num_nodes):
        if not placed[i]:
            positions_complex[i] = complex(
                np.random.uniform(-scale, scale),
                np.random.uniform(-scale, scale)
            )
            placed[i] = True

    positions = np.zeros((num_nodes, 2), dtype=float)
    for i in range(num_nodes):
        p = positions_complex.get(i, 0j)
        positions[i] = [p.real, p.imag]

    # The walk accumulates rounding error, so finish with gradient descent that
    # pushes adjacent circles back to true tangency.
    edges_idx = [(node_to_idx[u], node_to_idx[v]) for u, v in G.edges()]

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

    if initial_max > 0.01:
        print(f"  Refining positions (initial tangency error: {initial_max:.4f})...")

        learning_rate = 0.1
        refinement_iters = 200

        for ref_it in range(refinement_iters):
            gradients = np.zeros_like(positions)

            for ui, vi in edges_idx:
                diff = positions[ui] - positions[vi]
                dist = np.linalg.norm(diff)
                if dist < 1e-10:
                    continue

                target = radii[ui] + radii[vi]
                error = dist - target

                direction = diff / dist
                grad = error * direction

                gradients[ui] += grad
                gradients[vi] -= grad

            grad_norm = np.linalg.norm(gradients)
            if grad_norm > 1e-10:
                # Damped by the gradient norm, so one bad node cannot dominate.
                step = learning_rate * gradients / (1.0 + 0.1 * grad_norm)
                positions -= step

            _, current_max = compute_tangency_error(positions, radii, edges_idx)
            if current_max < 0.01:
                print(f"  Position refinement converged at iteration {ref_it}")
                break

            if ref_it > 0 and ref_it % 50 == 0:
                learning_rate *= 0.8

        _, final_max = compute_tangency_error(positions, radii, edges_idx)
        if final_max < initial_max:
            print(f"  Position refinement: {initial_max:.4f} -> {final_max:.4f}")
        else:
            print(f"  Position refinement did not improve (kept original)")

    center = positions.mean(axis=0)
    positions -= center

    max_extent = np.max(np.abs(positions)) + np.max(radii)
    if max_extent > 1e-6:
        scale_factor = (scale * 0.45) / max_extent
        positions *= scale_factor
        radii *= scale_factor

    # Report against the original graph's edges, not the triangulated ones.
    max_error = 0.0
    for u, v in G.edges():
        ui, vi = node_to_idx[u], node_to_idx[v]
        dist = np.linalg.norm(positions[ui] - positions[vi])
        target = radii[ui] + radii[vi]
        error = abs(dist - target)
        max_error = max(max_error, error)

    positions_3d = np.zeros((num_nodes, 3))
    positions_3d[:, :2] = positions

    elapsed = time.time() - start
    print(f"  Circle Packing completed in {elapsed:.2f}s")
    print(f"  Radii range: {radii.min():.4f} to {radii.max():.4f}")
    print(f"  Final max tangency error: {max_error:.4f}")

    return positions_3d, radii

def _circle_packing_force_directed(G, iterations=500, scale=5.0):
    """Circle packing by force simulation, for graphs Collins-Stephenson cannot
    take: adjacent circles pull toward tangency, overlapping ones push apart, and
    radius follows degree.
    """
    import time
    import math

    start = time.time()
    num_nodes = len(G.nodes())

    degrees = np.array([G.degree(n) for n in G.nodes()], dtype=float)
    max_degree = max(degrees.max(), 1.0)

    radii = 0.3 + 0.7 * (degrees / max_degree)
    radii = radii * scale * 0.1

    pos_dict = nx.spring_layout(G, dim=2, scale=scale * 0.3, iterations=50)
    nodes_list = list(G.nodes())
    positions = np.array([pos_dict[n] for n in nodes_list])

    node_to_idx = {n: i for i, n in enumerate(nodes_list)}
    adj_set = [set() for _ in range(num_nodes)]
    for u, v in G.edges():
        ui, vi = node_to_idx[u], node_to_idx[v]
        adj_set[ui].add(vi)
        adj_set[vi].add(ui)

    temperature = scale * 0.2
    cooling_rate = 0.98

    for it in range(iterations):
        forces = np.zeros((num_nodes, 2))

        for i in range(num_nodes):
            for j in adj_set[i]:
                delta = positions[j] - positions[i]
                dist = np.linalg.norm(delta)

                if dist < 1e-6:
                    delta = np.random.rand(2) - 0.5
                    dist = np.linalg.norm(delta)

                target_dist = radii[i] + radii[j]

                diff = dist - target_dist
                force_mag = diff * 0.3
                forces[i] += force_mag * (delta / dist)

            # Overlap is pushed apart whether or not the pair shares an edge.
            for j in range(num_nodes):
                if i == j:
                    continue

                delta = positions[j] - positions[i]
                dist = np.linalg.norm(delta)

                if dist < 1e-6:
                    delta = np.random.rand(2) - 0.5
                    dist = np.linalg.norm(delta)

                min_dist = radii[i] + radii[j]

                if dist < min_dist:
                    overlap = min_dist - dist
                    forces[i] -= overlap * 0.5 * (delta / dist)
                elif j not in adj_set[i]:
                    repulsion = 0.1 * (radii[i] + radii[j]) / (dist * dist + 0.1)
                    forces[i] -= repulsion * (delta / dist)

        for i in range(num_nodes):
            force_mag = np.linalg.norm(forces[i])
            if force_mag > temperature:
                forces[i] = forces[i] / force_mag * temperature
            positions[i] += forces[i]

        temperature *= cooling_rate

        if temperature < 1e-4:
            break

    center = positions.mean(axis=0)
    positions -= center

    max_extent = np.max(np.abs(positions)) + np.max(radii)
    if max_extent > 1e-6:
        scale_factor = (scale * 0.45) / max_extent
        positions *= scale_factor
        radii *= scale_factor

    max_error = 0.0
    for u, v in G.edges():
        ui, vi = node_to_idx[u], node_to_idx[v]
        dist = np.linalg.norm(positions[ui] - positions[vi])
        target = radii[ui] + radii[vi]
        error = abs(dist - target)
        max_error = max(max_error, error)

    positions_3d = np.zeros((num_nodes, 3))
    positions_3d[:, :2] = positions

    elapsed = time.time() - start
    print(f"  Force-directed packing completed in {elapsed:.2f}s")
    print(f"  Radii range: {radii.min():.4f} to {radii.max():.4f}")
    print(f"  Final max tangency error: {max_error:.4f}")

    return positions_3d, radii

__all__ = [name for name in globals() if not name.startswith('__')]
