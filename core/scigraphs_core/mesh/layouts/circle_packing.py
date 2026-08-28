"""Circle-packing layout algorithms."""

from collections import deque

from .common import *
from .basic import _random_layout

try:
    from scipy.spatial import cKDTree
except ImportError:
    cKDTree = None

_MAX_RADIUS_SPREAD = 1e3

def _store_radii_as_mesh_attribute(obj, radii):
    """Store one packing radius per node under ``obj["circle_radius"]``, padding
    a short list with the mean radius. Turning that into a mesh point attribute
    is the add-on's job; this package never touches ``obj.data``."""
    if obj is None:
        return

    try:
        num_verts = int(obj["num_nodes"])
    except (KeyError, IndexError, TypeError, ValueError):
        return

    num_radii = len(radii)

    if num_radii < num_verts:
        avg_radius = float(np.mean(radii)) if num_radii > 0 else 0.1
        extended_radii = [float(r) for r in radii] + [avg_radius] * (num_verts - num_radii)
    else:
        extended_radii = [float(r) for r in radii[:num_verts]]

    obj["circle_radius"] = extended_radii
    print(f"  Radii stored under 'circle_radius' for {num_verts} nodes")

def _face_key(face):
    """Canonical rotation of a cyclic index sequence, for comparing faces."""
    face = list(face)
    if not face:
        return ()
    start = face.index(min(face))
    return tuple(face[start:] + face[:start])

def _planar_triangulation(G, node_to_idx):
    """Complete a planar graph to a triangulated disk and return
    ``(triangles, outer_face)`` as index tuples, or None when that is not
    possible. Edges are only ever added, so every edge of *G* survives into the
    triangulation and can therefore reach exact tangency."""
    if nx is None:
        return None

    try:
        from networkx.algorithms.planar_drawing import triangulate_embedding
    except ImportError:
        return None

    simple = nx.Graph()
    simple.add_nodes_from(G.nodes())
    simple.add_edges_from((u, v) for u, v in G.edges() if u != v)

    is_planar, embedding = nx.check_planarity(simple)
    if not is_planar:
        return None

    embedding, outer_face = triangulate_embedding(embedding, False)

    faces = []
    visited_half_edges = set()
    for v in embedding:
        for w in embedding.neighbors_cw_order(v):
            if (v, w) in visited_half_edges:
                continue
            faces.append(embedding.traverse_face(v, w, mark_half_edges=visited_half_edges))

    faces = [tuple(node_to_idx[n] for n in f) for f in faces if len(f) >= 3]
    if not faces:
        return None

    outer_idx = tuple(node_to_idx[n] for n in outer_face)
    outer_keys = {_face_key(outer_idx), _face_key(outer_idx[::-1])}
    which = next((i for i, f in enumerate(faces) if _face_key(f) in outer_keys), None)
    if which is None:
        which = max(range(len(faces)), key=lambda i: len(faces[i]))

    triangles = [f for i, f in enumerate(faces) if i != which]
    if not triangles or any(len(f) != 3 for f in triangles):
        return None

    return triangles, faces[which]

def _packing_angle(r_i, r_j, r_k):
    """Angle at vertex i in the triangle of mutually tangent circles (i, j, k)."""
    a = r_i + r_j
    b = r_i + r_k
    c = r_j + r_k

    denom = 2.0 * a * b
    if denom < 1e-12:
        return math.pi / 3.0

    cos_val = (a*a + b*b - c*c) / denom
    cos_val = max(-1.0, min(1.0, cos_val))
    return math.acos(cos_val)

def _packing_aims(triangles_at_vertex, boundary):
    """Target angle sum per vertex: 2*pi inside, and on the boundary a share of
    the (B-2)*pi that a convex polygon has, split in proportion to the face count
    and capped at pi. Discrete Gauss-Bonnet makes the two halves add up exactly,
    and a convex outer polygon over locally flat interiors is what keeps the
    laid-out packing embedded."""
    aims = np.full(len(triangles_at_vertex), 2.0 * math.pi)

    boundary_list = sorted(boundary)
    counts = [len(triangles_at_vertex[i]) for i in boundary_list]
    total = (len(boundary_list) - 2.0) * math.pi
    if len(boundary_list) < 3 or total <= 0.0 or sum(counts) == 0:
        return aims

    lo, hi = 0.0, math.pi
    for _ in range(60):
        c = 0.5 * (lo + hi)
        if sum(min(c * k, math.pi) for k in counts) < total:
            lo = c
        else:
            hi = c

    c = 0.5 * (lo + hi)
    for i, k in zip(boundary_list, counts):
        aims[i] = min(c * k, math.pi)
    return aims

def _solve_packing_radii(triangles_at_vertex, aims, free, max_iterations, tolerance=1e-9):
    """Collins-Stephenson radius solver. Each sweep replaces a vertex's flower by
    a uniform-neighbour flower carrying the same angle sum, then rescales the
    centre radius so that flower closes on the vertex's aim. Jacobi rather than
    Gauss-Seidel, so a sweep is one vectorised pass over the corners.
    Returns ``(radii, max_error, sweeps)``."""
    num_nodes = len(triangles_at_vertex)
    radii = np.ones(num_nodes, dtype=float)

    corners = [(i, v, w) for i, pairs in enumerate(triangles_at_vertex) for v, w in pairs]
    free_mask = np.zeros(num_nodes, dtype=bool)
    free_mask[list(free)] = True
    if not corners or not free_mask.any():
        return radii, 0.0, 0

    centre, left, right = (np.array(col, dtype=int) for col in zip(*corners))
    counts = np.maximum(np.bincount(centre, minlength=num_nodes), 1).astype(float)
    delta = np.sin(np.minimum(aims / (2.0 * counts), 0.5 * math.pi))

    max_error = 0.0
    sweeps = 0

    for sweeps in range(1, max_iterations + 1):
        a = radii[centre] + radii[left]
        b = radii[centre] + radii[right]
        c = radii[left] + radii[right]
        denom = np.maximum(2.0 * a * b, 1e-12)
        angles = np.arccos(np.clip((a*a + b*b - c*c) / denom, -1.0, 1.0))
        angle_sum = np.bincount(centre, weights=angles, minlength=num_nodes)

        max_error = float(np.abs(angle_sum - aims)[free_mask].max())
        if max_error < tolerance:
            break

        beta = np.minimum(np.sin(angle_sum / (2.0 * counts)), 1.0 - 1e-12)
        uniform_neighbour = beta * radii / (1.0 - beta)
        updated = np.maximum(1e-12, uniform_neighbour * (1.0 - delta) / delta)
        radii = np.where(free_mask, updated, radii)

        mean = radii.mean()
        if mean > 0.0:
            radii /= mean

    return radii, max_error, sweeps

def _lay_out_packing(triangles, radii, num_nodes):
    """Place centres by walking the triangulation's dual, each new circle
    tangent to the two already placed. Returns ``(positions, placed)``."""
    # (u, v) -> w: w lies left of u->v; consistent winding keeps this stable.
    half_edge_map = {}
    for a, b, c in triangles:
        half_edge_map[(a, b)] = c
        half_edge_map[(b, c)] = a
        half_edge_map[(c, a)] = b

    centres = {}
    placed = np.zeros(num_nodes, dtype=bool)

    a0, b0, c0 = triangles[0]
    centres[a0] = 0j
    centres[b0] = complex(radii[a0] + radii[b0], 0.0)
    centres[c0] = cmath.rect(radii[a0] + radii[c0],
                             _packing_angle(radii[a0], radii[b0], radii[c0]))
    placed[a0] = placed[b0] = placed[c0] = True

    edge_queue = deque([(b0, a0), (c0, b0), (a0, c0)])
    processed_edges = set()

    while edge_queue:
        a, b = edge_queue.popleft()

        if (a, b) in processed_edges:
            continue
        processed_edges.add((a, b))

        c = half_edge_map.get((a, b))
        if c is None:
            continue

        if not placed[c]:
            d_ac = radii[a] + radii[c]
            d_bc = radii[b] + radii[c]

            # Measure a to b, not r_a + r_b: rounding drift must not break closure.
            vec_ab = centres[b] - centres[a]
            dist_ab = abs(vec_ab)

            if dist_ab > 1e-12:
                denom = 2.0 * d_ac * dist_ab
                if denom < 1e-12:
                    theta = math.pi / 3.0
                else:
                    cos_val = (d_ac * d_ac + dist_ab * dist_ab - d_bc * d_bc) / denom
                    cos_val = max(-1.0, min(1.0, cos_val))
                    theta = math.acos(cos_val)

                centres[c] = centres[a] + cmath.rect(d_ac, cmath.phase(vec_ab) + theta)
                placed[c] = True

        if placed[c]:
            edge_queue.append((c, b))
            edge_queue.append((a, c))

    positions = np.zeros((num_nodes, 2), dtype=float)
    for i, z in centres.items():
        positions[i] = (z.real, z.imag)

    return positions, placed

def _refine_tangency(positions, radii, edges, iterations, learning_rate=0.1):
    """Gradient descent pulling *edges* back to tangency, for the rounding the
    placement walk accumulates. Runs on the triangulation's edges: chasing the
    graph's edges alone would pull the packing apart along the chords."""
    if iterations <= 0 or len(edges) == 0:
        return positions

    ui, vi = edges[:, 0], edges[:, 1]
    target = radii[ui] + radii[vi]

    for _ in range(iterations):
        diff = positions[ui] - positions[vi]
        dist = np.linalg.norm(diff, axis=1)
        alive = dist > 1e-12
        if not alive.any():
            break

        error = np.zeros_like(dist)
        error[alive] = dist[alive] - target[alive]
        if np.abs(error).max() < 1e-9:
            break

        step = np.zeros_like(diff)
        step[alive] = (error[alive] / dist[alive])[:, None] * diff[alive]

        gradients = np.zeros_like(positions)
        np.add.at(gradients, ui, step)
        np.add.at(gradients, vi, -step)

        grad_norm = np.linalg.norm(gradients)
        if grad_norm < 1e-12:
            break

        # Damped by the gradient norm, so one bad node cannot dominate.
        positions -= learning_rate * gradients / (1.0 + 0.1 * grad_norm)

    return positions

def _circle_packing_layout(G, iterations=500, scale=5.0):
    """Circle packing by Collins-Stephenson (Koebe-Andreev-Thurston), returning
    ``(positions, radii)`` with Z = 0. Koebe needs a triangulated disk, so the
    planar embedding is completed to one and every graph edge ends up tangent.
    A non-planar graph, or a complex whose Euclidean packing crowds past double
    precision, goes force-directed instead."""
    import time

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

    nodes_list = list(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(nodes_list)}

    completed = _planar_triangulation(G, node_to_idx)
    if completed is None:
        print("  No triangulated disk available (non-planar) - force-directed fallback")
        return _circle_packing_force_directed(G, iterations, scale)

    triangles, outer_face = completed
    boundary_nodes = set(outer_face)

    triangles_at_vertex = [[] for _ in range(num_nodes)]
    for a, b, c in triangles:
        triangles_at_vertex[a].append((b, c))
        triangles_at_vertex[b].append((c, a))
        triangles_at_vertex[c].append((a, b))

    print(f"  Triangulated disk: {len(triangles)} faces, "
          f"{num_nodes - len(boundary_nodes)} interior, {len(boundary_nodes)} boundary")

    aims = _packing_aims(triangles_at_vertex, boundary_nodes)
    all_free = [i for i in range(num_nodes) if triangles_at_vertex[i]]
    interior_free = [i for i in all_free if i not in boundary_nodes]

    tolerance = 1e-9
    radius_sweeps = max(int(iterations), 1) * 20

    radii, max_error, sweeps = _solve_packing_radii(
        triangles_at_vertex, aims, all_free, radius_sweeps, tolerance)
    spread = float(radii.max() / max(radii.min(), 1e-300))

    embedded = max_error < tolerance and spread <= _MAX_RADIUS_SPREAD
    if embedded:
        print(f"  Radii converged in {sweeps} sweeps, spread={spread:.4g}")
    else:
        print(f"  Convex boundary crowds (error={max_error:.3e}, spread={spread:.4g}) "
              f"- pinning the boundary radii instead")
        radii, max_error, sweeps = _solve_packing_radii(
            triangles_at_vertex, aims, interior_free, radius_sweeps, tolerance)
        spread = float(radii.max() / max(radii.min(), 1e-300))
        print(f"  Pinned-boundary radii after {sweeps} sweeps: "
              f"error={max_error:.3e}, spread={spread:.4g}")

    positions, placed = _lay_out_packing(triangles, radii, num_nodes)

    unplaced = int((~placed).sum())
    if unplaced:
        print(f"  Warning: {unplaced} nodes unreachable in the placement walk")
        positions[~placed] = _get_layout_rng().uniform(-scale, scale, (unplaced, 2))

    tri_edges = set()
    for a, b, c in triangles:
        for x, y in ((a, b), (b, c), (c, a)):
            tri_edges.add((x, y) if x < y else (y, x))
    tri_edges = np.array(sorted(tri_edges), dtype=int)

    positions = _refine_tangency(positions, radii, tri_edges, max(int(iterations), 1))

    center = positions.mean(axis=0)
    positions -= center

    max_extent = np.max(np.abs(positions)) + np.max(radii)
    if max_extent > 1e-6:
        scale_factor = (scale * 0.45) / max_extent
        positions *= scale_factor
        radii *= scale_factor

    # Report against the original graph's edges, not the triangulated ones.
    max_tangency_error = 0.0
    for u, v in G.edges():
        if u == v:
            continue
        ui, vi = node_to_idx[u], node_to_idx[v]
        dist = np.linalg.norm(positions[ui] - positions[vi])
        error = abs(dist - (radii[ui] + radii[vi]))
        max_tangency_error = max(max_tangency_error, error)

    positions_3d = np.zeros((num_nodes, 3))
    positions_3d[:, :2] = positions

    elapsed = time.time() - start
    print(f"  Circle Packing completed in {elapsed:.2f}s")
    print(f"  Radii range: {radii.min():.4g} to {radii.max():.4g}")
    print(f"  Final max tangency error: {max_tangency_error:.3e}")
    if not embedded:
        print("  Circles may overlap: the boundary was pinned, so the packing "
              "condition holds only at interior vertices")

    return positions_3d, radii

def _close_pairs(positions, cutoff):
    """Index pairs closer than *cutoff*, through a k-d tree when SciPy is here."""
    if cKDTree is not None:
        return cKDTree(positions).query_pairs(cutoff, output_type='ndarray')

    n = len(positions)
    diff = positions[:, None, :] - positions[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    ui, vi = np.triu_indices(n, 1)
    keep = dist[ui, vi] < cutoff
    return np.column_stack((ui[keep], vi[keep]))

def _circle_packing_force_directed(G, iterations=500, scale=5.0):
    """Circle packing by force relaxation, for the graphs Collins-Stephenson
    cannot take: adjacent circles are pulled toward tangency and every
    overlapping pair is pushed apart, but neither is guaranteed to be reached,
    so expect residual overlap and tangency error."""
    import time

    start = time.time()
    num_nodes = len(G.nodes())
    nodes_list = list(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(nodes_list)}
    rng = _get_layout_rng()

    degrees = np.array([G.degree(n) for n in nodes_list], dtype=float)
    max_degree = max(degrees.max(), 1.0)

    radii = 0.3 + 0.7 * (degrees / max_degree)
    frame = scale * 0.45
    radii *= math.sqrt(0.35 * frame * frame / float(np.sum(radii * radii)))

    seed_iterations = max(10, min(50, 20000 // max(num_nodes, 1)))
    pos_dict = nx.spring_layout(G, dim=2, scale=frame, iterations=seed_iterations,
                                seed=get_layout_seed())
    positions = np.array([pos_dict[n] for n in nodes_list], dtype=float)

    edges = np.array([(node_to_idx[u], node_to_idx[v]) for u, v in G.edges()
                      if u != v], dtype=int).reshape(-1, 2)
    edge_keys = np.sort(np.minimum(edges[:, 0], edges[:, 1]) * num_nodes
                        + np.maximum(edges[:, 0], edges[:, 1]))

    temperature = scale * 0.2
    cooling_rate = 0.98
    cutoff = 2.2 * float(radii.max())

    for _ in range(iterations):
        forces = np.zeros((num_nodes, 2))

        if len(edges):
            delta = positions[edges[:, 1]] - positions[edges[:, 0]]
            dist = np.linalg.norm(delta, axis=1)
            weak = dist < 1e-6
            if weak.any():
                delta[weak] = rng.rand(int(weak.sum()), 2) - 0.5
                dist[weak] = np.linalg.norm(delta[weak], axis=1)
            target = radii[edges[:, 0]] + radii[edges[:, 1]]
            pull = ((dist - target) * 0.3 / dist)[:, None] * delta
            np.add.at(forces, edges[:, 0], pull)
            np.add.at(forces, edges[:, 1], -pull)

        pairs = _close_pairs(positions, cutoff)
        if len(pairs):
            ui, vi = pairs[:, 0], pairs[:, 1]
            delta = positions[vi] - positions[ui]
            dist = np.linalg.norm(delta, axis=1)
            weak = dist < 1e-6
            if weak.any():
                delta[weak] = rng.rand(int(weak.sum()), 2) - 0.5
                dist[weak] = np.linalg.norm(delta[weak], axis=1)

            sums = radii[ui] + radii[vi]
            unit = delta / dist[:, None]

            # Overlap is pushed apart whether or not the pair shares an edge.
            overlapping = dist < sums
            push = np.zeros_like(delta)
            push[overlapping] = ((sums - dist)[overlapping] * 0.5)[:, None] * unit[overlapping]

            keys = np.minimum(ui, vi) * num_nodes + np.maximum(ui, vi)
            idx = np.flatnonzero(~overlapping & ~np.isin(keys, edge_keys))
            repulsion = 0.1 * sums[idx] / (dist[idx] * dist[idx] + 0.1)
            push[idx] = repulsion[:, None] * unit[idx]

            np.add.at(forces, ui, -push)
            np.add.at(forces, vi, push)

        magnitude = np.linalg.norm(forces, axis=1)
        hot = magnitude > temperature
        if hot.any():
            forces[hot] *= (temperature / magnitude[hot])[:, None]
        positions += forces

        temperature *= cooling_rate

        if temperature < 1e-4:
            break

    for _ in range(max(int(iterations) // 2, 10)):
        pairs = _close_pairs(positions, cutoff)
        if not len(pairs):
            break

        ui, vi = pairs[:, 0], pairs[:, 1]
        delta = positions[vi] - positions[ui]
        dist = np.linalg.norm(delta, axis=1)
        weak = dist < 1e-6
        if weak.any():
            delta[weak] = rng.rand(int(weak.sum()), 2) - 0.5
            dist[weak] = np.linalg.norm(delta[weak], axis=1)

        sums = radii[ui] + radii[vi]
        hit = np.flatnonzero(dist < sums)
        if not len(hit):
            break

        shift = ((sums[hit] - dist[hit]) * 0.55 / dist[hit])[:, None] * delta[hit]
        np.add.at(positions, ui[hit], -shift)
        np.add.at(positions, vi[hit], shift)

    center = positions.mean(axis=0)
    positions -= center

    max_extent = np.max(np.abs(positions)) + np.max(radii)
    if max_extent > 1e-6:
        scale_factor = (scale * 0.45) / max_extent
        positions *= scale_factor
        radii *= scale_factor

    max_error = 0.0
    for u, v in G.edges():
        if u == v:
            continue
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
