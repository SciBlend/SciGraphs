"""NetworkX-based layout algorithms."""

from .common import *
from .basic import _random_layout

_DENSE_EIG_LIMIT = 256

_EIG_RESIDUAL_TOL = 1e-2

_LOBPCG_MAXITER = 300

_MDS_PIVOTS = 100

_COMPONENT_SPACING = 2.5

def _spring_layout_2d(G, iterations, scale):
    pos_dict = nx.spring_layout(G, iterations=iterations, dim=2, scale=scale,
                                seed=get_layout_seed())

    positions = np.zeros((len(G.nodes()), 3))
    for node, (x, y) in pos_dict.items():
        positions[node] = [x, y, 0]

    return positions

def _spring_layout_3d(G, iterations, scale):
    pos_dict = nx.spring_layout(G, iterations=iterations, dim=3, scale=scale,
                                seed=get_layout_seed())

    positions = np.zeros((len(G.nodes()), 3))
    for node, (x, y, z) in pos_dict.items():
        positions[node] = [x, y, z]

    return positions

def _connected_component_indices(G):
    """Node list, plus the indices into it of each connected component."""
    nodes = list(G.nodes())
    order = {node: i for i, node in enumerate(nodes)}
    components = []
    for comp in nx.connected_components(G):
        idx = np.fromiter((order[n] for n in comp), dtype=int, count=len(comp))
        idx.sort()
        components.append(idx)
    return nodes, components

def _component_adjacency(G, nodes, idx):
    """Adjacency of one component, rows and columns ordered like *idx*."""
    member = [nodes[i] for i in idx]
    sub = G if len(idx) == len(nodes) else G.subgraph(member)
    return nx.adjacency_matrix(sub, nodelist=member).astype(float).tocsr()

def _eig_start_vector(n):
    """Fixed start vector, orthogonal to the constant null vector. Grids and
    trees have lambda2 == lambda3, and inside a degenerate eigenspace the
    solver returns whatever rotation its start vector lands on, so an arbitrary
    one makes the layout differ between runs."""
    i = np.arange(n, dtype=float)
    v = np.cos(i * 0.9124345) + np.sin(i * 0.3141593)
    v = v - v.mean()
    if not np.any(v):
        v = np.ones(n)
        v[0] = -1.0
    return v

def _fix_eigenvector_signs(vecs):
    """Eigenvector sign is arbitrary; pin it by the largest-magnitude entry."""
    for j in range(vecs.shape[1]):
        col = vecs[:, j]
        if col[np.abs(col).argmax()] < 0:
            col *= -1.0
    return vecs

def _eig_converged(L, values, vectors):
    """True when the returned pairs really satisfy L v = lambda v. Needed
    because a solver that stops on its iteration cap still returns numbers."""
    if not (np.all(np.isfinite(values)) and np.all(np.isfinite(vectors))):
        return False
    residual = np.linalg.norm(L @ vectors - vectors * values, axis=0).max()
    return residual <= _EIG_RESIDUAL_TOL * max(float(np.abs(values).max()), 1e-12)

def _laplacian_nontrivial_eigenvectors(L, dims):
    """The *dims* eigenvectors just above the null vector of a connected
    component's Laplacian, or None when nothing converged. ``which='SM'`` is not
    used: the small Laplacian eigenvalues sit in a tight cluster, so ARPACK
    restarts until its default cap of ``10 * n``, which on a 200k-node mesh
    means hours rather than a failure the caller can fall back from."""
    import warnings
    import scipy.sparse.linalg as spla

    n = L.shape[0]
    dims = min(dims, n - 1)
    if dims < 1:
        return np.zeros((n, 0))

    if n <= _DENSE_EIG_LIMIT:
        _, vectors = np.linalg.eigh(L.toarray())
        return _fix_eigenvector_signs(np.array(vectors[:, 1:1 + dims]))

    try:
        rng = np.random.RandomState(get_layout_seed())
        k = min(dims + 2, n - 1)
        X = rng.uniform(-1.0, 1.0, (n, k))
        X[:, 0] = _eig_start_vector(n)
        inv_diag = 1.0 / np.maximum(L.diagonal(), 1e-9)
        M = spla.LinearOperator((n, n), matvec=lambda x: (x.T * inv_diag).T)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            values, vectors = spla.lobpcg(L, X, M=M, Y=np.ones((n, 1)),
                                          largest=False,
                                          maxiter=_LOBPCG_MAXITER, tol=1e-6)
        order = np.argsort(values)[:dims]
        values, vectors = values[order], vectors[:, order]
        if _eig_converged(L, values, vectors):
            return _fix_eigenvector_signs(vectors)
        print("  LOBPCG eigensolver did not converge, trying shift-invert")
    except Exception as e:
        print(f"  LOBPCG eigensolver failed ({e}), trying shift-invert")

    try:
        values, vectors = spla.eigsh(L, k=dims + 1, sigma=-1e-3, which='LM',
                                     v0=_eig_start_vector(n), maxiter=300)
        order = np.argsort(values)
        values, vectors = values[order], vectors[:, order]
        if _eig_converged(L, values, vectors):
            return _fix_eigenvector_signs(vectors[:, 1:1 + dims])
        print("  Shift-invert eigensolver did not converge")
    except Exception as e:
        print(f"  Shift-invert eigensolver failed ({e})")

    return None

def _spectral_component_coordinates(G, dims):
    """*dims* spectral coordinates per node, solved one connected component at a
    time. The Laplacian null space holds one vector per component, so a
    whole-graph solve hands back component indicators and every component
    collapses to a single point. Returns ``(coords, components)``, or
    ``(None, None)`` when no component could be solved."""
    import scipy.sparse as sp

    nodes, components = _connected_component_indices(G)
    coords = np.zeros((len(nodes), dims))
    solved = False

    for idx in components:
        if len(idx) < 2:
            continue
        adj = _component_adjacency(G, nodes, idx)
        deg = np.array(adj.sum(axis=1)).flatten()
        L = (sp.diags(deg) - adj).tocsr()
        vectors = _laplacian_nontrivial_eigenvectors(L, dims)
        if vectors is None or vectors.shape[1] == 0:
            continue
        peak = np.abs(vectors).max()
        if peak > 0:
            vectors = vectors / peak
        coords[idx, :vectors.shape[1]] = vectors
        if vectors.shape[1] < dims:
            coords[idx, vectors.shape[1]:] = vectors[:, -1:]
        solved = True

    if not solved:
        return None, None
    return coords, components

def _pivot_mds_coordinates(adj, dims, num_pivots):
    """Pivot MDS for one connected component: hop distances to a few pivots,
    double centered, then the leading eigenvectors of the k by k matrix
    C.T @ C. O(k * (n + m)) time and O(n * k) memory, against O(n^3) and
    O(n^2) for classical MDS over the full distance matrix."""
    from scipy.sparse import csgraph

    n = adj.shape[0]
    k = int(min(num_pivots, n))

    dist = np.empty((n, k))
    chosen = 0
    covered = np.full(n, np.inf)
    for j in range(k):
        d = csgraph.dijkstra(adj, directed=False, unweighted=True, indices=chosen)
        d[~np.isfinite(d)] = 0.0
        dist[:, j] = d
        covered = np.minimum(covered, d)
        covered[chosen] = -1.0
        chosen = int(np.argmax(covered))

    np.square(dist, out=dist)
    col_mean = dist.mean(axis=0)
    row_mean = dist.mean(axis=1)
    grand_mean = float(col_mean.mean())
    dist -= col_mean[None, :]
    dist -= row_mean[:, None]
    dist += grand_mean
    dist *= -0.5

    _, vectors = np.linalg.eigh(dist.T @ dist)
    vectors = np.array(vectors[:, ::-1][:, :min(dims, k)])
    return _fix_eigenvector_signs(dist @ vectors)

def _pivot_mds_component_coordinates(G, dims, num_pivots):
    """Pivot MDS coordinates per node, one connected component at a time.
    Returns ``(coords, components)``."""
    nodes, components = _connected_component_indices(G)
    coords = np.zeros((len(nodes), dims))

    for idx in components:
        if len(idx) < 2:
            continue
        block = _pivot_mds_coordinates(_component_adjacency(G, nodes, idx),
                                       dims, num_pivots)
        peak = np.abs(block).max()
        if peak > 0:
            block = block / peak
        coords[idx, :block.shape[1]] = block

    return coords, components

def _pack_component_blocks(coords, components):
    """Spread the per-component blocks over a cubic lattice. Each block already
    fits a unit cube, so a spacing above its diagonal keeps components apart."""
    if len(components) < 2:
        return coords

    order = sorted(range(len(components)), key=lambda c: -len(components[c]))
    biggest = float(max(len(idx) for idx in components))
    side = max(1, int(np.ceil(len(components) ** (1.0 / 3.0))))
    offset = (side - 1) * _COMPONENT_SPACING * 0.5

    packed = np.array(coords)
    for slot, c in enumerate(order):
        idx = components[c]
        size = (len(idx) / biggest) ** (1.0 / 3.0)
        cell = np.array([slot % side, (slot // side) % side,
                         slot // (side * side)], dtype=float)
        packed[idx] = coords[idx] * size + cell * _COMPONENT_SPACING - offset
    return packed

def _rescale_positions(positions, scale):
    """Center on the origin and fit the cube of half-side *scale*, keeping
    *scale* a strict multiplier."""
    if positions.shape[0] == 0:
        return positions
    positions = positions - positions.mean(axis=0)
    peak = np.abs(positions).max()
    if peak > 0:
        positions = positions / peak
    return positions * scale

def _spectral_layout_3d(G, scale):
    import time
    start = time.time()
    print(f"Computing Spectral 3D layout for {len(G.nodes())} nodes...")

    num_nodes = len(G.nodes())

    # Fewer than four nodes leaves too few eigenvectors to place them.
    if num_nodes < 4:
        return _random_layout(num_nodes, scale)

    coords, components = _spectral_component_coordinates(G, 3)
    if coords is None:
        print("  Spectral 3D solved no component, using random layout")
        return _random_layout(num_nodes, scale)

    positions = _pack_component_blocks(coords, components)
    positions = _rescale_positions(positions, scale)

    print(f"  Spectral 3D completed in {time.time() - start:.2f}s")
    return positions

def _mds_layout_3d(G, scale):
    """Pivot MDS in 3D, over shortest-path distances. Components are placed
    separately: hop distance between them is undefined, and standing in a
    placeholder larger than any real distance lets the between-component term
    take the whole spread and collapses each component to a point."""
    import time

    start = time.time()
    print(f"Computing MDS 3D layout for {len(G.nodes())} nodes...")

    num_nodes = len(G.nodes())

    if num_nodes < 4:
        return _random_layout(num_nodes, scale)

    coords, components = _pivot_mds_component_coordinates(G, 3, _MDS_PIVOTS)
    positions = _pack_component_blocks(coords, components)
    positions = _rescale_positions(positions, scale)

    print(f"  MDS 3D completed in {time.time() - start:.2f}s")
    return positions

def _center_z(values):
    """Center on zero and scale the largest deviation to 0.5."""
    z = np.asarray(values, dtype=float)
    if z.size == 0:
        return z
    z = z - z.mean()
    max_abs = np.abs(z).max()
    if max_abs > 0:
        z = z / max_abs * 0.5
    return z

def _generate_z_component(G, num_nodes, method='SPECTRAL'):
    """Derive a Z coordinate per node from graph structure, so a 2D layout gets
    a third axis. *method* is DEGREE, BETWEENNESS, SPECTRAL or RANDOM, the
    middle two falling back to DEGREE; output has mean zero and stays within
    -0.5 to 0.5. A structureless graph, such as a regular one under DEGREE,
    gets zeros rather than raw values."""
    if method == 'DEGREE':
        degrees = np.array([G.degree(n) - 2 * int(G.has_edge(n, n))
                            for n in G.nodes()])
        return _center_z(degrees)

    elif method == 'BETWEENNESS':
        try:
            k = None if num_nodes <= 200 else 200
            bc = nx.betweenness_centrality(G, k=k, seed=get_layout_seed())
            return _center_z([bc.get(n, 0) for n in G.nodes()])
        except Exception as e:
            print(f"  Betweenness Z failed ({e}), using degree fallback")
            return _generate_z_component(G, num_nodes, 'DEGREE')

    elif method == 'SPECTRAL':
        try:
            coords, components = _spectral_component_coordinates(G, 2)
            if coords is None:
                raise ValueError("no component yielded an eigenvector")
            z = np.zeros(len(coords))
            for idx in components:
                block = coords[idx, -1]
                peak = np.abs(block).max()
                if peak > 0:
                    z[idx] = block / peak
            return _center_z(z)
        except Exception as e:
            print(f"  Spectral Z failed ({e}), using degree fallback")
            return _generate_z_component(G, num_nodes, 'DEGREE')

    elif method == 'RANDOM':
        rng = np.random.RandomState(42)
        return _center_z(rng.randn(num_nodes))

    return np.zeros(num_nodes)

__all__ = [name for name in globals() if not name.startswith('__')]
