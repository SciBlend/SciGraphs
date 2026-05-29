"""NetworkX-based layout algorithms."""

from .common import *
from .basic import _random_layout

def _spring_layout_2d(G, iterations, scale):
    """Force-directed layout in 2D using NetworkX."""
    pos_dict = nx.spring_layout(G, iterations=iterations, dim=2, scale=scale)

    positions = np.zeros((len(G.nodes()), 3))
    for node, (x, y) in pos_dict.items():
        positions[node] = [x, y, 0]

    return positions

def _spring_layout_3d(G, iterations, scale):
    """Force-directed layout in 3D using NetworkX."""
    pos_dict = nx.spring_layout(G, iterations=iterations, dim=3, scale=scale)

    positions = np.zeros((len(G.nodes()), 3))
    for node, (x, y, z) in pos_dict.items():
        positions[node] = [x, y, z]

    return positions

def _spectral_layout_3d(G, scale):
    """
    Use graph Laplacian eigenvectors for 3D positioning.
    Places nodes based on spectral decomposition of the graph.
    """
    import time
    start = time.time()
    print(f"Computing Spectral 3D layout for {len(G.nodes())} nodes...")

    num_nodes = len(G.nodes())

    if num_nodes < 4:
        # Not enough nodes for spectral layout, use random
        return _random_layout(num_nodes, scale)

    pos_dict = nx.spectral_layout(G, dim=3, scale=scale)

    positions = np.zeros((num_nodes, 3))
    for node, (x, y, z) in pos_dict.items():
        positions[node] = [x, y, z]

    print(f"  Spectral 3D completed in {time.time() - start:.2f}s")
    return positions

def _mds_layout_3d(G, scale):
    """
    Multidimensional Scaling (MDS) layout in 3D.
    Uses shortest path distances to position nodes.
    """
    import time
    from scipy.spatial.distance import squareform
    from scipy.linalg import eigh

    start = time.time()
    print(f"Computing MDS 3D layout for {len(G.nodes())} nodes...")

    num_nodes = len(G.nodes())

    if num_nodes < 4:
        return _random_layout(num_nodes, scale)

    # Compute shortest path distances
    path_lengths = dict(nx.all_pairs_shortest_path_length(G))

    # Build distance matrix
    dist_matrix = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i in path_lengths and j in path_lengths[i]:
                dist_matrix[i, j] = path_lengths[i][j]
            else:
                # Disconnected nodes: use large distance
                dist_matrix[i, j] = num_nodes * 2

    # Classical MDS
    # Center the distance matrix
    n = dist_matrix.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n

    # Apply double centering
    B = -0.5 * H @ (dist_matrix ** 2) @ H

    # Eigenvalue decomposition
    eigenvalues, eigenvectors = eigh(B)

    # Sort by eigenvalues (descending)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Take top 3 dimensions
    eigenvalues = np.maximum(eigenvalues[:3], 0)  # Ensure non-negative
    positions = eigenvectors[:, :3] @ np.diag(np.sqrt(eigenvalues))

    # Normalize and scale
    positions = positions - positions.mean(axis=0)
    std = positions.std()
    if std > 0:
        positions = positions / std * scale
    else:
        positions = positions * scale

    print(f"  MDS 3D completed in {time.time() - start:.2f}s")
    return positions

def _generate_z_component(G, num_nodes, method='SPECTRAL'):
    """Generate Z coordinates from graph structure for 2D layouts."""
    if method == 'DEGREE':
        degrees = np.array([G.degree(n) for n in G.nodes()])
        z = degrees.astype(float)
        if z.max() > z.min():
            z = (z - z.min()) / (z.max() - z.min()) - 0.5
        return z

    elif method == 'BETWEENNESS':
        try:
            bc = nx.betweenness_centrality(G, k=min(200, num_nodes))
            z = np.array([bc.get(n, 0) for n in G.nodes()])
            if z.max() > z.min():
                z = (z - z.min()) / (z.max() - z.min()) - 0.5
            return z
        except Exception as e:
            print(f"  Betweenness Z failed ({e}), using degree fallback")
            return _generate_z_component(G, num_nodes, 'DEGREE')

    elif method == 'SPECTRAL':
        try:
            import scipy.sparse as sp
            import scipy.sparse.linalg as spla

            adj = nx.adjacency_matrix(G).astype(float)
            deg = np.array(adj.sum(axis=1)).flatten()
            D = sp.diags(deg)
            L = D - adj

            k_eig = min(4, num_nodes - 1)
            eigenvalues, eigenvectors = spla.eigsh(L, k=k_eig, which='SM')
            idx = min(2, eigenvectors.shape[1] - 1)
            z = eigenvectors[:, idx]
            z = z - z.mean()
            max_abs = np.abs(z).max()
            if max_abs > 0:
                z = z / max_abs * 0.5
            return z
        except Exception as e:
            print(f"  Spectral Z failed ({e}), using degree fallback")
            return _generate_z_component(G, num_nodes, 'DEGREE')

    elif method == 'RANDOM':
        rng = np.random.RandomState(42)
        z = rng.randn(num_nodes)
        z = z - z.mean()
        max_abs = np.abs(z).max()
        if max_abs > 0:
            z = z / max_abs * 0.5
        return z

    return np.zeros(num_nodes)

__all__ = [name for name in globals() if not name.startswith('__')]
