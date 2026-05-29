# SuiteSparse Matrix Collection importer for SciGraphs
#
# Downloads sparse matrices from https://sparse.tamu.edu and converts
# them to GraphData for visualization in Blender.
# Supports both symmetric (A+A^T) and bipartite graph representations.

import numpy as np
import pandas as pd
import io
import re
from ..algorithms import graph
from ...utils.logger import log


# Base URL for the SuiteSparse Matrix Market archive
SUITESPARSE_MM_URL = "https://sparse.tamu.edu/MM"


def parse_matrix_id(identifier):
    """
    Parse a SuiteSparse matrix identifier into group and name.
    
    Accepts:
        - "Grund/bayer09"
        - "https://sparse.tamu.edu/Grund/bayer09"
        - "https://sparse.tamu.edu/MM/Grund/bayer09.tar.gz"
    
    Returns:
        (group, name) tuple or (None, None) if invalid.
    """
    identifier = identifier.strip()
    
    # Strip full URL prefixes
    for prefix in [
        "https://sparse.tamu.edu/MM/",
        "http://sparse.tamu.edu/MM/",
        "https://sparse.tamu.edu/",
        "http://sparse.tamu.edu/",
    ]:
        if identifier.lower().startswith(prefix.lower()):
            identifier = identifier[len(prefix):]
            break
    
    # Remove .tar.gz suffix
    identifier = re.sub(r'\.tar\.gz$', '', identifier)
    
    # Split into group/name
    parts = identifier.strip('/').split('/')
    if len(parts) == 2 and parts[0] and parts[1]:
        return parts[0], parts[1]
    
    return None, None


def download_matrix(group, name, timeout=120):
    """
    Download a SuiteSparse matrix in Matrix Market format.
    
    Returns:
        (mtx_text, coord_text) tuple. coord_text is None if no coordinates.
    """
    import requests
    import tarfile
    
    url = f"{SUITESPARSE_MM_URL}/{group}/{name}.tar.gz"
    log(f"Downloading {group}/{name} from SuiteSparse...")
    log(f"  URL: {url}")
    
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
    except Exception as e:
        log(f"  Download failed: {e}")
        return None, None
    
    log(f"  Downloaded {len(r.content) / 1024:.0f} KB")
    
    # Extract files from tar.gz
    mtx_text = None
    coord_text = None
    
    try:
        with tarfile.open(fileobj=io.BytesIO(r.content), mode='r:gz') as tar:
            for member in tar.getmembers():
                if member.name.endswith(f'/{name}.mtx'):
                    mtx_text = tar.extractfile(member).read().decode('utf-8')
                elif member.name.endswith(f'/{name}_coord.mtx'):
                    coord_text = tar.extractfile(member).read().decode('utf-8')
    except Exception as e:
        log(f"  Error extracting archive: {e}")
        return None, None
    
    if mtx_text is None:
        log(f"  Could not find {name}.mtx in archive")
        return None, None
    
    log(f"  Extracted MTX file ({len(mtx_text)} chars)")
    if coord_text:
        log(f"  Found coordinate file ({len(coord_text)} chars)")
    else:
        log(f"  No coordinate file found")
    
    return mtx_text, coord_text


def parse_mtx(mtx_text):
    """
    Parse Matrix Market format text.
    
    Returns:
        (nrows, ncols, entries, is_symmetric) where entries is list of (row, col) tuples (0-indexed).
    """
    lines = mtx_text.strip().split('\n')
    
    # Parse header
    header = lines[0].lower()
    is_symmetric = 'symmetric' in header
    is_pattern = 'pattern' in header
    
    # Skip comment lines
    idx = 1
    while idx < len(lines):
        s = lines[idx].strip()
        if s.startswith('%') or s == '':
            idx += 1
        else:
            break
    
    # Parse dimensions
    parts = lines[idx].split()
    nrows = int(parts[0])
    ncols = int(parts[1])
    nnz = int(parts[2]) if len(parts) > 2 else 0
    idx += 1
    
    # Parse entries
    entries = []
    for i in range(idx, len(lines)):
        line = lines[i].strip()
        if not line:
            continue
        p = line.split()
        if len(p) < 2:
            continue
        r_idx = int(p[0]) - 1  # Convert to 0-indexed
        c_idx = int(p[1]) - 1
        entries.append((r_idx, c_idx))
    
    log(f"  Matrix: {nrows} x {ncols}, {len(entries)} entries, symmetric={is_symmetric}")
    return nrows, ncols, entries, is_symmetric


def parse_coord_mtx(coord_text):
    """
    Parse a Matrix Market coordinate file (array format, column-major).
    
    The file has format:
        %%MatrixMarket matrix array real general
        N D
        val1   (row 0, col 0)
        val2   (row 1, col 0)
        ...    (all column 0 values, then all column 1, etc.)
    
    Returns:
        numpy array of shape (N, D) with coordinates, or None on failure.
    """
    if not coord_text:
        return None
    
    try:
        lines = coord_text.strip().split('\n')
        
        # Skip header and comments
        idx = 0
        while idx < len(lines):
            s = lines[idx].strip()
            if s.startswith('%') or s == '':
                idx += 1
            else:
                break
        
        # Dimensions: N rows, D columns
        parts = lines[idx].split()
        n_nodes = int(parts[0])
        n_dims = int(parts[1])
        idx += 1
        
        # Read all values (column-major order)
        values = []
        for i in range(idx, len(lines)):
            line = lines[i].strip()
            if line:
                values.append(float(line))
        
        expected = n_nodes * n_dims
        if len(values) != expected:
            log(f"  Warning: coord file has {len(values)} values, expected {expected}")
            return None
        
        # Reshape column-major to (N, D)
        coords = np.array(values).reshape((n_dims, n_nodes)).T
        
        log(f"  Coordinates: {n_nodes} nodes x {n_dims}D")
        return coords
    
    except Exception as e:
        log(f"  Error parsing coordinate file: {e}")
        return None


def build_symmetric_graph(nrows, ncols, entries, giant_only=True, coords=None):
    """
    Build symmetric (A+A^T) graph: nodes are row indices, edges are undirected.
    This treats the matrix as a standard adjacency matrix.
    
    Returns:
        GraphData object.
    """
    import time
    start = time.time()
    log("  Building symmetric (A+A^T) graph...")
    
    # Collect unique undirected edges (skip self-loops)
    edge_set = set()
    for r_i, c_i in entries:
        if r_i != c_i and r_i < nrows and c_i < nrows:
            edge_set.add((min(r_i, c_i), max(r_i, c_i)))
    
    # Find connected nodes
    connected = set()
    for u, v in edge_set:
        connected.add(u)
        connected.add(v)
    
    nodes = sorted(connected)
    edges = list(edge_set)
    
    if giant_only:
        nodes, edges = _extract_giant_component(nodes, edges)
    
    # Create DataFrame for compatibility
    df = pd.DataFrame(edges, columns=['source', 'target'])
    graph_data = graph.GraphData(nodes, edges, df)
    
    # Attach coordinates if available (symmetric uses row indices directly)
    if coords is not None:
        node_set = set(nodes)
        node_coords = {}
        for n in nodes:
            if isinstance(n, int) and n < len(coords):
                node_coords[n] = coords[n]
        if node_coords:
            graph_data.node_coordinates = node_coords
            log(f"  Attached coordinates for {len(node_coords)} nodes")
    
    log(f"  Symmetric graph: {len(nodes)} nodes, {len(edges)} edges ({time.time()-start:.2f}s)")
    return graph_data


def build_bipartite_graph(nrows, ncols, entries, giant_only=True, coords=None):
    """
    Build bipartite graph: row nodes (R0, R1, ...) connect to column nodes (C0, C1, ...).
    Preserves the original matrix structure and produces elongated layouts.
    
    Returns:
        GraphData object.
    """
    import time
    start = time.time()
    log("  Building bipartite graph...")
    
    # Build edge list with prefixed node names
    edge_set = set()
    row_nodes_used = set()
    col_nodes_used = set()
    
    for r_i, c_i in entries:
        r_name = f"R{r_i}"
        c_name = f"C{c_i}"
        edge_set.add((r_name, c_name))
        row_nodes_used.add(r_name)
        col_nodes_used.add(c_name)
    
    nodes = sorted(row_nodes_used) + sorted(col_nodes_used)
    edges = list(edge_set)
    
    if giant_only:
        nodes, edges = _extract_giant_component(nodes, edges)
    
    # Create DataFrame
    df = pd.DataFrame(edges, columns=['source', 'target'])
    graph_data = graph.GraphData(nodes, edges, df)
    
    # Attach coordinates if available
    # Row nodes (R0..Rn) use row coords, Column nodes (C0..Cn) also use row coords
    # (since for symmetric matrices rows==cols, coords apply to both)
    if coords is not None:
        node_coords = {}
        for n in nodes:
            if isinstance(n, str) and len(n) > 1:
                idx = int(n[1:])
                if idx < len(coords):
                    node_coords[n] = coords[idx]
        if node_coords:
            graph_data.node_coordinates = node_coords
            log(f"  Attached coordinates for {len(node_coords)}/{len(nodes)} nodes")
    
    log(f"  Bipartite graph: {len(nodes)} nodes, {len(edges)} edges ({time.time()-start:.2f}s)")
    return graph_data


def _extract_giant_component(nodes, edges):
    """
    Extract the largest connected component from a graph.
    Uses efficient union-find for large graphs.
    """
    if not edges:
        return nodes, edges
    
    # Union-Find
    parent = {}
    rank = {}
    
    def find(x):
        if x not in parent:
            parent[x] = x
            rank[x] = 0
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # Path compression
            x = parent[x]
        return x
    
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        if rank[ra] == rank[rb]:
            rank[ra] += 1
    
    # Initialize all nodes
    for n in nodes:
        find(n)
    
    # Union edges
    for u, v in edges:
        union(u, v)
    
    # Find component sizes
    comp_map = {}
    for n in nodes:
        root = find(n)
        if root not in comp_map:
            comp_map[root] = []
        comp_map[root].append(n)
    
    # Get giant component
    giant_root = max(comp_map, key=lambda k: len(comp_map[k]))
    giant_nodes = set(comp_map[giant_root])
    
    num_components = len(comp_map)
    giant_size = len(giant_nodes)
    total_nodes = len(nodes)
    
    if num_components > 1:
        log(f"    {num_components} connected components found")
        log(f"    Giant component: {giant_size}/{total_nodes} nodes ({100*giant_size/total_nodes:.1f}%)")
    
    # Filter
    filtered_nodes = [n for n in nodes if n in giant_nodes]
    filtered_edges = [(u, v) for u, v in edges if u in giant_nodes and v in giant_nodes]
    
    return filtered_nodes, filtered_edges


def load_suitesparse_graph(identifier, mode='BIPARTITE', giant_only=True):
    """
    Main entry point: download a SuiteSparse matrix and build a graph.
    
    Args:
        identifier: "Group/Name" or full URL
        mode: 'SYMMETRIC' for A+A^T, 'BIPARTITE' for row↔col bipartite graph
        giant_only: If True, keep only the largest connected component
    
    Returns:
        GraphData object or None on failure.
    """
    import time
    start = time.time()
    
    group, name = parse_matrix_id(identifier)
    if not group or not name:
        log(f"Error: Invalid SuiteSparse identifier: '{identifier}'")
        log(f"  Expected format: 'Group/Name' (e.g. 'Grund/bayer09')")
        return None
    
    # Download
    mtx_text, coord_text = download_matrix(group, name)
    if mtx_text is None:
        return None
    
    # Parse matrix
    nrows, ncols, entries, is_symmetric = parse_mtx(mtx_text)
    
    if not entries:
        log("Error: No entries found in matrix")
        return None
    
    # Parse coordinates if available
    coords = parse_coord_mtx(coord_text)
    
    # Build graph
    if mode == 'BIPARTITE':
        graph_data = build_bipartite_graph(nrows, ncols, entries, giant_only=giant_only, coords=coords)
    else:
        graph_data = build_symmetric_graph(nrows, ncols, entries, giant_only=giant_only, coords=coords)
    
    # Store metadata
    graph_data.source_column_name = 'source'
    graph_data.target_column_name = 'target'
    graph_data.suitesparse_group = group
    graph_data.suitesparse_name = name
    graph_data.suitesparse_mode = mode
    graph_data.matrix_rows = nrows
    graph_data.matrix_cols = ncols
    graph_data.has_coordinates = coords is not None
    
    log(f"  SuiteSparse import complete in {time.time()-start:.2f}s")
    return graph_data
