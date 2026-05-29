"""3D network layer splitting helpers."""

from .common import *

def apply_network_splitter_3d(obj, criterion='COMMUNITY', attribute=None,
                               layer_height=2.0, layer_order='SIZE_DESC',
                               degree_bins=3, centrality_bins=3,
                               community_algorithm='RN', resolution=1.0,
                               preserve_xy=True, center_layers=False,
                               scale_by_size=False, base_z=0.0):
    """
    Split a network layout into distinct Z-layers based on the specified criterion.

    This is a post-processor that takes an existing 2D or 3D layout and separates
    nodes into vertical layers based on communities, attributes, degree, etc.

    Args:
        obj: Blender mesh object with graph data
        criterion: 'COMMUNITY', 'ATTRIBUTE', 'DEGREE', 'COMPONENT', 'CENTRALITY'
        attribute: Node attribute name (for ATTRIBUTE criterion)
        layer_height: Vertical distance between layers
        layer_order: How to order layers ('SIZE_ASC', 'SIZE_DESC', 'VALUE_ASC', 'VALUE_DESC', 'ALPHA')
        degree_bins: Number of bins for degree-based splitting
        centrality_bins: Number of bins for centrality-based splitting
        community_algorithm: 'CPM', 'INFOMAP', 'RB', 'RN', 'RNSC', 'SCLUSTER', 'UVCLUSTER'
        resolution: Resolution parameter for community detection
        preserve_xy: Keep original X,Y positions
        center_layers: Center each layer around its centroid
        scale_by_size: Scale layer extent by node count
        base_z: Starting Z coordinate

    Returns:
        Tuple (success: bool, num_layers: int, layer_info: dict)
    """
    import time
    start_time = time.time()

    G, num_nodes = _build_networkx_graph(obj)
    if G is None or num_nodes == 0:
        print("Network Splitter 3D: Failed to build graph")
        return False, 0, {}

    # Get current positions
    pos_flat = obj.get("node_positions", [])
    if not pos_flat:
        print("Network Splitter 3D: No node positions found")
        return False, 0, {}

    positions = np.array(pos_flat).reshape(-1, 3)

    # Determine layer assignments based on criterion
    if criterion == 'COMMUNITY':
        layer_assignments, layer_names = _split_by_community(
            G, community_algorithm, resolution
        )
    elif criterion == 'ATTRIBUTE':
        layer_assignments, layer_names = _split_by_attribute(obj, attribute)
    elif criterion == 'DEGREE':
        layer_assignments, layer_names = _split_by_degree(G, degree_bins)
    elif criterion == 'COMPONENT':
        layer_assignments, layer_names = _split_by_component(G)
    elif criterion == 'CENTRALITY':
        layer_assignments, layer_names = _split_by_centrality(G, centrality_bins)
    else:
        print(f"Network Splitter 3D: Unknown criterion '{criterion}'")
        return False, 0, {}

    if layer_assignments is None:
        return False, 0, {}

    # Order layers
    unique_layers = sorted(set(layer_assignments))
    layer_sizes = {l: layer_assignments.count(l) for l in unique_layers}

    if layer_order == 'SIZE_ASC':
        ordered_layers = sorted(unique_layers, key=lambda l: layer_sizes[l])
    elif layer_order == 'SIZE_DESC':
        ordered_layers = sorted(unique_layers, key=lambda l: -layer_sizes[l])
    elif layer_order == 'VALUE_ASC':
        ordered_layers = sorted(unique_layers)
    elif layer_order == 'VALUE_DESC':
        ordered_layers = sorted(unique_layers, reverse=True)
    elif layer_order == 'ALPHA':
        ordered_layers = sorted(unique_layers, key=lambda l: str(layer_names.get(l, l)))
    else:
        ordered_layers = unique_layers

    # Create layer index mapping
    layer_to_z_idx = {l: i for i, l in enumerate(ordered_layers)}

    # Apply Z positions
    new_positions = positions.copy()

    for node_idx in range(num_nodes):
        layer = layer_assignments[node_idx]
        z_idx = layer_to_z_idx[layer]
        z = base_z + z_idx * layer_height

        if preserve_xy:
            new_positions[node_idx, 2] = z
        else:
            new_positions[node_idx, 2] = z

    # Center layers if requested
    if center_layers:
        for layer in ordered_layers:
            layer_nodes = [i for i, l in enumerate(layer_assignments) if l == layer]
            if layer_nodes:
                centroid_x = np.mean([new_positions[i, 0] for i in layer_nodes])
                centroid_y = np.mean([new_positions[i, 1] for i in layer_nodes])
                for i in layer_nodes:
                    new_positions[i, 0] -= centroid_x
                    new_positions[i, 1] -= centroid_y

    # Scale by size if requested
    if scale_by_size:
        max_size = max(layer_sizes.values())
        for layer in ordered_layers:
            layer_nodes = [i for i, l in enumerate(layer_assignments) if l == layer]
            if layer_nodes:
                scale_factor = np.sqrt(layer_sizes[layer] / max_size)
                centroid_x = np.mean([new_positions[i, 0] for i in layer_nodes])
                centroid_y = np.mean([new_positions[i, 1] for i in layer_nodes])
                for i in layer_nodes:
                    new_positions[i, 0] = centroid_x + (new_positions[i, 0] - centroid_x) * scale_factor
                    new_positions[i, 1] = centroid_y + (new_positions[i, 1] - centroid_y) * scale_factor

    # Update positions
    obj["node_positions"] = new_positions.flatten().tolist()

    # Store layer info as custom properties
    obj["splitter_criterion"] = criterion
    obj["splitter_num_layers"] = len(ordered_layers)
    obj["splitter_layer_assignments"] = layer_assignments

    # Create layer info
    layer_info = {
        'num_layers': len(ordered_layers),
        'layer_sizes': layer_sizes,
        'layer_names': layer_names,
        'layer_order': [layer_names.get(l, str(l)) for l in ordered_layers],
    }

    # Store layer assignment as mesh attribute
    _store_layer_as_mesh_attribute(obj, layer_assignments, layer_to_z_idx)

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print("Network splitter 3D")
    print(f"Criterion: {criterion}")
    print(f"Layers created: {len(ordered_layers)}")
    for layer in ordered_layers:
        name = layer_names.get(layer, str(layer))
        print(f"  - Layer {layer_to_z_idx[layer]}: {name} ({layer_sizes[layer]} nodes)")
    print(f"Time: {elapsed:.3f}s")
    print(f"{'='*60}\n")

    return True, len(ordered_layers), layer_info

def _store_layer_as_mesh_attribute(obj, layer_assignments, layer_to_z_idx):
    """Store layer assignments as mesh attribute for Geometry Nodes."""
    if obj is None or obj.data is None:
        return

    mesh = obj.data
    attr_name = "layer_index"

    if attr_name in mesh.attributes:
        mesh.attributes.remove(mesh.attributes[attr_name])

    num_verts = len(mesh.vertices)

    # Map layer assignments to Z indices
    layer_values = [layer_to_z_idx.get(layer_assignments[i], 0)
                    if i < len(layer_assignments) else 0
                    for i in range(num_verts)]

    attr = mesh.attributes.new(name=attr_name, type='INT', domain='POINT')
    attr.data.foreach_set("value", layer_values)
    mesh.update()

def _split_by_community(G, algorithm='RN', resolution=1.0):
    """Split network by detected communities using pySurprise algorithms."""
    try:
        num_nodes = len(G.nodes())
        # Build integer edge list from networkx graph
        edges_int = list(G.edges())
        if not edges_int:
            return [0] * num_nodes, {0: "Community 0"}

        pysurprise_ok = False
        try:
            from pysurprise import algorithms as ps_algo
            pysurprise_ok = True
        except ImportError:
            pass

        if pysurprise_ok:
            from ..algorithms.analysis import _ensure_pysurprise_bin_permissions
            _ensure_pysurprise_bin_permissions(ps_algo._PKG_BIN_DIR)
            algo_key = algorithm.lower()
            algo_map = {
                'cpm': ps_algo.cpm,
                'infomap': ps_algo.infomap,
                'rb': ps_algo.rb,
                'rn': ps_algo.rn,
                'rnsc': ps_algo.rnsc,
                'scluster': ps_algo.scluster,
                'uvcluster': ps_algo.uvcluster,
            }
            func = algo_map.get(algo_key, ps_algo.rn)

            str_edges = [(str(u), str(v)) for u, v in edges_int]
            partition_dict = func(str_edges, timeout=300)

            if partition_dict:
                layer_assignments = [0] * num_nodes
                for str_node, comm_id in partition_dict.items():
                    idx = int(str_node)
                    if 0 <= idx < num_nodes:
                        layer_assignments[idx] = comm_id

                # Remap arbitrary community IDs to contiguous 0..N-1
                unique_ids = sorted(set(layer_assignments))
                remap = {old: new for new, old in enumerate(unique_ids)}
                layer_assignments = [remap[c] for c in layer_assignments]

                layer_names = {i: f"Community {i}" for i in set(layer_assignments)}
                return layer_assignments, layer_names

            raise RuntimeError(f"{algorithm} returned empty partition")

        # Fallback: networkx greedy_modularity
        from networkx.algorithms.community import greedy_modularity_communities
        communities = list(greedy_modularity_communities(G))
        partition = {}
        for i, comm in enumerate(communities):
            for node in comm:
                partition[node] = i
        layer_assignments = [partition.get(n, 0) for n in range(num_nodes)]

        layer_names = {i: f"Community {i}" for i in set(layer_assignments)}
        return layer_assignments, layer_names

    except Exception as e:
        print(f"Community detection error: {e}")
        return None, {}

def _split_by_attribute(obj, attribute):
    """Split network by node attribute values."""
    if not attribute or attribute == 'NONE':
        print("Network Splitter: No attribute specified")
        return None, {}

    mesh = obj.data

    if attribute not in mesh.attributes:
        print(f"Network Splitter: Attribute '{attribute}' not found")
        return None, {}

    attr = mesh.attributes[attribute]
    num_nodes = obj.get("num_nodes", len(mesh.vertices))

    # Read attribute values
    values = []
    for i in range(min(num_nodes, len(attr.data))):
        values.append(attr.data[i].value)

    # Extend if needed
    while len(values) < num_nodes:
        values.append(0)

    # Determine if categorical or numeric
    unique_values = sorted(set(values))

    if len(unique_values) <= 20:  # Treat as categorical
        value_to_layer = {v: i for i, v in enumerate(unique_values)}
        layer_assignments = [value_to_layer[v] for v in values]
        layer_names = {i: str(v) for v, i in value_to_layer.items()}
    else:  # Bin numeric values
        min_val, max_val = min(values), max(values)
        num_bins = min(10, len(unique_values))
        bin_width = (max_val - min_val) / num_bins if max_val > min_val else 1

        layer_assignments = []
        for v in values:
            bin_idx = int((v - min_val) / bin_width)
            bin_idx = min(bin_idx, num_bins - 1)
            layer_assignments.append(bin_idx)

        layer_names = {i: f"{min_val + i*bin_width:.2f} - {min_val + (i+1)*bin_width:.2f}"
                       for i in range(num_bins)}

    return layer_assignments, layer_names

def _split_by_degree(G, num_bins=3):
    """Split network by node degree ranges."""
    degrees = dict(G.degree())
    degree_values = [degrees[n] for n in range(len(G.nodes()))]

    if not degree_values:
        return None, {}

    min_deg, max_deg = min(degree_values), max(degree_values)

    if max_deg == min_deg:
        return [0] * len(degree_values), {0: f"Degree {min_deg}"}

    # Create bins
    bin_width = (max_deg - min_deg) / num_bins

    layer_assignments = []
    for d in degree_values:
        bin_idx = int((d - min_deg) / bin_width)
        bin_idx = min(bin_idx, num_bins - 1)
        layer_assignments.append(bin_idx)

    # Create descriptive names
    layer_names = {}
    for i in range(num_bins):
        low = int(min_deg + i * bin_width)
        high = int(min_deg + (i + 1) * bin_width)
        if i == 0:
            layer_names[i] = f"Low Degree ({low}-{high})"
        elif i == num_bins - 1:
            layer_names[i] = f"High Degree ({low}-{high})"
        else:
            layer_names[i] = f"Medium Degree ({low}-{high})"

    return layer_assignments, layer_names

def _split_by_component(G):
    """Split network by connected components."""
    components = list(nx.connected_components(G))

    # Sort by size (largest first by default)
    components = sorted(components, key=len, reverse=True)

    # Assign layer per component
    node_to_component = {}
    for comp_idx, comp in enumerate(components):
        for node in comp:
            node_to_component[node] = comp_idx

    layer_assignments = [node_to_component.get(n, 0) for n in range(len(G.nodes()))]
    layer_names = {i: f"Component {i} ({len(components[i])} nodes)"
                   for i in range(len(components))}

    return layer_assignments, layer_names

def _split_by_centrality(G, num_bins=3, method='betweenness'):
    """Split network by centrality score ranges."""
    try:
        centrality = nx.betweenness_centrality(G)
    except:
        centrality = nx.degree_centrality(G)

    centrality_values = [centrality[n] for n in range(len(G.nodes()))]

    if not centrality_values:
        return None, {}

    min_c, max_c = min(centrality_values), max(centrality_values)

    if max_c == min_c:
        return [0] * len(centrality_values), {0: "Equal Centrality"}

    bin_width = (max_c - min_c) / num_bins

    layer_assignments = []
    for c in centrality_values:
        bin_idx = int((c - min_c) / bin_width)
        bin_idx = min(bin_idx, num_bins - 1)
        layer_assignments.append(bin_idx)

    layer_names = {}
    for i in range(num_bins):
        if i == 0:
            layer_names[i] = "Low Centrality"
        elif i == num_bins - 1:
            layer_names[i] = "High Centrality"
        else:
            layer_names[i] = "Medium Centrality"

    return layer_assignments, layer_names

__all__ = [name for name in globals() if not name.startswith('__')]
