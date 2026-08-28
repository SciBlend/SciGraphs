"""3D network layer splitting helpers."""

from .common import *

_BETWEENNESS_EXACT_WORK = 200_000_000
_BETWEENNESS_EXACT_MAX = 1000
_BETWEENNESS_PIVOTS = 200

_CUSTOM_MAX_LAYERS = 100

def apply_network_splitter_3d(obj, criterion='COMMUNITY', attribute=None,
                               layer_height=2.0, layer_order='SIZE_DESC',
                               degree_bins=3, centrality_bins=3,
                               community_algorithm='LEIDEN', resolution=1.0,
                               preserve_xy=True, center_layers=False,
                               scale_by_size=False, base_z=0.0,
                               edge_pairs=None, custom_expression=None):
    """Stack an existing layout into Z-layers, as ``(success, num_layers,
    layer_info)``. *criterion*: 'COMMUNITY', 'ATTRIBUTE', 'DEGREE', 'COMPONENT',
    'CENTRALITY', 'CUSTOM'. *layer_order*: 'SIZE_ASC', 'SIZE_DESC', 'VALUE_ASC',
    'VALUE_DESC', 'ALPHA'. *community_algorithm*: 'LEIDEN', 'LOUVAIN', 'CPM',
    'INFOMAP', 'RB', 'RN', 'RNSC', 'SCLUSTER', 'UVCLUSTER'; only LEIDEN and
    LOUVAIN read *resolution*. *preserve_xy* False replaces XY with a per-layer
    radial spread, hubs at each layer's centre. *custom_expression* is required
    by CUSTOM and is evaluated under a restricted grammar, never as free Python.

    *edge_pairs* carries a mesh-native object's topology. Every criterion but
    ATTRIBUTE and CUSTOM reads the edges, and an edgeless graph does not fail:
    COMPONENT reports n singleton layers, COMMUNITY, DEGREE and CENTRALITY
    report a single flat layer, and all of them report success."""
    import time
    start_time = time.time()

    G, num_nodes = _build_networkx_graph(obj, edge_pairs)
    if G is None or num_nodes == 0:
        print("Network Splitter 3D: Failed to build graph")
        return False, 0, {}

    pos_flat = obj.get("node_positions", [])
    if not pos_flat:
        print("Network Splitter 3D: No node positions found")
        return False, 0, {}

    positions = np.array(pos_flat).reshape(-1, 3)

    if G.number_of_edges() == 0 and criterion in ('COMMUNITY', 'DEGREE', 'CENTRALITY'):
        print(f"Network Splitter 3D: the graph has no edges, so {criterion} "
              "has nothing to separate and puts every node on one flat layer")

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
    elif criterion == 'CUSTOM':
        layer_assignments, layer_names = _split_by_custom(G, custom_expression)
    else:
        print(f"Network Splitter 3D: Unknown criterion '{criterion}'")
        return False, 0, {}

    if layer_assignments is None:
        return False, 0, {}

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

    layer_to_z_idx = {l: i for i, l in enumerate(ordered_layers)}

    if preserve_xy:
        new_positions = positions.copy()
    else:
        new_positions = _radial_layer_positions(
            G, positions, layer_assignments, ordered_layers
        )

    for node_idx in range(num_nodes):
        z_idx = layer_to_z_idx[layer_assignments[node_idx]]
        new_positions[node_idx, 2] = base_z + z_idx * layer_height

    if center_layers:
        for layer in ordered_layers:
            layer_nodes = [i for i, l in enumerate(layer_assignments) if l == layer]
            if layer_nodes:
                centroid_x = np.mean([new_positions[i, 0] for i in layer_nodes])
                centroid_y = np.mean([new_positions[i, 1] for i in layer_nodes])
                for i in layer_nodes:
                    new_positions[i, 0] -= centroid_x
                    new_positions[i, 1] -= centroid_y

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

    obj["node_positions"] = new_positions.flatten().tolist()

    obj["splitter_criterion"] = criterion
    obj["splitter_num_layers"] = len(ordered_layers)
    obj["splitter_layer_assignments"] = layer_assignments
    obj["splitter_layer_index"] = [layer_to_z_idx[l] for l in layer_assignments]

    layer_info = {
        'num_layers': len(ordered_layers),
        'layer_sizes': layer_sizes,
        'layer_names': layer_names,
        'layer_order': [layer_names.get(l, str(l)) for l in ordered_layers],
    }

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

def _radial_layer_positions(G, positions, layer_assignments, ordered_layers):
    """Give every layer its own sunflower disc, highest degree at the centre.

    This is what ``preserve_xy=False`` means: the input XY is discarded so the
    layers read as discrete plates instead of one silhouette seen edge-on."""
    xy = positions[:, :2]
    radius = float(np.max(np.abs(xy - xy.mean(axis=0)))) if len(xy) else 0.0
    if not radius > 0:
        radius = 1.0

    degrees = dict(G.degree())
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))

    out = positions.copy()
    for layer in ordered_layers:
        members = [i for i, l in enumerate(layer_assignments) if l == layer]
        members.sort(key=lambda i: (-degrees.get(i, 0), i))
        span = max(len(members) - 1, 1)
        for rank, node_idx in enumerate(members):
            r = radius * np.sqrt(rank / span)
            theta = rank * golden_angle
            out[node_idx, 0] = r * np.cos(theta)
            out[node_idx, 1] = r * np.sin(theta)
    return out

def _store_layer_as_mesh_attribute(obj, layer_assignments, layer_to_z_idx):
    """Write layer assignments to a ``layer_index`` point attribute.

    Blender-only, inside a package core/pyproject.toml declares Blender-free.
    Kept only so the attribute does not vanish before the add-on builds it from
    ``obj["splitter_layer_index"]``, the way circle packing does with
    ``circle_radius``; delete it once the operator does that. The caller has
    already stored node_positions, so a failure here costs the attribute and not
    the split."""
    mesh = getattr(obj, "data", None)
    if mesh is None:
        return False

    try:
        attr_name = "layer_index"

        if attr_name in mesh.attributes:
            mesh.attributes.remove(mesh.attributes[attr_name])

        num_verts = len(mesh.vertices)

        layer_values = [layer_to_z_idx.get(layer_assignments[i], 0)
                        if i < len(layer_assignments) else 0
                        for i in range(num_verts)]

        attr = mesh.attributes.new(name=attr_name, type='INT', domain='POINT')
        attr.data.foreach_set("value", layer_values)
        mesh.update()
        return True
    except Exception as e:
        print(f"Network Splitter 3D: could not write the 'layer_index' mesh "
              f"attribute ({e}); the split itself is in node_positions")
        return False

def _split_by_community(G, algorithm='LEIDEN', resolution=1.0):
    """Detect communities with Leiden/Louvain, pySurprise, or NetworkX greedy.

    Every method here reads only the edges, so nodes with no neighbour cannot be
    placed and go to layer -1, "Unassigned", rather than passing for members of
    community 0."""
    try:
        num_nodes = len(G.nodes())
        isolated = {n for n in range(num_nodes)
                    if not (set(G.neighbors(n)) - {n})}
        edges_int = [(u, v) for u, v in G.edges() if u != v]
        if not edges_int:
            return [0] * num_nodes, {0: "Community 0"}

        algo_key = algorithm.lower()

        if algo_key in ('leiden', 'louvain'):
            partition = _modularity_partition(G, algo_key, resolution)
        else:
            partition = _pysurprise_partition(G, algo_key, edges_int, resolution)

        if partition is None:
            print(f"Network Splitter: {algo_key.upper()} is unavailable, "
                  "detecting communities with Leiden instead")
            partition = _modularity_partition(G, 'leiden', resolution)

        return _finalize_communities(partition, num_nodes, isolated)

    except Exception as e:
        print(f"Community detection error: {e}")
        return None, {}

def _modularity_partition(G, algo_key, resolution):
    """Leiden through igraph, else Louvain; both honour *resolution*.

    Gephi's Modularity and igraph both default to this family, and unlike the
    pySurprise algorithms they answer the resolution knob the UI advertises:
    karate gives 1 community at 0.1, 4 at 1.0 and 17-18 at 5.0."""
    from ...repro.determinism import get_seed_context
    seed = get_layout_seed()

    if algo_key == 'leiden':
        if IGRAPH_AVAILABLE:
            g_igraph = _nx_to_igraph(G)
            with get_seed_context(seed):
                clusters = g_igraph.community_leiden(
                    objective_function="modularity",
                    resolution=resolution,
                    n_iterations=10,
                )
            return {n: i for i, comm in enumerate(clusters) for n in comm}
        print("Network Splitter: python-igraph is missing, running Louvain "
              "instead of Leiden")

    if IGRAPH_AVAILABLE:
        g_igraph = _nx_to_igraph(G)
        with get_seed_context(seed):
            clusters = g_igraph.community_multilevel(resolution=resolution)
        return {n: i for i, comm in enumerate(clusters) for n in comm}

    from networkx.algorithms.community import louvain_communities
    communities = louvain_communities(G, resolution=resolution, seed=seed)
    return {n: i for i, comm in enumerate(communities) for n in comm}

def _pysurprise_partition(G, algo_key, edges_int, resolution):
    """pySurprise partition as ``{node_index: community}``, None if unavailable."""
    try:
        from pysurprise import algorithms as ps_algo
    except ImportError:
        return None

    # Three dots, not two: this module is
    # SciGraphs.core.mesh.layouts.splitter, so `..` lands on
    # SciGraphs.core.mesh, which has no `algorithms`.
    from ...algorithms.analysis import _ensure_pysurprise_bin_permissions
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
    func = algo_map.get(algo_key)
    if func is None:
        print(f"Network Splitter: unknown community algorithm '{algo_key}', "
              "using RN")
        func = ps_algo.rn

    if abs(resolution - 1.0) > 1e-9:
        print(f"Network Splitter: {algo_key.upper()} has no resolution "
              f"parameter, so resolution={resolution} is ignored; pick LEIDEN "
              "or LOUVAIN to control the number of communities")

    str_edges = [(str(u), str(v)) for u, v in edges_int]
    partition_dict = func(str_edges, timeout=300)

    if not partition_dict:
        raise RuntimeError(f"{algo_key} returned empty partition")

    partition = {}
    for str_node, comm_id in partition_dict.items():
        idx = int(str_node)
        if 0 <= idx < len(G.nodes()):
            partition[idx] = comm_id
    return partition

def _finalize_communities(partition, num_nodes, isolated):
    """Renumber a partition 0-based and route unplaceable nodes to layer -1."""
    layer_assignments = [-1 if n in isolated else partition.get(n, -1)
                         for n in range(num_nodes)]

    unique_ids = sorted({c for c in layer_assignments if c != -1})
    remap = {old: new for new, old in enumerate(unique_ids)}
    layer_assignments = [c if c == -1 else remap[c] for c in layer_assignments]

    layer_names = {i: f"Community {i}" for i in remap.values()}
    num_unassigned = layer_assignments.count(-1)
    if num_unassigned:
        plural = "node" if num_unassigned == 1 else "nodes"
        layer_names[-1] = f"Unassigned ({num_unassigned} {plural})"
    return layer_assignments, layer_names

def _split_by_attribute(obj, attribute):
    if not attribute or attribute == 'NONE':
        print("Network Splitter: No attribute specified")
        return None, {}

    mesh = getattr(obj, "data", None)
    if mesh is None:
        print("Network Splitter: ATTRIBUTE needs a Blender mesh datablock and "
              "this object has none")
        return None, {}

    try:
        if attribute not in mesh.attributes:
            print(f"Network Splitter: Attribute '{attribute}' not found")
            return None, {}

        attr = mesh.attributes[attribute]
        num_nodes = obj.get("num_nodes", len(mesh.vertices))

        values = []
        for i in range(min(num_nodes, len(attr.data))):
            values.append(attr.data[i].value)
    except Exception as e:
        print(f"Network Splitter: could not read attribute '{attribute}' ({e})")
        return None, {}

    while len(values) < num_nodes:
        values.append(0)

    unique_values = sorted(set(values))

    # Twenty or fewer distinct values reads as categorical, more as numeric.
    if len(unique_values) <= 20:
        value_to_layer = {v: i for i, v in enumerate(unique_values)}
        layer_assignments = [value_to_layer[v] for v in values]
        layer_names = {i: str(v) for v, i in value_to_layer.items()}
    else:
        num_bins = min(10, len(unique_values))
        layer_assignments, bin_values = _quantile_bins(values, num_bins)
        layer_names = {i: f"{lo:.2f} - {hi:.2f}"
                       for i, (lo, hi) in bin_values.items()}

    return layer_assignments, layer_names

def _quantile_bins(values, num_bins):
    """Rank-based bins as ``(assignments, {bin: (min, max)})``.

    Equal-width bins over [min, max] collapse on the heavy tails these layouts
    actually meet: karate degree in 3 bins gives 29/2/3 and BA(5000,3) gives
    4992/6/2. Splitting on equal node mass instead gives 12/12/10 and
    2010/1609/1381. Equal values always share a bin, and empty bins are dropped
    so the layer count never overstates the spread in the data."""
    unique_values = sorted(set(values))

    if len(unique_values) <= num_bins:
        bin_of_value = {v: i for i, v in enumerate(unique_values)}
    else:
        counts = {}
        for v in values:
            counts[v] = counts.get(v, 0) + 1

        target = len(values) / float(num_bins)
        bin_of_value = {}
        current_bin = 0
        cumulative = 0
        for i, v in enumerate(unique_values):
            remaining_values = len(unique_values) - i
            if (current_bin < num_bins - 1
                    and cumulative >= target * (current_bin + 1)
                    and remaining_values >= num_bins - current_bin - 1):
                current_bin += 1
            bin_of_value[v] = current_bin
            cumulative += counts[v]

    assignments = [bin_of_value[v] for v in values]

    bin_values = {}
    for v, b in bin_of_value.items():
        lo, hi = bin_values.get(b, (v, v))
        bin_values[b] = (min(lo, v), max(hi, v))
    return assignments, bin_values

def _split_by_degree(G, num_bins=3):
    degrees = dict(G.degree())
    degree_values = [degrees[n] for n in range(len(G.nodes()))]

    if not degree_values:
        return None, {}

    layer_assignments, bin_values = _quantile_bins(degree_values, num_bins)
    return layer_assignments, _range_layer_names(bin_values, "Degree", int)

def _range_layer_names(bin_values, noun, cast):
    """Low/Medium/High names carrying each bin's real value range."""
    num_bins = len(bin_values)
    layer_names = {}
    for i, (lo, hi) in bin_values.items():
        span = f"{cast(lo)}" if lo == hi else f"{cast(lo)}-{cast(hi)}"
        if num_bins == 1:
            layer_names[i] = f"{noun} {span}"
        elif i == 0:
            layer_names[i] = f"Low {noun} ({span})"
        elif i == num_bins - 1:
            layer_names[i] = f"High {noun} ({span})"
        else:
            layer_names[i] = f"Medium {noun} ({span})"
    return layer_names

def _split_by_component(G):
    """One layer per connected component, largest first."""
    components = list(nx.connected_components(G))

    components = sorted(components, key=len, reverse=True)

    node_to_component = {}
    for comp_idx, comp in enumerate(components):
        for node in comp:
            node_to_component[node] = comp_idx

    layer_assignments = [node_to_component.get(n, 0) for n in range(len(G.nodes()))]
    layer_names = {i: f"Component {i} ({len(components[i])} nodes)"
                   for i in range(len(components))}

    return layer_assignments, layer_names

def _igraph_betweenness(G, num_nodes, pivots=None):
    """Betweenness on networkx's normalized scale, computed by igraph.

    igraph returns raw path counts and networkx divides an undirected graph's by
    (n-1)(n-2)/2; *pivots* starts the paths at that many sources only, which
    networkx rescales by n/k on top. Exact matches nx.betweenness_centrality to
    7e-18 and is 23x faster; the pivot estimate scores the same Spearman 0.90
    against exact as networkx's own k= sampling."""
    if num_nodes <= 2:
        return [0.0] * num_nodes

    g_igraph = _nx_to_igraph(G)
    denom = (num_nodes - 1) * (num_nodes - 2) / 2.0

    if pivots is None:
        raw = g_igraph.betweenness()
    else:
        sources = sorted(random.Random(get_layout_seed()).sample(
            range(num_nodes), pivots))
        raw = g_igraph.betweenness(sources=sources)
        denom *= pivots / float(num_nodes)

    return [v / denom for v in raw]

def _split_by_centrality(G, num_bins=3, method='betweenness'):
    """Bin nodes into *num_bins* centrality ranges, betweenness or degree.

    Exact betweenness is O(nm) however it is computed, so it runs only while the
    node-edge product stays under _BETWEENNESS_EXACT_WORK; n=10000/m=30000 sits
    just over it at 7.2 s. Past that it estimates from seeded pivots, 3.5 s at
    n=100000, and both paths are reproducible run to run."""
    num_nodes = len(G.nodes())
    num_edges = G.number_of_edges()
    try:
        if method != 'betweenness':
            centrality = nx.degree_centrality(G)
            centrality_values = [centrality[n] for n in range(num_nodes)]
        elif IGRAPH_AVAILABLE:
            if num_nodes * num_edges <= _BETWEENNESS_EXACT_WORK:
                centrality_values = _igraph_betweenness(G, num_nodes)
            else:
                pivots = min(_BETWEENNESS_PIVOTS, num_nodes)
                print(f"Network Splitter: {num_nodes} nodes and {num_edges} "
                      f"edges, estimating betweenness from {pivots} seeded "
                      "pivots instead of all shortest paths")
                centrality_values = _igraph_betweenness(G, num_nodes, pivots)
        else:
            if num_nodes > _BETWEENNESS_EXACT_MAX:
                pivots = min(_BETWEENNESS_PIVOTS, num_nodes)
                print(f"Network Splitter: python-igraph is missing, so "
                      f"{num_nodes} nodes go through networkx, estimating "
                      f"betweenness from {pivots} seeded pivots")
                centrality = nx.betweenness_centrality(G, k=pivots,
                                                       seed=get_layout_seed())
            else:
                centrality = nx.betweenness_centrality(G)
            centrality_values = [centrality[n] for n in range(num_nodes)]
    except Exception as e:
        print(f"Network Splitter: betweenness failed ({e}), falling back to "
              "degree centrality")
        centrality = nx.degree_centrality(G)
        centrality_values = [centrality[n] for n in range(num_nodes)]

    if not centrality_values:
        return None, {}

    layer_assignments, bin_values = _quantile_bins(centrality_values, num_bins)
    layer_names = {}
    for i in bin_values:
        if len(bin_values) == 1:
            layer_names[i] = "Equal Centrality"
        elif i == 0:
            layer_names[i] = "Low Centrality"
        elif i == len(bin_values) - 1:
            layer_names[i] = "High Centrality"
        else:
            layer_names[i] = "Medium Centrality"

    return layer_assignments, layer_names

_CUSTOM_NODES = (
    'Expression', 'BoolOp', 'BinOp', 'UnaryOp', 'IfExp', 'Compare', 'Call',
    'Constant', 'Name', 'Load', 'Tuple', 'List', 'And', 'Or', 'Not', 'USub',
    'UAdd', 'Add', 'Sub', 'Mult', 'Div', 'FloorDiv', 'Mod', 'Eq', 'NotEq',
    'Lt', 'LtE', 'Gt', 'GtE', 'In', 'NotIn',
)

_CUSTOM_FUNCS = {
    'abs': abs, 'min': min, 'max': max, 'round': round, 'int': int,
    'float': float, 'bool': bool, 'str': str, 'sqrt': math.sqrt,
    'log': math.log, 'log10': math.log10, 'exp': math.exp,
    'floor': math.floor, 'ceil': math.ceil,
}

_CUSTOM_VARS = ('index', 'degree', 'clustering', 'component', 'n', 'm')

def _compile_custom_expression(expression):
    """Compile *expression* under the restricted grammar, as ``(code, names)``.

    These strings ride in .blend scenes and in the repro registry, so a bare
    eval() would be arbitrary code execution on file load. This is a grammar
    restriction and not a sandbox: only literals, arithmetic, comparisons,
    _CUSTOM_FUNCS and _CUSTOM_VARS survive parsing, and everything else raises
    before any evaluation happens."""
    import ast

    tree = ast.parse(expression, mode='eval')
    used_names = set()

    for node in ast.walk(tree):
        if type(node).__name__ not in _CUSTOM_NODES:
            raise ValueError(f"{type(node).__name__} is not allowed in a "
                             "custom expression")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _CUSTOM_FUNCS:
                raise ValueError("the only calls allowed are: "
                                 + ", ".join(sorted(_CUSTOM_FUNCS)))
            if node.keywords:
                raise ValueError("keyword arguments are not allowed")
        elif isinstance(node, ast.Name):
            if node.id in _CUSTOM_VARS:
                used_names.add(node.id)
            elif node.id not in _CUSTOM_FUNCS:
                raise ValueError(f"unknown name '{node.id}'; the variables are: "
                                 + ", ".join(_CUSTOM_VARS))

    return compile(tree, "<splitter custom expression>", "eval"), used_names

def _split_by_custom(G, expression):
    """One layer per distinct value of *expression* evaluated at each node.

    Variables: index, degree, clustering, component, n, m."""
    if not expression:
        print("Network Splitter: criterion CUSTOM needs an expression, and the "
              "operator passed none; see custom_expression")
        return None, {}

    try:
        code, used_names = _compile_custom_expression(expression)
    except (SyntaxError, ValueError) as e:
        print(f"Network Splitter: rejected custom expression: {e}")
        return None, {}

    num_nodes = len(G.nodes())
    num_edges = G.number_of_edges()

    degrees = dict(G.degree()) if 'degree' in used_names else {}
    clustering = nx.clustering(G) if 'clustering' in used_names else {}
    component = {}
    if 'component' in used_names:
        ordered = sorted(nx.connected_components(G), key=len, reverse=True)
        for comp_idx, comp in enumerate(ordered):
            for node in comp:
                component[node] = comp_idx

    values = []
    for i in range(num_nodes):
        scope = dict(_CUSTOM_FUNCS)
        scope.update(index=i, n=num_nodes, m=num_edges,
                     degree=degrees.get(i, 0),
                     clustering=clustering.get(i, 0.0),
                     component=component.get(i, 0))
        try:
            value = eval(code, {"__builtins__": {}}, scope)
        except Exception as e:
            print(f"Network Splitter: custom expression failed at node {i}: {e}")
            return None, {}

        if isinstance(value, bool):
            value = int(value)
        if not isinstance(value, (int, float, str)):
            print(f"Network Splitter: custom expression returned "
                  f"{type(value).__name__} at node {i}; it must return a number "
                  "or a string")
            return None, {}
        values.append(value)

    if any(isinstance(v, str) for v in values):
        values = [str(v) for v in values]

    unique_values = sorted(set(values))
    if len(unique_values) > _CUSTOM_MAX_LAYERS:
        print(f"Network Splitter: custom expression produced "
              f"{len(unique_values)} distinct values, over the "
              f"{_CUSTOM_MAX_LAYERS}-layer limit; round or bin it")
        return None, {}

    value_to_layer = {v: i for i, v in enumerate(unique_values)}
    layer_assignments = [value_to_layer[v] for v in values]
    layer_names = {i: str(v) for v, i in value_to_layer.items()}
    return layer_assignments, layer_names

__all__ = [name for name in globals() if not name.startswith('__')]
