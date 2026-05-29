import math

import bpy
import bmesh
import numpy as np
import pandas as pd
from ...utils.logger import log
from ..repro.determinism import get_geometry_seed


# ---------------------------------------------------------------------------
# Default visual appearance (used by SciGraphs_Viz Geometry Nodes builders
# and by the SCIGRAPHS_OT_UpdateAppearance dialog defaults). Centralised here
# so the GN tree and the operator never disagree.
# ---------------------------------------------------------------------------

DEFAULT_NODE_SIZE = 0.02
DEFAULT_NODE_RESOLUTION = 10
DEFAULT_NODE_SHAPE_INDEX = 0  # 0=Sphere, 1=Icosphere, 2=Cube, 3=Cone, 4=Cylinder
DEFAULT_NODE_ATTR_MULT = 1.0
DEFAULT_EDGE_THICKNESS = 0.005
DEFAULT_EDGE_RESOLUTION = 8
DEFAULT_EDGE_ATTR_MULT = 1.0

NODE_SHAPE_INDEX_MAP = {
    'SPHERE': 0,
    'ICOSPHERE': 1,
    'CUBE': 2,
    'CONE': 3,
    'CYLINDER': 4,
}


def _ico_subdivisions_from_resolution(resolution: int) -> int:
    """Derive a sensible icosphere ``Subdivisions`` value from a UI resolution.

    The resolution slider is shared across primitives. UV spheres take
    ``Segments``/``Rings`` directly, but the icosphere only exposes
    ``Subdivisions`` which doubles the triangle count exponentially. We map
    the linear resolution to the icosphere range ``[1, 5]`` so users get
    visually similar density without tanking the viewport.
    """
    return max(1, min(5, max(1, int(resolution) // 4)))


def _apply_node_primitive_inputs(node, node_size: float, resolution: int) -> int:
    """Push ``node_size`` / ``resolution`` defaults into a primitive node.

    Returns the number of input sockets actually written. Each primitive
    exposes a slightly different set of inputs, so this helper centralises
    the per-primitive mapping used by both the initial GN tree builders and
    the Update Appearance operator.
    """
    if node is None:
        return 0

    written = 0
    inputs = node.inputs

    if 'Radius' in inputs:
        inputs['Radius'].default_value = node_size
        written += 1
    if 'Radius Bottom' in inputs:
        inputs['Radius Bottom'].default_value = node_size
        written += 1
    if 'Radius Top' in inputs and node.type == 'GeometryNodeMeshCone':
        inputs['Radius Top'].default_value = 0.0
        written += 1
    if 'Depth' in inputs:
        inputs['Depth'].default_value = node_size * 2.0
        written += 1
    if 'Size' in inputs and node.type == 'GeometryNodeMeshCube':
        inputs['Size'].default_value = (node_size * 2.0,) * 3
        written += 1

    if 'Segments' in inputs:
        inputs['Segments'].default_value = max(3, int(resolution))
        written += 1
    if 'Rings' in inputs:
        inputs['Rings'].default_value = max(2, int(resolution) // 2)
        written += 1
    if 'Subdivisions' in inputs:
        inputs['Subdivisions'].default_value = _ico_subdivisions_from_resolution(resolution)
        written += 1
    if 'Vertices' in inputs:
        inputs['Vertices'].default_value = max(3, int(resolution))
        written += 1

    return written


def _add_smooth_by_angle_node(nodes, location):
    """Create the 'Smooth by Angle' (or equivalent) node.

    Blender ships *Smooth by Angle* as an Essentials asset, not as a
    built-in node, in many versions (4.x and 5.x as of writing). Trying to
    spawn ``GeometryNodeSetSmoothByAngle`` raises ``RuntimeError`` when the
    identifier is missing.

    With ``Angle=180°`` + ``Ignore Sharpness=True`` the asset is functionally
    equivalent to ``GeometryNodeSetShadeSmooth`` (every face becomes
    smooth-shaded, no edge is left sharp). We try the explicit node first
    for the future-proof case and gracefully fall back to ``Set Shade
    Smooth`` everywhere else.
    """
    smooth = None
    for ident in ('GeometryNodeSetSmoothByAngle',):
        try:
            smooth = nodes.new(ident)
            break
        except RuntimeError:
            smooth = None

    if smooth is None:
        smooth = nodes.new('GeometryNodeSetShadeSmooth')

    smooth.name = "SciGraphs_NodeSmoothByAngle"
    smooth.label = "Smooth by Angle"
    smooth.location = location

    if 'Angle' in smooth.inputs:
        smooth.inputs['Angle'].default_value = math.pi  # 180°
    if 'Ignore Sharpness' in smooth.inputs:
        smooth.inputs['Ignore Sharpness'].default_value = True
    if 'Shade Smooth' in smooth.inputs:
        smooth.inputs['Shade Smooth'].default_value = True

    return smooth


def _build_node_shape_switch(nodes, links, base_x: float, base_y: float,
                             node_size: float = DEFAULT_NODE_SIZE,
                             resolution: int = DEFAULT_NODE_RESOLUTION,
                             shape_index: int = DEFAULT_NODE_SHAPE_INDEX):
    """Spawn the 5 node primitives, an Index Switch and a Smooth-by-Angle.

    Returns ``(smooth_by_angle, primitives_dict)`` where the geometry coming
    out of the smooth-by-angle node is what should be fed into ``Instance on
    Points`` (already shape-selected and shaded).

    The primitives are named so the Update Appearance operator can find and
    re-tune them later without rebuilding the tree.
    """
    sphere = nodes.new('GeometryNodeMeshUVSphere')
    sphere.name = "SciGraphs_NodeSphere"
    sphere.label = "Node Sphere"
    sphere.location = (base_x, base_y)
    _apply_node_primitive_inputs(sphere, node_size, resolution)

    ico = nodes.new('GeometryNodeMeshIcoSphere')
    ico.name = "SciGraphs_NodeIcoSphere"
    ico.label = "Node Icosphere"
    ico.location = (base_x, base_y - 160)
    _apply_node_primitive_inputs(ico, node_size, resolution)

    cube = nodes.new('GeometryNodeMeshCube')
    cube.name = "SciGraphs_NodeCube"
    cube.label = "Node Cube"
    cube.location = (base_x, base_y - 320)
    _apply_node_primitive_inputs(cube, node_size, resolution)

    cone = nodes.new('GeometryNodeMeshCone')
    cone.name = "SciGraphs_NodeCone"
    cone.label = "Node Cone"
    cone.location = (base_x, base_y - 480)
    _apply_node_primitive_inputs(cone, node_size, resolution)

    cylinder = nodes.new('GeometryNodeMeshCylinder')
    cylinder.name = "SciGraphs_NodeCylinder"
    cylinder.label = "Node Cylinder"
    cylinder.location = (base_x, base_y - 640)
    _apply_node_primitive_inputs(cylinder, node_size, resolution)

    shape_switch = nodes.new('GeometryNodeIndexSwitch')
    shape_switch.name = "SciGraphs_NodeShapeSwitch"
    shape_switch.label = "Node Shape"
    shape_switch.data_type = 'GEOMETRY'
    shape_switch.location = (base_x + 240, base_y - 200)
    while len(shape_switch.inputs) < 6:
        shape_switch.index_switch_items.new()
    shape_switch.inputs['Index'].default_value = max(0, min(4, int(shape_index)))

    links.new(sphere.outputs['Mesh'], shape_switch.inputs[1])
    links.new(ico.outputs['Mesh'], shape_switch.inputs[2])
    links.new(cube.outputs['Mesh'], shape_switch.inputs[3])
    links.new(cone.outputs['Mesh'], shape_switch.inputs[4])
    links.new(cylinder.outputs['Mesh'], shape_switch.inputs[5])

    smooth_by_angle = _add_smooth_by_angle_node(
        nodes, location=(base_x + 460, base_y - 200)
    )
    links.new(shape_switch.outputs['Output'], smooth_by_angle.inputs['Geometry'])

    return smooth_by_angle, {
        'sphere': sphere,
        'icosphere': ico,
        'cube': cube,
        'cone': cone,
        'cylinder': cylinder,
        'switch': shape_switch,
    }

def create_graph_object(graph_data, is_directed=False, selected_attributes=None, remove_self_loops=True):
    """
    Creates a mesh object with vertices for nodes and edges for connections.
    Optimized for large graphs using batch operations.
    
    Args:
        graph_data: GraphData object with nodes, edges, and dataframe.
        is_directed: Whether the graph is directed.
        selected_attributes: Optional list of column names to import as mesh
            attributes. When None, all numeric columns are imported.
        remove_self_loops: When True, edges where source equals target are
            discarded during mesh construction.
    """
    import time
    start_time = time.time()
    
    num_nodes = len(graph_data.nodes)
    log(f"\nCreating graph object with {num_nodes:,} nodes...")
    
    # Generate initial positions
    pos_start = time.time()
    # Use reproducible seed for geometry generation
    geom_seed = get_geometry_seed()
    rng = np.random.RandomState(geom_seed)
    
    if hasattr(graph_data, 'node_coordinates') and graph_data.node_coordinates:
        # Use provided coordinates (e.g. from SuiteSparse _coord.mtx)
        initial_positions = np.zeros((num_nodes, 3))
        coords_used = 0
        for i, node in enumerate(graph_data.nodes):
            if node in graph_data.node_coordinates:
                c = graph_data.node_coordinates[node]
                initial_positions[i, 0] = c[0]
                initial_positions[i, 1] = c[1]
                if len(c) > 2:
                    initial_positions[i, 2] = c[2]
                coords_used += 1
            else:
                initial_positions[i] = rng.rand(3) * 5.0
        # Normalize to fit viewport (center and scale to ~5 units)
        mins = initial_positions.min(axis=0)
        maxs = initial_positions.max(axis=0)
        span = (maxs - mins).max()
        if span > 0:
            initial_positions = (initial_positions - mins) / span * 10.0 - 5.0
        log(f"  Coordinates applied for {coords_used}/{num_nodes} nodes in {time.time() - pos_start:.2f}s")
    else:
        initial_positions = rng.rand(num_nodes, 3) * 5.0
        log(f"  Random positions generated in {time.time() - pos_start:.2f}s (seed={geom_seed})")
    
    # Create mesh with vertices at node positions
    mesh_start = time.time()
    mesh = bpy.data.meshes.new(name="SciGraph_Mesh")
    bm = bmesh.new()
    
    # Batch create vertices (faster than loop)
    bm.verts.ensure_lookup_table()
    verts = []
    for i in range(num_nodes):
        pos = initial_positions[i]
        v = bm.verts.new(pos)
        verts.append(v)
    
    # Update internal tables for faster edge creation
    bm.verts.ensure_lookup_table()
    bm.verts.index_update()
    log(f"  Vertices created in {time.time() - mesh_start:.2f}s")
    
    # Create edges between nodes
    edges_start = time.time()
    
    # Create node to index mapping using numpy for speed
    node_to_idx = {node: i for i, node in enumerate(graph_data.nodes)}
    
    # Keep track of which dataframe rows correspond to created edges
    edge_df_indices = []
    
    # Track created edges to avoid duplicates
    created_edges = set()
    
    # Process edges in batches for progress reporting
    num_edges = len(graph_data.edges)
    batch_size = 10000
    edges_created = 0
    
    self_loops_skipped = 0
    
    for edge_idx, (src, tgt) in enumerate(graph_data.edges):
        if src in node_to_idx and tgt in node_to_idx:
            src_idx = node_to_idx[src]
            tgt_idx = node_to_idx[tgt]
            
            if remove_self_loops and src_idx == tgt_idx:
                self_loops_skipped += 1
                continue
            
            if src_idx < len(verts) and tgt_idx < len(verts):
                edge_key = (min(src_idx, tgt_idx), max(src_idx, tgt_idx))
                
                if edge_key not in created_edges:
                    try:
                        bm.edges.new([verts[src_idx], verts[tgt_idx]])
                        created_edges.add(edge_key)
                        edge_df_indices.append(edge_idx)
                        edges_created += 1
                    except ValueError:
                        pass
        
        # Progress reporting for large graphs
        if (edge_idx + 1) % batch_size == 0:
            progress = (edge_idx + 1) / num_edges * 100
            log(f"  Creating edges: {progress:.0f}% ({edge_idx + 1:,}/{num_edges:,})")
    
    log(f"  {edges_created:,} edges created in {time.time() - edges_start:.2f}s")
    if self_loops_skipped > 0:
        log(f"  {self_loops_skipped:,} self-loops removed")
    
    # Convert bmesh to mesh
    convert_start = time.time()
    bm.to_mesh(mesh)
    bm.free()
    log(f"  Mesh conversion in {time.time() - convert_start:.2f}s")
    
    # Create object
    obj_start = time.time()
    obj = bpy.data.objects.new("SciGraph_Object", mesh)
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    log(f"  Object created in {time.time() - obj_start:.2f}s")
    
    # Store graph data and positions
    obj["node_positions"] = initial_positions.flatten().tolist()
    obj["num_nodes"] = num_nodes
    obj["nodes_data"] = ",".join(str(node) for node in graph_data.nodes)
    
    if graph_data.edges:
        edges_flat = []
        stored_edge_count = 0
        for src, tgt in graph_data.edges:
            if remove_self_loops and str(src) == str(tgt):
                continue
            edges_flat.append(str(src))
            edges_flat.append(str(tgt))
            stored_edge_count += 1
        obj["edges_data"] = ",".join(edges_flat)
        obj["num_edges"] = stored_edge_count
    else:
        obj["edges_data"] = ""
        obj["num_edges"] = 0
    
    obj["is_directed"] = is_directed
    
    if graph_data.dataframe is not None:
        attr_start = time.time()
        import_attributes_from_dataframe(obj, graph_data, edge_df_indices, selected_attributes)
        log(f"  Attributes imported in {time.time() - attr_start:.2f}s")
    
    # Note: Geometry Nodes visualization is now applied separately
    # Use "Setup Visualization" button in the UI panel
    
    log(f"Total graph creation time: {time.time() - start_time:.2f}s")
    
    return obj

def import_attributes_from_dataframe(obj, graph_data, edge_df_indices, selected_attributes=None):
    """
    Imports columns from the dataframe as mesh attributes.
    Edge attributes are stored on edges, node attributes on vertices.
    
    When *selected_attributes* is provided only those column names are
    considered; otherwise every eligible column is imported.
    
    For each numeric column, creates:
    - edge_{column}: The value for each edge (from first occurrence)
    - vertex_{column}_sum: Sum of values for each node
    - vertex_{column}_mean: Mean of values for each node
    - vertex_{column}_min: Min value for each node
    - vertex_{column}_max: Max value for each node
    
    Args:
        obj: Blender mesh object to store attributes on.
        graph_data: GraphData with nodes, edges and dataframe.
        edge_df_indices: Mapping from mesh edge index to dataframe row.
        selected_attributes: Optional list of column names to import.
            When None every numeric column (except source/target) is imported.
    """
    import json
    
    df = graph_data.dataframe
    if df is None or len(df) == 0:
        return
    
    mesh = obj.data
    
    skip_cols = {'source', 'target', 'Source', 'Target'}
    
    source_col = getattr(graph_data, 'source_column_name', None)
    target_col = getattr(graph_data, 'target_column_name', None)
    
    if not source_col or not target_col:
        for col in df.columns:
            col_lower = col.lower()
            if col_lower == 'source':
                source_col = col
            elif col_lower == 'target':
                target_col = col
    
    if not source_col or not target_col:
        if len(df.columns) >= 2:
            source_col = df.columns[0]
            target_col = df.columns[1]
    
    log(f"DataFrame columns: {list(df.columns)}")
    log(f"DataFrame shape: {df.shape}")
    log(f"Source column: {source_col}, Target column: {target_col}")
    log(f"Number of mesh vertices: {len(mesh.vertices)}")
    log(f"Number of mesh edges: {len(mesh.edges)}")
    
    if selected_attributes is not None:
        selected_set = set(selected_attributes)
        log(f"Selected attributes for import: {selected_attributes}")
    else:
        selected_set = None
    
    node_to_idx = {str(node): i for i, node in enumerate(graph_data.nodes)}
    num_vertices = len(graph_data.nodes)
    imported_count = 0
    
    for col_name in df.columns:
        if col_name in skip_cols or col_name == source_col or col_name == target_col:
            continue
        
        if selected_set is not None and col_name not in selected_set:
            continue
        
        col_data = df[col_name]
        log(f"Processing column '{col_name}': {col_data.dtype}")
        
        if pd.api.types.is_numeric_dtype(col_data):
            _create_edge_attribute(mesh, col_name, col_data, edge_df_indices)
            
            _create_vertex_attributes_from_column(
                mesh, obj, df, col_name, source_col, target_col, 
                node_to_idx, num_vertices
            )
            imported_count += 1
        else:
            unique_vals = col_data.dropna().unique()
            if len(unique_vals) <= 100:
                obj[f"attr_{col_name}_values"] = ",".join(str(v) for v in unique_vals[:100])
            log(f"Stored string column '{col_name}' as property")
            imported_count += 1
    
    if "node_id" not in mesh.attributes:
        attr = mesh.attributes.new(name="node_id", type='INT', domain='POINT')
        node_ids = list(range(num_vertices))
        attr.data.foreach_set("value", node_ids)
    
    obj["node_names"] = json.dumps([str(node) for node in graph_data.nodes])
    
    log(f"Imported {imported_count} attribute columns")


def _create_edge_attribute(mesh, col_name, col_data, edge_df_indices):
    """Create edge attribute from column data."""
    attr_name = f"edge_{col_name}"
    if attr_name in mesh.attributes:
        return
    
    attr = mesh.attributes.new(name=attr_name, type='FLOAT', domain='EDGE')
    
    values = np.zeros(len(mesh.edges), dtype=np.float32)
    col_values = col_data.values
    
    for mesh_edge_idx in range(len(mesh.edges)):
        if mesh_edge_idx < len(edge_df_indices):
            df_row = edge_df_indices[mesh_edge_idx]
            if df_row < len(col_values):
                try:
                    values[mesh_edge_idx] = float(col_values[df_row])
                except (ValueError, TypeError):
                    pass
    
    attr.data.foreach_set("value", values.tolist())
    log(f"  Created edge attribute '{attr_name}'")


def _create_vertex_attributes_from_column(mesh, obj, df, col_name, source_col, target_col, 
                                          node_to_idx, num_vertices):
    """
    Create vertex attributes by aggregating edge data per node.
    Uses vectorized pandas operations for efficiency with large datasets.
    """
    vertex_sums = np.zeros(num_vertices, dtype=np.float64)
    vertex_counts = np.zeros(num_vertices, dtype=np.int32)
    vertex_mins = np.full(num_vertices, np.inf, dtype=np.float64)
    vertex_maxs = np.full(num_vertices, -np.inf, dtype=np.float64)
    
    work_df = df[[source_col, target_col, col_name]].copy()
    work_df[col_name] = pd.to_numeric(work_df[col_name], errors='coerce')
    work_df = work_df.dropna(subset=[col_name])
    
    if len(work_df) == 0:
        log(f"  No valid numeric values in '{col_name}', skipping vertex attributes")
        return
    
    for node_col in [source_col, target_col]:
        grouped = work_df.groupby(node_col)[col_name].agg(['sum', 'count', 'min', 'max'])
        
        for node_name, row in grouped.iterrows():
            node_key = str(node_name)
            if node_key in node_to_idx:
                idx = node_to_idx[node_key]
                vertex_sums[idx] += row['sum']
                vertex_counts[idx] += int(row['count'])
                if row['min'] < vertex_mins[idx]:
                    vertex_mins[idx] = row['min']
                if row['max'] > vertex_maxs[idx]:
                    vertex_maxs[idx] = row['max']
    
    vertex_mins[vertex_mins == np.inf] = 0.0
    vertex_maxs[vertex_maxs == -np.inf] = 0.0
    
    vertex_means = np.divide(vertex_sums, vertex_counts, 
                             out=np.zeros_like(vertex_sums), 
                             where=vertex_counts > 0)
    
    attr_configs = [
        (f"vertex_{col_name}_sum", vertex_sums, 'FLOAT'),
        (f"vertex_{col_name}_mean", vertex_means, 'FLOAT'),
        (f"vertex_{col_name}_min", vertex_mins, 'FLOAT'),
        (f"vertex_{col_name}_max", vertex_maxs, 'FLOAT'),
        (f"vertex_{col_name}_count", vertex_counts, 'INT'),
    ]
    
    for attr_name, values, attr_type in attr_configs:
        if attr_name not in mesh.attributes:
            attr = mesh.attributes.new(name=attr_name, type=attr_type, domain='POINT')
            attr.data.foreach_set("value", values.tolist())
    
    log(f"  Created vertex attributes for '{col_name}': sum, mean, min, max, count")


def import_node_attributes_from_file(obj, filepath, delimiter='\t', has_header=False):
    """
    Import vertex-only attributes from an external file onto an existing graph.

    The file must have a node identifier in the first column and one or more
    value columns.  Nodes present in the graph but absent from the file
    receive ``float('nan')`` so that downstream tools can distinguish
    missing data from zero.

    Args:
        obj: Blender mesh object that already holds a SciGraphs graph.
        filepath: Absolute path to the attribute file.
        delimiter: Column separator character.
        has_header: Whether the first row contains column names.

    Returns:
        Tuple (num_attributes_imported, num_nodes_matched).
    """
    import json

    if obj is None or obj.type != 'MESH':
        log("Error: No valid mesh object provided")
        return 0, 0

    nodes_data = obj.get("nodes_data", "")
    if not nodes_data:
        log("Error: Object has no stored node names")
        return 0, 0

    node_names = nodes_data.split(",")
    node_to_idx = {name: i for i, name in enumerate(node_names)}

    df = pd.read_csv(
        filepath,
        sep=delimiter,
        header=0 if has_header else None,
        dtype=str,
    )

    if df.shape[1] < 2:
        log("Error: File must contain at least two columns (node, value)")
        return 0, 0

    id_col = df.columns[0]
    value_cols = df.columns[1:]

    mesh = obj.data
    total_verts = len(mesh.vertices)
    attrs_imported = 0
    nodes_matched = set()

    for col in value_cols:
        numeric_series = pd.to_numeric(df[col], errors='coerce')
        all_nan = numeric_series.isna().all()

        if all_nan:
            log(f"  Skipping column '{col}': no valid numeric values")
            continue

        attr_name = str(col) if has_header else f"node_attr_{attrs_imported}"

        if attr_name in mesh.attributes:
            mesh.attributes.remove(mesh.attributes[attr_name])

        attr = mesh.attributes.new(name=attr_name, type='FLOAT', domain='POINT')
        values = np.full(total_verts, float('nan'), dtype=np.float32)

        for row_idx in range(len(df)):
            node_key = str(df[id_col].iloc[row_idx]).strip()
            if node_key in node_to_idx:
                vert_idx = node_to_idx[node_key]
                if vert_idx < total_verts:
                    val = numeric_series.iloc[row_idx]
                    if not pd.isna(val):
                        values[vert_idx] = float(val)
                        nodes_matched.add(node_key)

        attr.data.foreach_set("value", values.tolist())
        attrs_imported += 1
        log(f"  Created POINT attribute '{attr_name}' ({len(nodes_matched)} nodes matched)")

    node_names_json = obj.get("node_names")
    if not node_names_json:
        obj["node_names"] = json.dumps(node_names)

    log(f"Imported {attrs_imported} node attribute(s), {len(nodes_matched)} nodes matched out of {len(node_names)}")
    return attrs_imported, len(nodes_matched)


# Legacy per-type socket indexes (Blender <= 3.x).
# From Blender 4.0+ GeometryNodeInputNamedAttribute exposes a single "Attribute"
# output and GeometryNodeStoreNamedAttribute a single "Value" input regardless
# of data_type. We try the name first and fall back to these indexes for
# compatibility with older files.
_GN_STORE_VALUE_INDEX = {
    'FLOAT': 4,
    'INT': 7,
    'FLOAT_VECTOR': 3,
    'FLOAT_COLOR': 5,
    'BOOLEAN': 6,
}

_GN_ATTR_OUTPUT_INDEX = {
    'FLOAT': 1,
    'INT': 4,
    'FLOAT_VECTOR': 0,
    'FLOAT_COLOR': 2,
    'BOOLEAN': 3,
}


def _gn_attribute_output_socket(read_node, gn_type):
    """Return the value output of a GeometryNodeInputNamedAttribute, tolerant to
    Blender versions. In 4.x+ the only typed output is named 'Attribute';
    in older versions there is one output per type at a fixed index.
    """
    if "Attribute" in read_node.outputs:
        return read_node.outputs["Attribute"]
    idx = _GN_ATTR_OUTPUT_INDEX.get(gn_type, 0)
    if idx < len(read_node.outputs):
        return read_node.outputs[idx]
    return read_node.outputs[0]


def _gn_store_value_input(store_node, gn_type):
    """Return the value input of a GeometryNodeStoreNamedAttribute, tolerant to
    Blender versions. In 4.x+ it is always named 'Value'; in older versions
    it is one of several typed inputs at a fixed index.
    """
    if "Value" in store_node.inputs:
        return store_node.inputs["Value"]
    idx = _GN_STORE_VALUE_INDEX.get(gn_type, 3)
    if idx < len(store_node.inputs):
        return store_node.inputs[idx]
    return store_node.inputs[3]

_MESH_TYPE_TO_GN = {
    'FLOAT': 'FLOAT',
    'INT': 'INT',
    'FLOAT_VECTOR': 'FLOAT_VECTOR',
    'FLOAT_COLOR': 'FLOAT_COLOR',
    'BOOLEAN': 'BOOLEAN',
    'INT8': 'INT',
    'BYTE_COLOR': 'FLOAT_COLOR',
    'FLOAT2': 'FLOAT_VECTOR',
    'QUATERNION': 'FLOAT_VECTOR',
}


_BUILTIN_ATTRS = {
    'position', '.corner_vert', '.corner_edge', '.edge_verts',
    'sharp_edge', 'crease_edge', 'material_index', 'shade_smooth',
    'normal', 'UVMap',
}


def _strip_custom_attributes_from_geo(nodes, links, geo_socket, mesh, base_location):
    """
    Chain GeometryNodeRemoveAttribute nodes to strip every custom attribute
    from the geometry flowing through *geo_socket*.

    This prevents node-specific attribute values (centrality, community,
    imported columns, etc.) from bleeding into edge tube geometry via the
    Mesh-to-Curve pipeline, where the first/last tube segment would otherwise
    inherit the connected node vertex's attribute values.

    Args:
        nodes: Node tree nodes collection.
        links: Node tree links collection.
        geo_socket: The output socket carrying the geometry to clean.
        mesh: The Blender mesh whose attributes are inspected.
        base_location: (x, y) starting position for the new nodes.

    Returns:
        The output geometry socket after all removals (or the original
        socket if there are no custom attributes to remove).
    """
    # Allow the SciGraphs coloring toolbar (and any other system) to keep one
    # named attribute alive through the pipeline by setting
    # ``obj["scigraphs_color_attr"] = "<attr_name>"``. The associated FLOAT_COLOR
    # layer (``<attr>_color``) is also preserved so shaders can sample it.
    obj_for_attrs = getattr(mesh, "id_data", None)
    keep_names = set()
    color_attr = None
    if obj_for_attrs is not None:
        # Find the object owning this mesh to read its custom property.
        for candidate in bpy.data.objects:
            if candidate.data is mesh:
                color_attr = candidate.get("scigraphs_color_attr", "")
                break
    if color_attr:
        keep_names.add(color_attr)
        keep_names.add(f"{color_attr}_color")

    custom_names = [
        attr.name for attr in mesh.attributes
        if attr.name not in _BUILTIN_ATTRS
        and not attr.name.startswith('.')
        and attr.name not in keep_names
    ]

    if not custom_names:
        return geo_socket

    current = geo_socket
    x, y = base_location

    for attr_name in custom_names:
        remove = nodes.new(type='GeometryNodeRemoveAttribute')
        remove.location = (x, y)
        remove.inputs['Name'].default_value = attr_name
        links.new(current, remove.inputs['Geometry'])
        current = remove.outputs['Geometry']
        y -= 50

    return current


def _promote_edge_attributes_to_point(nodes, links, geo_socket, mesh):
    """
    Insert Geometry Nodes that re-domain EDGE attributes to POINT so they
    survive the Mesh-to-Curve and Curve-to-Mesh pipeline.

    For each EDGE-domain attribute on the mesh, a pair of InputNamedAttribute
    and StoreNamedAttribute nodes is added. The attribute is read on the edge
    domain and written back on the point domain (Blender averages connected
    edges per vertex automatically).

    Args:
        nodes: Node tree nodes collection.
        links: Node tree links collection.
        geo_socket: The geometry output socket to chain from.
        mesh: The Blender mesh data with attributes.

    Returns:
        The output geometry socket after all promotions (may be the same
        socket if no edge attributes exist).
    """
    _BUILTIN_EDGE = {'.edge_verts', 'sharp_edge', 'crease_edge', 'material_index'}

    edge_attrs = [
        (attr.name, attr.data_type)
        for attr in mesh.attributes
        if attr.domain == 'EDGE'
        and attr.name not in _BUILTIN_EDGE
        and not attr.name.startswith('.')
    ]

    if not edge_attrs:
        return geo_socket

    current = geo_socket
    y = -900

    for attr_name, mesh_type in edge_attrs:
        gn_type = _MESH_TYPE_TO_GN.get(mesh_type)
        if gn_type is None:
            continue

        read = nodes.new('GeometryNodeInputNamedAttribute')
        read.data_type = gn_type
        read.inputs['Name'].default_value = attr_name
        read.location = (-1100, y)

        store = nodes.new('GeometryNodeStoreNamedAttribute')
        store.data_type = gn_type
        store.domain = 'POINT'
        store.inputs['Name'].default_value = attr_name
        store.location = (-900, y)

        links.new(current, store.inputs['Geometry'])
        try:
            links.new(
                _gn_attribute_output_socket(read, gn_type),
                _gn_store_value_input(store, gn_type),
            )
        except (IndexError, RuntimeError) as exc:
            log(f"  WARNING: could not promote edge attribute '{attr_name}' "
                f"(type={gn_type}): {exc}")
            nodes.remove(read)
            nodes.remove(store)
            continue

        current = store.outputs['Geometry']
        y -= 150

    log(f"  Promoted {len(edge_attrs)} edge attribute(s) to point domain for GN propagation")
    return current


def _split_edges_for_individual_curves(nodes, links, geo_socket, location, mesh=None):
    """
    Duplicate shared edge vertices before Mesh-to-Curve when needed.

    Blender's Mesh to Curve node merges connected mesh edges into longer
    splines, which would render tubes passing through intermediate nodes
    instead of stopping at each graph edge.

    Two regimes:

    * **OSMnx-style meshes** (``is_intersection`` attribute is present):
      every street is a chain of intermediate curve points and the
      intersections themselves have degree ≥ 3. ``Mesh to Curve`` already
      breaks splines at any vertex whose degree is not 2, so adding a
      ``Split Edges`` here would only introduce visible gaps along each
      street between consecutive curve points. We skip the split in that
      case and let the topology drive the chain breaks.

    * **Other meshes** (no ``is_intersection``): every vertex is a node we
      want to render as a sphere. Path-like sub-graphs (degree-2 chains)
      would otherwise be merged into a single tube. We keep the global
      ``Split Edges`` so each graph edge stays its own tube.
    """
    if mesh is not None and "is_intersection" in mesh.attributes:
        log("  Skipping Split Edges: 'is_intersection' present, keeping streets continuous")
        return geo_socket

    try:
        split_edges = nodes.new(type='GeometryNodeSplitEdges')
    except RuntimeError as exc:
        log(f"  WARNING: Split Edges node unavailable; edge tubes may chain through shared nodes: {exc}")
        return geo_socket

    split_edges.location = location
    links.new(geo_socket, split_edges.inputs['Mesh'])
    return split_edges.outputs['Mesh']


def setup_geometry_nodes_visualization(obj):
    """
    Sets up Geometry Nodes to add sphere instances on vertices
    and cylinder instances on edges for better visualization.
    
    For OSMnx graphs, only places spheres on intersection vertices (is_intersection=1),
    not on the intermediate curve points.
    
    This function is idempotent - it will reuse existing modifier/node_group if present.
    """
    # Check if modifier already exists
    mod = obj.modifiers.get("SciGraphs_Viz")
    
    if mod is None:
        # Create new modifier
        mod = obj.modifiers.new(name="SciGraphs_Viz", type='NODES')
    
    # Check if we have an existing node group to reuse or need to create new
    if mod.node_group is not None:
        node_group = mod.node_group
        # Clear existing nodes to rebuild
        node_group.nodes.clear()
        # Clear existing interface sockets
        node_group.interface.clear()
    else:
        # Create new node group
        node_group = bpy.data.node_groups.new(name="SciGraphs_Visualization", type='GeometryNodeTree')
        mod.node_group = node_group
    
    # Create input and output sockets in the interface
    node_group.interface.new_socket(name="Geometry", in_out='INPUT', socket_type='NodeSocketGeometry')
    node_group.interface.new_socket(name="Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')
    
    nodes = node_group.nodes
    links = node_group.links
    
    # Check if this is an OSMnx graph with is_intersection attribute
    is_osmnx = obj.get("is_osmnx", False)
    has_intersection_attr = "is_intersection" in obj.data.attributes
    
    # Input node
    group_input = nodes.new(type='NodeGroupInput')
    group_input.location = (-800, 0)
    
    # Output node
    group_output = nodes.new(type='NodeGroupOutput')
    group_output.location = (800, 0)
    
    prepared_geo = _promote_edge_attributes_to_point(
        nodes, links, group_input.outputs['Geometry'], obj.data
    )
    
    # Separate vertices and edges
    mesh_to_points = nodes.new(type='GeometryNodeMeshToPoints')
    mesh_to_points.location = (-600, 200)
    links.new(prepared_geo, mesh_to_points.inputs['Mesh'])

    # Build node primitives (sphere/ico/cube/cone/cylinder) behind an Index
    # Switch + a Smooth by Angle (180°, ignore sharpness) so the user can swap
    # shapes from the Update Appearance dialog without rebuilding the tree.
    initial_shape = max(0, min(4, int(obj.get("scigraphs_node_shape_index",
                                              DEFAULT_NODE_SHAPE_INDEX))))
    initial_size = float(obj.get("scigraphs_node_size", DEFAULT_NODE_SIZE))
    initial_res = int(obj.get("scigraphs_node_resolution", DEFAULT_NODE_RESOLUTION))

    smooth_by_angle, _ = _build_node_shape_switch(
        nodes, links, base_x=-700, base_y=120,
        node_size=initial_size, resolution=initial_res, shape_index=initial_shape,
    )

    # Instance the resulting (shape-selected, smooth-shaded) mesh on vertices
    instance_on_points = nodes.new(type='GeometryNodeInstanceOnPoints')
    instance_on_points.location = (-200, 200)
    links.new(mesh_to_points.outputs['Points'], instance_on_points.inputs['Points'])
    links.new(smooth_by_angle.outputs['Geometry'], instance_on_points.inputs['Instance'])
    
    if is_osmnx or has_intersection_attr:
        named_attr = nodes.new(type='GeometryNodeInputNamedAttribute')
        named_attr.data_type = 'INT'
        named_attr.inputs['Name'].default_value = "is_intersection"
        named_attr.location = (-600, 400)
        
        compare_node = nodes.new(type='FunctionNodeCompare')
        compare_node.data_type = 'INT'
        compare_node.operation = 'EQUAL'
        compare_node.inputs['B'].default_value = 1
        compare_node.location = (-400, 400)
        
        links.new(named_attr.outputs['Attribute'], compare_node.inputs['A'])
        links.new(compare_node.outputs['Result'], instance_on_points.inputs['Selection'])
    
    realize_instances = nodes.new(type='GeometryNodeRealizeInstances')
    realize_instances.location = (0, 200)
    links.new(instance_on_points.outputs['Instances'], realize_instances.inputs['Geometry'])
    
    # Convert each graph edge to its own curve for better visualization
    edge_geo = _split_edges_for_individual_curves(
        nodes, links, prepared_geo, (-800, -200), mesh=obj.data
    )

    mesh_to_curve = nodes.new(type='GeometryNodeMeshToCurve')
    mesh_to_curve.location = (-600, -200)
    links.new(edge_geo, mesh_to_curve.inputs['Mesh'])
    
    initial_edge_thickness = float(obj.get("scigraphs_edge_thickness", DEFAULT_EDGE_THICKNESS))
    initial_edge_resolution = max(3, int(obj.get("scigraphs_edge_resolution", DEFAULT_EDGE_RESOLUTION)))

    curve_circle = nodes.new(type='GeometryNodeCurvePrimitiveCircle')
    curve_circle.name = "SciGraphs_EdgeProfile"
    curve_circle.label = "Edge Profile"
    curve_circle.mode = 'RADIUS'
    curve_circle.inputs['Radius'].default_value = initial_edge_thickness
    curve_circle.inputs['Resolution'].default_value = initial_edge_resolution
    curve_circle.location = (-600, -400)
    
    # Convert curve to mesh with profile
    curve_to_mesh = nodes.new(type='GeometryNodeCurveToMesh')
    curve_to_mesh.location = (-200, -300)
    links.new(mesh_to_curve.outputs['Curve'], curve_to_mesh.inputs['Curve'])
    links.new(curve_circle.outputs['Curve'], curve_to_mesh.inputs['Profile Curve'])
    
    edge_geo_output = _strip_custom_attributes_from_geo(
        nodes, links, curve_to_mesh.outputs['Mesh'], obj.data, (100, -300)
    )
    
    # Join nodes and edges
    join_geometry = nodes.new(type='GeometryNodeJoinGeometry')
    join_geometry.location = (400, 0)
    links.new(realize_instances.outputs['Geometry'], join_geometry.inputs['Geometry'])
    links.new(edge_geo_output, join_geometry.inputs['Geometry'])
    
    # Set material
    set_material = nodes.new(type='GeometryNodeSetMaterial')
    set_material.location = (600, 0)
    links.new(join_geometry.outputs['Geometry'], set_material.inputs['Geometry'])
    
    # Output
    links.new(set_material.outputs['Geometry'], group_output.inputs['Geometry'])

def fix_node_names(obj):
    """
    Updates node names in existing geometry node trees for compatibility.
    This is useful for graphs created before node naming was added.
    """
    mod = obj.modifiers.get("SciGraphs_Viz")
    if not mod or not mod.node_group:
        return False
    
    node_group = mod.node_group
    fixed = False
    
    # Find and rename the sphere node
    for node in node_group.nodes:
        if node.type == 'GeometryNodeMeshUVSphere' and not node.name.startswith("SciGraphs_"):
            node.name = "SciGraphs_NodeSphere"
            node.label = "Node Sphere"
            fixed = True
        elif node.type == 'GeometryNodeCurvePrimitiveCircle' and not node.name.startswith("SciGraphs_"):
            node.name = "SciGraphs_EdgeProfile"
            node.label = "Edge Profile"
            fixed = True
    
    return fixed

def update_node_positions_from_property(obj):
    """
    Updates vertex positions from the stored custom property.
    Called after layout calculations.
    """
    if "node_positions" not in obj or not obj.data.vertices:
        return
    
    positions = np.array(obj["node_positions"]).reshape(-1, 3)
    
    # Update vertex positions directly
    for i, vert in enumerate(obj.data.vertices):
        if i < len(positions):
            vert.co = positions[i]
    
    obj.data.update()
    
def rebuild_edges(obj):
    """
    Rebuilds edges after updating vertex positions.
    """
    if "edges_data" not in obj:
        return
    
    # Parse edges
    edges_str = obj.get("edges_data", "")
    if not edges_str:
        return
    
    edges_flat = edges_str.split(",")
    edges_list = [(edges_flat[i], edges_flat[i+1]) for i in range(0, len(edges_flat), 2)]
    
    # Parse nodes
    nodes_str = obj.get("nodes_data", "")
    nodes_list = nodes_str.split(",") if nodes_str else []
    node_to_idx = {node: i for i, node in enumerate(nodes_list)}
    
    # Rebuild edges in mesh
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    
    # Clear existing edges
    for edge in list(bm.edges):
        bm.edges.remove(edge)
    
    # Add edges back (avoiding duplicates for undirected graphs)
    verts_list = list(bm.verts)
    added_edges = set()
    
    for src, tgt in edges_list:
        if src in node_to_idx and tgt in node_to_idx:
            src_idx = node_to_idx[src]
            tgt_idx = node_to_idx[tgt]
            
            # Skip self-loops (edges from a node to itself)
            if src_idx == tgt_idx:
                continue
            
            if src_idx < len(verts_list) and tgt_idx < len(verts_list):
                # For undirected graphs, normalize edge to avoid duplicates
                edge_key = (min(src_idx, tgt_idx), max(src_idx, tgt_idx))
                if edge_key not in added_edges:
                    bm.edges.new([verts_list[src_idx], verts_list[tgt_idx]])
                    added_edges.add(edge_key)
    
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()

def create_geospatial_graph_object(graph_data, positions_3d, edge_style='GREAT_CIRCLE',
                                   is_directed=False, selected_attributes=None,
                                   remove_self_loops=True):
    """
    Creates a graph object with geospatial positioning.
    
    Args:
        graph_data: GraphData object with node_coordinates attribute.
        positions_3d: Dict mapping node_name -> [x, y, z] position.
        edge_style: 'STRAIGHT' or 'GREAT_CIRCLE'.
        is_directed: Whether the graph is directed.
        selected_attributes: Optional list of column names to import.
        remove_self_loops: Discard edges where source equals target.
    
    Returns:
        Created graph object.
    """
    from ..geo import geospatial
    import time
    
    start_time = time.time()
    num_nodes = len(graph_data.nodes)
    
    log(f"\nCreating geospatial graph with {num_nodes:,} nodes...")
    
    # Create mesh for nodes ONLY
    mesh = bpy.data.meshes.new(name="GeoGraph_Mesh")
    bm = bmesh.new()
    
    # Create custom layer for marking real nodes
    is_node_layer = bm.verts.layers.int.new("is_real_node")
    
    # Create vertices at geographic positions (NODES ONLY)
    verts = []
    node_to_vert = {}
    node_to_index = {}
    
    for i, node in enumerate(graph_data.nodes):
        node_str = str(node)
        
        if node_str in positions_3d:
            pos = positions_3d[node_str]
            v = bm.verts.new(pos)
            v[is_node_layer] = 1  # Mark as real node
            verts.append(v)
            node_to_vert[node] = v
            node_to_index[node] = i
        else:
            log(f"Warning: No position for node {node}")
    
    bm.verts.ensure_lookup_table()
    bm.verts.index_update()
    log(f"  Created {len(verts):,} node vertices with positions")
    
    # Create edges and track mapping from mesh edges to graph edges
    edges_start = time.time()
    edges_created = 0
    created_edges = set()
    
    # Map: (src_node, tgt_node) -> list of bmesh edge indices that represent this connection
    graph_edge_to_mesh_edges = {}
    
    # For great circle arcs, create intermediate vertices but mark them as NOT real nodes
    if edge_style == 'GREAT_CIRCLE':
        log("  Creating great circle arcs...")
        
        for edge_idx, (src, tgt) in enumerate(graph_data.edges):
            if remove_self_loops and src == tgt:
                continue
            
            if src in node_to_vert and tgt in node_to_vert:
                edge_key = (min(src, tgt), max(src, tgt))
                
                if edge_key not in created_edges:
                    src_pos = positions_3d[str(src)]
                    tgt_pos = positions_3d[str(tgt)]
                    
                    arc_points = geospatial.generate_great_circle_points(
                        src_pos, tgt_pos, num_segments=10
                    )
                    
                    # Create vertices for arc points (except first and last)
                    arc_verts = [node_to_vert[src]]
                    
                    for point in arc_points[1:-1]:
                        arc_v = bm.verts.new(point)
                        arc_v[is_node_layer] = 0  # Mark as curve point, NOT a real node
                        arc_verts.append(arc_v)
                    
                    arc_verts.append(node_to_vert[tgt])
                    
                    # Create edges connecting arc vertices and track them
                    mesh_edge_indices = []
                    for j in range(len(arc_verts) - 1):
                        try:
                            new_edge = bm.edges.new([arc_verts[j], arc_verts[j+1]])
                            mesh_edge_indices.append(len(bm.edges) - 1)
                        except ValueError:
                            pass
                    
                    # Store mapping for later attribute assignment
                    graph_edge_to_mesh_edges[(src, tgt)] = mesh_edge_indices
                    
                    created_edges.add(edge_key)
                    edges_created += 1
            
            if (edges_created % 100) == 0 and edges_created > 0:
                log(f"    Created {edges_created:,} arcs...")
    
    else:
        # Straight line edges
        log("  Creating straight line edges...")
        
        for edge_idx, (src, tgt) in enumerate(graph_data.edges):
            if remove_self_loops and src == tgt:
                continue
            
            if src in node_to_vert and tgt in node_to_vert:
                edge_key = (min(src, tgt), max(src, tgt))
                
                if edge_key not in created_edges:
                    try:
                        current_edge_count = len(bm.edges)
                        bm.edges.new([node_to_vert[src], node_to_vert[tgt]])
                        
                        # Store mapping: this graph edge maps to this single mesh edge
                        graph_edge_to_mesh_edges[(src, tgt)] = [current_edge_count]
                        
                        created_edges.add(edge_key)
                        edges_created += 1
                    except ValueError:
                        pass
    
    log(f"  Created {edges_created:,} edges in {time.time() - edges_start:.2f}s")
    
    # Convert to mesh
    bm.to_mesh(mesh)
    bm.free()
    
    # Create object
    obj = bpy.data.objects.new("GeoGraph_Object", mesh)
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    
    # Add edge attributes from CSV data
    log("  Adding edge attributes from CSV...")
    
    # Get unique attribute columns from dataframe (excluding source/target)
    if hasattr(graph_data, 'dataframe') and graph_data.dataframe is not None:
        df = graph_data.dataframe
        
        # Get source and target column names
        source_col_name = getattr(graph_data, 'source_column_name', df.columns[0])
        target_col_name = getattr(graph_data, 'target_column_name', df.columns[1])
        
        # Determine which columns to import as attributes
        candidate_cols = []
        if selected_attributes is None:
            # Import all columns except source/target
            candidate_cols = [col for col in df.columns 
                            if col not in [source_col_name, target_col_name]]
        else:
            # Import only selected columns
            candidate_cols = [col for col in selected_attributes 
                            if col not in [source_col_name, target_col_name]]
        
        # Filter to columns that can be converted to numeric
        numeric_cols = []
        for col in candidate_cols:
            try:
                # Try to convert to numeric
                converted = pd.to_numeric(df[col], errors='coerce')
                non_null_count = converted.notna().sum()
                total_count = len(df[col])
                
                # If at least 50% of values are numeric, include it
                if total_count > 0 and (non_null_count / total_count) > 0.5:
                    numeric_cols.append(col)
            except:
                pass
        
        attrs_added = 0
        for col in numeric_cols:
            # Create an EDGE attribute for this column
            try:
                attribute = obj.data.attributes.new(name=col, type='FLOAT', domain='EDGE')
                
                # For each row in the dataframe, find the corresponding mesh edges
                for row_idx, row in df.iterrows():
                    src = row[source_col_name]
                    tgt = row[target_col_name]
                    
                    # Get the value for this column (convert to numeric)
                    value_raw = pd.to_numeric(row[col], errors='coerce')
                    
                    if pd.notna(value_raw):
                        value = float(value_raw)
                        
                        if not np.isnan(value) and not np.isinf(value):
                            # Find mesh edges corresponding to this graph edge
                            mesh_edge_indices = graph_edge_to_mesh_edges.get((src, tgt), [])
                            
                            # Assign value to all mesh edges representing this connection
                            for mesh_edge_idx in mesh_edge_indices:
                                if mesh_edge_idx < len(attribute.data):
                                    attribute.data[mesh_edge_idx].value = value
                
                attrs_added += 1
                log(f"    Added edge attribute '{col}' to {len(graph_edge_to_mesh_edges)} connections")
            except Exception as e:
                log(f"    Warning: Could not add edge attribute {col}: {e}")
        
        if attrs_added > 0:
            log(f"    Added {attrs_added} edge attributes to Spreadsheet")
        else:
            log(f"    No numeric attributes found to import")
    
    # Store metadata
    obj["num_nodes"] = num_nodes
    obj["is_geospatial"] = True
    obj["edge_style"] = edge_style
    
    # Store node positions as flat array
    positions_list = []
    for node in graph_data.nodes:
        node_str = str(node)
        if node_str in positions_3d:
            pos = positions_3d[node_str]
            positions_list.extend([pos[0], pos[1], pos[2]])
        else:
            positions_list.extend([0, 0, 0])
    
    obj["node_positions"] = positions_list
    obj["nodes_data"] = ",".join(str(node) for node in graph_data.nodes)
    
    edges_flat = []
    stored_edge_count = 0
    for src, tgt in graph_data.edges:
        if remove_self_loops and str(src) == str(tgt):
            continue
        edges_flat.append(str(src))
        edges_flat.append(str(tgt))
        stored_edge_count += 1
    obj["edges_data"] = ",".join(edges_flat)
    obj["num_edges"] = stored_edge_count
    
    # Store edge weights if available
    if hasattr(graph_data, 'edge_weights') and graph_data.edge_weights is not None:
        try:
            # Convert to float array to ensure compatibility
            weights = np.array(graph_data.edge_weights, dtype=np.float64)
            obj["edge_weights"] = weights.tolist()
        except Exception as e:
            log(f"Warning: Could not store edge weights: {e}")
    
    # Create VERTEX attributes from CSV data
    if hasattr(graph_data, 'dataframe') and graph_data.dataframe is not None:
        log("  Creating vertex attributes...")
        _create_geospatial_vertex_attributes(obj, graph_data, numeric_cols if 'numeric_cols' in dir() else [])
    
    log(f"Total geospatial graph creation time: {time.time() - start_time:.2f}s")
    
    return obj


def _create_geospatial_vertex_attributes(obj, graph_data, numeric_cols):
    """
    Create vertex attributes for geospatial graphs.
    
    For each numeric column, creates aggregated vertex attributes:
    - vertex_{col}_sum, vertex_{col}_mean, vertex_{col}_min, vertex_{col}_max, vertex_{col}_count
    """
    import json
    
    df = graph_data.dataframe
    if df is None or len(df) == 0:
        return
    
    mesh = obj.data
    
    source_col_name = getattr(graph_data, 'source_column_name', df.columns[0])
    target_col_name = getattr(graph_data, 'target_column_name', df.columns[1])
    
    # Build node to vertex index mapping
    # In geospatial graphs, only "real" nodes have vertices (marked with is_real_node=1)
    node_to_idx = {}
    nodes_data = obj.get("nodes_data", "")
    if nodes_data:
        node_list = nodes_data.split(",")
        for i, node in enumerate(node_list):
            node_to_idx[node.strip()] = i
    
    num_real_nodes = len(node_to_idx)
    
    if num_real_nodes == 0:
        log("  Warning: No nodes found for vertex attributes")
        return
    
    # Determine columns to process
    cols_to_process = numeric_cols if numeric_cols else []
    if not cols_to_process:
        # Find numeric columns automatically
        for col in df.columns:
            if col in [source_col_name, target_col_name]:
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                cols_to_process.append(col)
    
    attrs_created = 0
    for col_name in cols_to_process:
        try:
            _create_vertex_attr_for_geospatial(
                mesh, df, col_name, source_col_name, target_col_name, 
                node_to_idx, num_real_nodes
            )
            attrs_created += 1
        except Exception as e:
            log(f"    Warning: Could not create vertex attributes for '{col_name}': {e}")
    
    # Store node_names as JSON for temporal filtering
    obj["node_names"] = json.dumps(list(node_to_idx.keys()))
    
    if attrs_created > 0:
        log(f"    Created vertex attributes for {attrs_created} columns")


def _create_vertex_attr_for_geospatial(mesh, df, col_name, source_col_name, target_col_name,
                                        node_to_idx, num_vertices):
    """
    Create vertex attributes for a single column in geospatial graph.
    Uses efficient pandas groupby operations.
    """
    vertex_sums = np.zeros(num_vertices, dtype=np.float64)
    vertex_counts = np.zeros(num_vertices, dtype=np.int32)
    vertex_mins = np.full(num_vertices, np.inf, dtype=np.float64)
    vertex_maxs = np.full(num_vertices, -np.inf, dtype=np.float64)
    
    work_df = df[[source_col_name, target_col_name, col_name]].copy()
    work_df[col_name] = pd.to_numeric(work_df[col_name], errors='coerce')
    work_df = work_df.dropna(subset=[col_name])
    
    if len(work_df) == 0:
        return
    
    for node_col in [source_col_name, target_col_name]:
        grouped = work_df.groupby(node_col)[col_name].agg(['sum', 'count', 'min', 'max'])
        
        for node_name, row in grouped.iterrows():
            node_key = str(node_name)
            if node_key in node_to_idx:
                idx = node_to_idx[node_key]
                vertex_sums[idx] += row['sum']
                vertex_counts[idx] += int(row['count'])
                if row['min'] < vertex_mins[idx]:
                    vertex_mins[idx] = row['min']
                if row['max'] > vertex_maxs[idx]:
                    vertex_maxs[idx] = row['max']
    
    vertex_mins[vertex_mins == np.inf] = 0.0
    vertex_maxs[vertex_maxs == -np.inf] = 0.0
    
    vertex_means = np.divide(vertex_sums, vertex_counts, 
                             out=np.zeros_like(vertex_sums), 
                             where=vertex_counts > 0)
    
    attr_configs = [
        (f"vertex_{col_name}_sum", vertex_sums, 'FLOAT'),
        (f"vertex_{col_name}_mean", vertex_means, 'FLOAT'),
        (f"vertex_{col_name}_min", vertex_mins, 'FLOAT'),
        (f"vertex_{col_name}_max", vertex_maxs, 'FLOAT'),
        (f"vertex_{col_name}_count", vertex_counts, 'INT'),
    ]
    
    total_mesh_verts = len(mesh.vertices)
    
    for attr_name, values, attr_type in attr_configs:
        if attr_name not in mesh.attributes:
            attr = mesh.attributes.new(name=attr_name, type=attr_type, domain='POINT')
            
            # Expand values array to match total mesh vertices
            # Extra vertices (arc points) get value 0
            if len(values) < total_mesh_verts:
                expanded = np.zeros(total_mesh_verts, dtype=values.dtype)
                expanded[:len(values)] = values
                values = expanded
            
            attr.data.foreach_set("value", values.tolist())


# ============================================================================
# OSMnx GRAPH VISUALIZATION
# ============================================================================

def create_osmnx_graph_object(graph_data, edge_geometries, scale=0.001, retain_geometry=True):
    """
    Creates a graph object from OSMnx data with curved street geometries.
    
    Args:
        graph_data: GraphData object with node_coordinates and osmnx metadata
        edge_geometries: Dict mapping (u, v) -> list of (lat, lon) coordinate tuples
        scale: Scale factor to convert meters to Blender units
        retain_geometry: Whether to create curved streets or straight edges
    
    Returns:
        Created Blender object
    """
    import time
    start_time = time.time()
    
    num_nodes = len(graph_data.nodes)
    num_edges = len(graph_data.edges)
    
    log(f"\nCreating OSMnx graph: {num_nodes:,} nodes, {num_edges:,} edges...")
    
    # Create mesh
    mesh = bpy.data.meshes.new(name="OSMnx_Mesh")
    bm = bmesh.new()
    
    # Custom layer to mark real intersection nodes
    is_intersection_layer = bm.verts.layers.int.new("is_intersection")
    
    # Convert geographic coordinates to 3D positions
    # Use a local coordinate system centered on the network
    node_positions, center_lat, center_lon = _convert_osmnx_coords_to_3d(graph_data.node_coordinates, scale)
    
    # Create vertices for intersection nodes
    node_verts = {}
    
    for node_id in graph_data.nodes:
        if node_id in node_positions:
            pos = node_positions[node_id]
            v = bm.verts.new(pos)
            v[is_intersection_layer] = 1  # Mark as real intersection
            node_verts[node_id] = v
    
    bm.verts.ensure_lookup_table()
    log(f"  Created {len(node_verts):,} intersection vertices")
    
    # Create edges with curved geometry
    edges_start = time.time()
    edges_created = 0
    created_edge_keys = set()
    
    for src, tgt in graph_data.edges:
        if src not in node_verts or tgt not in node_verts:
            continue
        
        # Avoid duplicate edges in undirected representation
        edge_key = (min(src, tgt), max(src, tgt))
        if edge_key in created_edge_keys:
            continue
        
        created_edge_keys.add(edge_key)
        
        # Check if we have curved geometry for this edge
        if retain_geometry and (src, tgt) in edge_geometries:
            geom_coords = edge_geometries[(src, tgt)]
            _create_curved_edge(bm, node_verts, src, tgt, geom_coords, 
                               node_positions, scale, is_intersection_layer)
        elif retain_geometry and (tgt, src) in edge_geometries:
            # Check reversed direction
            geom_coords = edge_geometries[(tgt, src)]
            # Reverse the coordinates
            geom_coords = list(reversed(geom_coords))
            _create_curved_edge(bm, node_verts, src, tgt, geom_coords, 
                               node_positions, scale, is_intersection_layer)
        else:
            # Straight edge
            try:
                bm.edges.new([node_verts[src], node_verts[tgt]])
            except ValueError:
                pass  # Edge already exists
        
        edges_created += 1
        
        if edges_created % 1000 == 0:
            log(f"    Created {edges_created:,} edges...")
    
    log(f"  Created {edges_created:,} street edges in {time.time() - edges_start:.2f}s")
    
    # Convert to mesh
    bm.to_mesh(mesh)
    bm.free()
    
    # Create object
    obj = bpy.data.objects.new("OSMnx_StreetNetwork", mesh)
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    
    # Store metadata
    obj["num_nodes"] = num_nodes
    obj["num_edges"] = num_edges
    obj["is_osmnx"] = True
    obj["scale"] = scale
    obj["num_mesh_verts"] = len(mesh.vertices)  # Total vertices including curve points
    obj["osmnx_center_lat"] = center_lat
    obj["osmnx_center_lon"] = center_lon
    # OSMnx graphs are MultiDiGraphs by construction (oneway / turn restrictions
    # are encoded as directed edges). Mark the Blender object accordingly so the
    # "Directed Analysis" panels, direction arrows and direction-aware algorithms
    # match the actual graph stored in the cache.
    obj["is_directed"] = bool(getattr(graph_data, "is_directed", True))
    
    # Store node IDs
    obj["nodes_data"] = ",".join(str(n) for n in graph_data.nodes)
    
    # Store edges data for graph reconstruction
    edges_flat = []
    for src, tgt in graph_data.edges:
        edges_flat.append(str(src))
        edges_flat.append(str(tgt))
    obj["edges_data"] = ",".join(edges_flat)
    
    # Store total street length
    if hasattr(graph_data, 'edge_lengths'):
        total_length = sum(graph_data.edge_lengths)
        obj["total_length_m"] = total_length
    
    log(f"Total OSMnx graph creation time: {time.time() - start_time:.2f}s")
    
    return obj


def _convert_osmnx_coords_to_3d(node_coordinates, scale):
    """
    Convert lat/lon coordinates to local 3D positions.
    Uses a simple equirectangular projection centered on the network centroid.
    
    Args:
        node_coordinates: Dict mapping node_id -> (lat, lon)
        scale: Scale factor (meters to Blender units)
    
    Returns:
        Tuple: (positions dict, center_lat, center_lon)
    """
    if not node_coordinates:
        return {}, 0.0, 0.0
    
    # Calculate centroid
    lats = [coord[0] for coord in node_coordinates.values()]
    lons = [coord[1] for coord in node_coordinates.values()]
    
    center_lat = sum(lats) / len(lats)
    center_lon = sum(lons) / len(lons)
    
    # Earth radius in meters
    EARTH_RADIUS = 6371000.0
    
    # Convert to local coordinates
    positions = {}
    
    # Correction factor for longitude based on latitude
    cos_lat = np.cos(np.radians(center_lat))
    
    for node_id, (lat, lon) in node_coordinates.items():
        # Convert lat/lon to meters from centroid
        y_m = (lat - center_lat) * (np.pi / 180.0) * EARTH_RADIUS
        x_m = (lon - center_lon) * (np.pi / 180.0) * EARTH_RADIUS * cos_lat
        
        # Apply scale
        x = x_m * scale
        y = y_m * scale
        z = 0.0  # Flat projection (could add elevation later)
        
        positions[node_id] = (x, y, z)
    
    return positions, center_lat, center_lon


def _blender_coords_to_latlon(x, y, center_lat, center_lon, scale):
    """
    Convert Blender XY coordinates back to lat/lon.
    Reverse of _convert_osmnx_coords_to_3d transformation.
    
    Args:
        x, y: Blender coordinates
        center_lat, center_lon: Center of the projection
        scale: Scale factor (meters to Blender units)
    
    Returns:
        Tuple (lat, lon)
    """
    EARTH_RADIUS = 6371000.0
    cos_lat = np.cos(np.radians(center_lat))
    
    x_m = x / scale
    y_m = y / scale
    
    lat = center_lat + (y_m / (np.pi / 180.0 * EARTH_RADIUS))
    lon = center_lon + (x_m / (np.pi / 180.0 * EARTH_RADIUS * cos_lat))
    
    return lat, lon


def _latlon_to_local_3d(lat, lon, center_lat, center_lon, scale):
    """
    Convert a single lat/lon coordinate to local 3D position.
    
    Args:
        lat, lon: Coordinates to convert
        center_lat, center_lon: Center of the projection
        scale: Scale factor
    
    Returns:
        Tuple (x, y, z) position
    """
    EARTH_RADIUS = 6371000.0
    cos_lat = np.cos(np.radians(center_lat))
    
    y_m = (lat - center_lat) * (np.pi / 180.0) * EARTH_RADIUS
    x_m = (lon - center_lon) * (np.pi / 180.0) * EARTH_RADIUS * cos_lat
    
    return (x_m * scale, y_m * scale, 0.0)


def _local_3d_to_latlon(x, y, center_lat, center_lon, scale):
    """
    Convert a local 3D position back to lat/lon coordinates.
    
    Args:
        x, y: Local coordinates
        center_lat, center_lon: Center of the projection
        scale: Scale factor
    
    Returns:
        Tuple (lat, lon) coordinates
    """
    EARTH_RADIUS = 6371000.0
    cos_lat = np.cos(np.radians(center_lat))
    
    x_m = x / scale
    y_m = y / scale
    
    lat = center_lat + (y_m / EARTH_RADIUS) * (180.0 / np.pi)
    lon = center_lon + (x_m / EARTH_RADIUS / cos_lat) * (180.0 / np.pi)
    
    return (lat, lon)


def _create_curved_edge(bm, node_verts, src, tgt, geom_coords, node_positions, scale, is_intersection_layer):
    """
    Create a curved edge following street geometry.
    
    Args:
        bm: BMesh object
        node_verts: Dict mapping node_id -> BMVert for intersection nodes
        src, tgt: Source and target node IDs
        geom_coords: List of (lat, lon) coordinates along the street
        node_positions: Dict mapping node_id -> (x, y, z) already converted positions
        scale: Scale factor
        is_intersection_layer: BMesh vertex layer for marking intersections
    """
    if len(geom_coords) < 2:
        # Not enough points for a curve, create straight edge
        try:
            bm.edges.new([node_verts[src], node_verts[tgt]])
        except ValueError:
            pass
        return
    
    # Calculate center for local coordinate conversion
    all_coords = list(node_positions.values())
    if all_coords:
        center_x = sum(p[0] for p in all_coords) / len(all_coords)
        center_y = sum(p[1] for p in all_coords) / len(all_coords)
        
        # Reverse-calculate center lat/lon from any node
        # This is approximate but works for local areas
        sample_node = list(node_positions.keys())[0]
        sample_lat, sample_lon = list(geom_coords)[0] if geom_coords else (0, 0)
    
    # Get center lat/lon from the geometry coords
    center_lat = sum(c[0] for c in geom_coords) / len(geom_coords)
    center_lon = sum(c[1] for c in geom_coords) / len(geom_coords)
    
    # Create intermediate vertices along the curve
    prev_vert = node_verts[src]
    
    for i, (lat, lon) in enumerate(geom_coords[1:-1], 1):
        pos = _latlon_to_local_3d(lat, lon, center_lat, center_lon, scale)
        
        # Adjust position relative to the node positions
        # (compensate for potential offset)
        src_pos = node_positions[src]
        first_geom_pos = _latlon_to_local_3d(
            geom_coords[0][0], geom_coords[0][1], 
            center_lat, center_lon, scale
        )
        
        offset_x = src_pos[0] - first_geom_pos[0]
        offset_y = src_pos[1] - first_geom_pos[1]
        
        adjusted_pos = (pos[0] + offset_x, pos[1] + offset_y, pos[2])
        
        new_vert = bm.verts.new(adjusted_pos)
        new_vert[is_intersection_layer] = 0  # Not a real intersection
        
        try:
            bm.edges.new([prev_vert, new_vert])
        except ValueError:
            pass
        
        prev_vert = new_vert
    
    # Connect last intermediate vertex to target
    try:
        bm.edges.new([prev_vert, node_verts[tgt]])
    except ValueError:
        pass


# ============================================================================
# INTERACTIVE GEOMETRY NODES VISUALIZATION SYSTEM
# ============================================================================

def update_geometry_nodes_parameters(obj):
    """
    Updates Geometry Nodes modifier parameters in real-time.
    Called by property update callbacks for instant visual feedback.
    
    This function reads values from scene.scigraphs_viz and applies them
    to the modifier's input sockets without rebuilding the node tree.
    """
    mod = obj.modifiers.get("SciGraphs_Viz")
    if not mod or not mod.node_group:
        return
    
    props = bpy.context.scene.scigraphs_viz
    node_group = mod.node_group
    
    # Build a mapping of socket names to identifiers
    socket_map = {}
    for item in node_group.interface.items_tree:
        if item.item_type == 'SOCKET' and item.in_out == 'INPUT':
            socket_map[item.name] = item.identifier
    
    # Property to socket mapping
    shape_map = {'SPHERE': 0, 'ICOSPHERE': 1, 'CUBE': 2, 'CONE': 3, 'CYLINDER': 4}
    
    param_values = {
        "Node Scale": props.node_scale,
        "Node Resolution": props.node_resolution,
        "Node Shape": shape_map.get(props.node_shape, 0),
        "Node Attr Name": props.node_scale_attribute if props.node_scale_attribute != 'NONE' else "",
        "Node Attr Mult": props.node_scale_multiplier,
        "Edge Thickness": props.edge_thickness,
        "Edge Resolution": props.edge_resolution,
        "Edge Attr Name": props.edge_scale_attribute if props.edge_scale_attribute != 'NONE' else "",
        "Edge Attr Mult": props.edge_thickness_multiplier,
        "Show Arrows": props.show_arrows,
        "Arrow Size": props.arrow_size,
        "Arrow Position": props.arrow_position,
        "Filter Enable": props.enable_filtering,
        "Filter Min": props.filter_min,
        "Filter Max": props.filter_max,
        "Filter Attr": props.filter_attribute if props.filter_attribute != 'NONE' else "",
    }
    
    # Apply values to modifier inputs using Blender 4.x API
    for socket_name, value in param_values.items():
        if socket_name in socket_map:
            identifier = socket_map[socket_name]
            # In Blender 4.x, modifier inputs are accessed via the identifier
            try:
                mod[identifier] = value
            except (KeyError, TypeError):
                # Socket might not exist or type mismatch, skip silently
                pass
    
    # Force viewport update
    if obj.data:
        obj.data.update()
    
    # Tag redraw for all 3D views
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()


def setup_interactive_geometry_nodes(obj):
    """
    Creates an advanced Geometry Nodes tree for interactive visualization.
    
    Features:
    - Configurable node shapes (sphere, cube, icosphere, etc.)
    - Attribute-driven node scaling
    - Attribute-driven edge thickness
    - Direction arrows for directed graphs
    - Real-time filtering by attribute values
    
    The tree exposes inputs that can be controlled from the UI panel.
    """
    # Get or create modifier
    mod = obj.modifiers.get("SciGraphs_Viz")
    if mod is None:
        mod = obj.modifiers.new(name="SciGraphs_Viz", type='NODES')
    
    # Create or reuse node group
    tree_name = "SciGraphs_Interactive_Viz"
    if mod.node_group is not None:
        node_group = mod.node_group
        node_group.nodes.clear()
        node_group.interface.clear()
    else:
        node_group = bpy.data.node_groups.new(tree_name, 'GeometryNodeTree')
        mod.node_group = node_group
    
    # Mark object as having scigraph visualization
    obj["is_scigraph"] = True
    
    # ========================================================================
    # DEFINE INPUT/OUTPUT INTERFACE
    # ========================================================================
    
    # Input geometry
    node_group.interface.new_socket("Geometry", in_out='INPUT', socket_type='NodeSocketGeometry')
    
    # Node parameters
    s = node_group.interface.new_socket("Node Scale", in_out='INPUT', socket_type='NodeSocketFloat')
    s.default_value = 0.1
    s.min_value = 0.001
    s.max_value = 5.0
    
    s = node_group.interface.new_socket("Node Resolution", in_out='INPUT', socket_type='NodeSocketInt')
    s.default_value = 3
    s.min_value = 1
    s.max_value = 6
    
    s = node_group.interface.new_socket("Node Shape", in_out='INPUT', socket_type='NodeSocketInt')
    s.default_value = 0
    
    node_group.interface.new_socket("Node Attr Name", in_out='INPUT', socket_type='NodeSocketString')
    
    s = node_group.interface.new_socket("Node Attr Mult", in_out='INPUT', socket_type='NodeSocketFloat')
    s.default_value = 1.0
    
    # Edge parameters
    s = node_group.interface.new_socket("Edge Thickness", in_out='INPUT', socket_type='NodeSocketFloat')
    s.default_value = 0.02
    s.min_value = 0.001
    s.max_value = 1.0
    
    s = node_group.interface.new_socket("Edge Resolution", in_out='INPUT', socket_type='NodeSocketInt')
    s.default_value = 4
    s.min_value = 3
    s.max_value = 32
    
    node_group.interface.new_socket("Edge Attr Name", in_out='INPUT', socket_type='NodeSocketString')
    
    s = node_group.interface.new_socket("Edge Attr Mult", in_out='INPUT', socket_type='NodeSocketFloat')
    s.default_value = 1.0
    
    # Arrow parameters
    s = node_group.interface.new_socket("Show Arrows", in_out='INPUT', socket_type='NodeSocketBool')
    s.default_value = False
    
    s = node_group.interface.new_socket("Arrow Size", in_out='INPUT', socket_type='NodeSocketFloat')
    s.default_value = 0.15
    
    s = node_group.interface.new_socket("Arrow Position", in_out='INPUT', socket_type='NodeSocketFloat')
    s.default_value = 0.7
    s.min_value = 0.0
    s.max_value = 1.0
    
    # Filter parameters
    s = node_group.interface.new_socket("Filter Enable", in_out='INPUT', socket_type='NodeSocketBool')
    s.default_value = False
    
    node_group.interface.new_socket("Filter Attr", in_out='INPUT', socket_type='NodeSocketString')
    
    s = node_group.interface.new_socket("Filter Min", in_out='INPUT', socket_type='NodeSocketFloat')
    s.default_value = 0.0
    
    s = node_group.interface.new_socket("Filter Max", in_out='INPUT', socket_type='NodeSocketFloat')
    s.default_value = 1.0
    
    # Output
    node_group.interface.new_socket("Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')
    
    # ========================================================================
    # BUILD NODE TREE
    # ========================================================================
    
    nodes = node_group.nodes
    links = node_group.links
    
    # Group Input/Output
    group_in = nodes.new('NodeGroupInput')
    group_in.location = (-1200, 0)
    
    group_out = nodes.new('NodeGroupOutput')
    group_out.location = (1200, 0)
    
    prepared_geo = _promote_edge_attributes_to_point(
        nodes, links, group_in.outputs['Geometry'], obj.data
    )
    
    # --------------------------------------------------------------------------
    # NODE VISUALIZATION (vertices -> spheres)
    # --------------------------------------------------------------------------
    
    is_osmnx = obj.get("is_osmnx", False)
    has_intersection_attr = "is_intersection" in obj.data.attributes if obj.data else False
    
    mesh_to_points = nodes.new('GeometryNodeMeshToPoints')
    mesh_to_points.location = (-900, 300)
    links.new(prepared_geo, mesh_to_points.inputs['Mesh'])
    
    # Create UV Sphere for node instances
    sphere = nodes.new('GeometryNodeMeshUVSphere')
    sphere.name = "SciGraphs_NodeSphere"
    sphere.location = (-900, 100)
    sphere.inputs['Radius'].default_value = 1.0  # Base radius, scaled by instance
    links.new(group_in.outputs['Node Resolution'], sphere.inputs['Segments'])
    links.new(group_in.outputs['Node Resolution'], sphere.inputs['Rings'])

    # Create Icosphere alternative
    ico_sphere = nodes.new('GeometryNodeMeshIcoSphere')
    ico_sphere.name = "SciGraphs_NodeIcoSphere"
    ico_sphere.location = (-900, -50)
    ico_sphere.inputs['Radius'].default_value = 1.0
    links.new(group_in.outputs['Node Resolution'], ico_sphere.inputs['Subdivisions'])

    # Create Cube alternative
    cube = nodes.new('GeometryNodeMeshCube')
    cube.name = "SciGraphs_NodeCube"
    cube.location = (-900, -200)
    cube.inputs['Size'].default_value = (1.0, 1.0, 1.0)

    # Create Cone alternative
    cone = nodes.new('GeometryNodeMeshCone')
    cone.name = "SciGraphs_NodeCone"
    cone.location = (-900, -350)
    cone.inputs['Radius Bottom'].default_value = 1.0
    cone.inputs['Radius Top'].default_value = 0.0
    cone.inputs['Depth'].default_value = 1.5
    links.new(group_in.outputs['Node Resolution'], cone.inputs['Vertices'])

    # Cylinder primitive (was previously a fallback to Sphere)
    cylinder = nodes.new('GeometryNodeMeshCylinder')
    cylinder.name = "SciGraphs_NodeCylinder"
    cylinder.location = (-900, -500)
    cylinder.inputs['Radius'].default_value = 1.0
    cylinder.inputs['Depth'].default_value = 1.5
    links.new(group_in.outputs['Node Resolution'], cylinder.inputs['Vertices'])

    # Shape switch (using Index Switch node for Blender 4.1+)
    shape_switch = nodes.new('GeometryNodeIndexSwitch')
    shape_switch.name = "SciGraphs_NodeShapeSwitch"
    shape_switch.data_type = 'GEOMETRY'
    shape_switch.location = (-600, 0)

    while len(shape_switch.inputs) < 6:
        shape_switch.index_switch_items.new()

    links.new(group_in.outputs['Node Shape'], shape_switch.inputs['Index'])
    links.new(sphere.outputs['Mesh'], shape_switch.inputs[1])       # 0 = Sphere
    links.new(ico_sphere.outputs['Mesh'], shape_switch.inputs[2])   # 1 = Icosphere
    links.new(cube.outputs['Mesh'], shape_switch.inputs[3])         # 2 = Cube
    links.new(cone.outputs['Mesh'], shape_switch.inputs[4])         # 3 = Cone
    links.new(cylinder.outputs['Mesh'], shape_switch.inputs[5])     # 4 = Cylinder

    # Smooth by Angle on the selected primitive (180°, ignore sharpness).
    # Falls back to Set Shade Smooth when the asset isn't available.
    smooth_by_angle = _add_smooth_by_angle_node(nodes, location=(-420, 0))
    links.new(shape_switch.outputs['Output'], smooth_by_angle.inputs['Geometry'])
    
    # --------------------------------------------------------------------------
    # ATTRIBUTE-BASED SCALING
    # --------------------------------------------------------------------------
    
    # Read named attribute for scaling
    attr_node = nodes.new('GeometryNodeInputNamedAttribute')
    attr_node.data_type = 'FLOAT'
    attr_node.location = (-600, 400)
    links.new(group_in.outputs['Node Attr Name'], attr_node.inputs['Name'])
    
    # Check if attribute name is empty (use string length)
    str_length = nodes.new('FunctionNodeStringLength')
    str_length.location = (-600, 500)
    links.new(group_in.outputs['Node Attr Name'], str_length.inputs['String'])
    
    # Compare string length > 0
    compare_str = nodes.new('FunctionNodeCompare')
    compare_str.data_type = 'INT'
    compare_str.operation = 'GREATER_THAN'
    compare_str.location = (-400, 500)
    compare_str.inputs['B'].default_value = 0
    links.new(str_length.outputs['Length'], compare_str.inputs['A'])
    
    # Multiply attribute by multiplier
    attr_mult = nodes.new('ShaderNodeMath')
    attr_mult.operation = 'MULTIPLY'
    attr_mult.location = (-400, 350)
    links.new(attr_node.outputs['Attribute'], attr_mult.inputs[0])
    links.new(group_in.outputs['Node Attr Mult'], attr_mult.inputs[1])
    
    # Switch between attribute scale and 1.0
    scale_switch = nodes.new('GeometryNodeSwitch')
    scale_switch.input_type = 'FLOAT'
    scale_switch.location = (-200, 400)
    scale_switch.inputs['False'].default_value = 1.0
    links.new(compare_str.outputs['Result'], scale_switch.inputs['Switch'])
    links.new(attr_mult.outputs['Value'], scale_switch.inputs['True'])
    
    # Multiply by base scale
    final_scale = nodes.new('ShaderNodeMath')
    final_scale.operation = 'MULTIPLY'
    final_scale.location = (0, 400)
    links.new(scale_switch.outputs['Output'], final_scale.inputs[0])
    links.new(group_in.outputs['Node Scale'], final_scale.inputs[1])
    
    # --------------------------------------------------------------------------
    # FILTERING (Optional)
    # --------------------------------------------------------------------------
    
    # Read filter attribute
    filter_attr = nodes.new('GeometryNodeInputNamedAttribute')
    filter_attr.data_type = 'FLOAT'
    filter_attr.location = (-600, 600)
    links.new(group_in.outputs['Filter Attr'], filter_attr.inputs['Name'])
    
    # Compare >= min
    compare_min = nodes.new('FunctionNodeCompare')
    compare_min.data_type = 'FLOAT'
    compare_min.operation = 'GREATER_EQUAL'
    compare_min.location = (-400, 650)
    links.new(filter_attr.outputs['Attribute'], compare_min.inputs['A'])
    links.new(group_in.outputs['Filter Min'], compare_min.inputs['B'])
    
    # Compare <= max
    compare_max = nodes.new('FunctionNodeCompare')
    compare_max.data_type = 'FLOAT'
    compare_max.operation = 'LESS_EQUAL'
    compare_max.location = (-400, 750)
    links.new(filter_attr.outputs['Attribute'], compare_max.inputs['A'])
    links.new(group_in.outputs['Filter Max'], compare_max.inputs['B'])
    
    # AND both conditions
    bool_and = nodes.new('FunctionNodeBooleanMath')
    bool_and.operation = 'AND'
    bool_and.location = (-200, 700)
    links.new(compare_min.outputs['Result'], bool_and.inputs[0])
    links.new(compare_max.outputs['Result'], bool_and.inputs[1])
    
    # Switch filter on/off
    filter_switch = nodes.new('GeometryNodeSwitch')
    filter_switch.input_type = 'BOOLEAN'
    filter_switch.location = (0, 700)
    filter_switch.inputs['False'].default_value = True  # No filtering = show all
    links.new(group_in.outputs['Filter Enable'], filter_switch.inputs['Switch'])
    links.new(bool_and.outputs['Boolean'], filter_switch.inputs['True'])
    
    # Selection for OSMnx (only intersections)
    if is_osmnx or has_intersection_attr:
        intersection_attr = nodes.new('GeometryNodeInputNamedAttribute')
        intersection_attr.data_type = 'INT'
        intersection_attr.location = (-600, 800)
        intersection_attr.inputs['Name'].default_value = "is_intersection"
        
        compare_intersection = nodes.new('FunctionNodeCompare')
        compare_intersection.data_type = 'INT'
        compare_intersection.operation = 'EQUAL'
        compare_intersection.location = (-400, 850)
        compare_intersection.inputs['B'].default_value = 1
        links.new(intersection_attr.outputs['Attribute'], compare_intersection.inputs['A'])
        
        # Combine with filter
        combined_filter = nodes.new('FunctionNodeBooleanMath')
        combined_filter.operation = 'AND'
        combined_filter.location = (200, 750)
        links.new(filter_switch.outputs['Output'], combined_filter.inputs[0])
        links.new(compare_intersection.outputs['Result'], combined_filter.inputs[1])
        
        final_selection = combined_filter.outputs['Boolean']
    else:
        final_selection = filter_switch.outputs['Output']
    
    # --------------------------------------------------------------------------
    # INSTANCE NODES ON POINTS
    # --------------------------------------------------------------------------
    
    instance_on_points = nodes.new('GeometryNodeInstanceOnPoints')
    instance_on_points.location = (200, 300)
    links.new(mesh_to_points.outputs['Points'], instance_on_points.inputs['Points'])
    links.new(smooth_by_angle.outputs['Geometry'], instance_on_points.inputs['Instance'])
    links.new(final_scale.outputs['Value'], instance_on_points.inputs['Scale'])
    links.new(final_selection, instance_on_points.inputs['Selection'])
    
    # Realize instances for proper rendering
    realize_nodes = nodes.new('GeometryNodeRealizeInstances')
    realize_nodes.location = (400, 300)
    links.new(instance_on_points.outputs['Instances'], realize_nodes.inputs['Geometry'])
    
    # --------------------------------------------------------------------------
    # EDGE VISUALIZATION (edges -> tubes)
    # --------------------------------------------------------------------------
    
    # Convert each graph edge to its own curve
    edge_geo = _split_edges_for_individual_curves(
        nodes, links, prepared_geo, (-1100, -500), mesh=obj.data
    )

    mesh_to_curve = nodes.new('GeometryNodeMeshToCurve')
    mesh_to_curve.location = (-900, -500)
    links.new(edge_geo, mesh_to_curve.inputs['Mesh'])
    
    # Read edge attribute for thickness
    edge_attr = nodes.new('GeometryNodeInputNamedAttribute')
    edge_attr.data_type = 'FLOAT'
    edge_attr.location = (-700, -400)
    links.new(group_in.outputs['Edge Attr Name'], edge_attr.inputs['Name'])
    
    # Check if edge attribute name is empty
    edge_str_len = nodes.new('FunctionNodeStringLength')
    edge_str_len.location = (-700, -300)
    links.new(group_in.outputs['Edge Attr Name'], edge_str_len.inputs['String'])
    
    compare_edge_str = nodes.new('FunctionNodeCompare')
    compare_edge_str.data_type = 'INT'
    compare_edge_str.operation = 'GREATER_THAN'
    compare_edge_str.location = (-500, -300)
    compare_edge_str.inputs['B'].default_value = 0
    links.new(edge_str_len.outputs['Length'], compare_edge_str.inputs['A'])
    
    # Multiply edge attribute by multiplier
    edge_attr_mult = nodes.new('ShaderNodeMath')
    edge_attr_mult.operation = 'MULTIPLY'
    edge_attr_mult.location = (-500, -450)
    links.new(edge_attr.outputs['Attribute'], edge_attr_mult.inputs[0])
    links.new(group_in.outputs['Edge Attr Mult'], edge_attr_mult.inputs[1])
    
    # Switch between attribute thickness and 1.0
    edge_scale_switch = nodes.new('GeometryNodeSwitch')
    edge_scale_switch.input_type = 'FLOAT'
    edge_scale_switch.location = (-300, -400)
    edge_scale_switch.inputs['False'].default_value = 1.0
    links.new(compare_edge_str.outputs['Result'], edge_scale_switch.inputs['Switch'])
    links.new(edge_attr_mult.outputs['Value'], edge_scale_switch.inputs['True'])
    
    # Final edge thickness
    final_edge_thick = nodes.new('ShaderNodeMath')
    final_edge_thick.operation = 'MULTIPLY'
    final_edge_thick.location = (-100, -400)
    links.new(edge_scale_switch.outputs['Output'], final_edge_thick.inputs[0])
    links.new(group_in.outputs['Edge Thickness'], final_edge_thick.inputs[1])
    
    # Set curve radius
    set_radius = nodes.new('GeometryNodeSetCurveRadius')
    set_radius.location = (100, -500)
    links.new(mesh_to_curve.outputs['Curve'], set_radius.inputs['Curve'])
    links.new(final_edge_thick.outputs['Value'], set_radius.inputs['Radius'])
    
    # Create curve profile (circle)
    curve_circle = nodes.new('GeometryNodeCurvePrimitiveCircle')
    curve_circle.name = "SciGraphs_EdgeProfile"
    curve_circle.mode = 'RADIUS'
    curve_circle.location = (100, -700)
    curve_circle.inputs['Radius'].default_value = 1.0
    links.new(group_in.outputs['Edge Resolution'], curve_circle.inputs['Resolution'])
    
    # Convert curve to mesh with profile
    curve_to_mesh = nodes.new('GeometryNodeCurveToMesh')
    curve_to_mesh.location = (300, -500)
    links.new(set_radius.outputs['Curve'], curve_to_mesh.inputs['Curve'])
    links.new(curve_circle.outputs['Curve'], curve_to_mesh.inputs['Profile Curve'])
    
    # --------------------------------------------------------------------------
    # DIRECTION ARROWS (for directed graphs)
    # --------------------------------------------------------------------------
    
    # Sample curve at arrow position
    sample_curve = nodes.new('GeometryNodeSampleCurve')
    sample_curve.mode = 'FACTOR'
    sample_curve.location = (100, -900)
    links.new(mesh_to_curve.outputs['Curve'], sample_curve.inputs['Curves'])
    links.new(group_in.outputs['Arrow Position'], sample_curve.inputs['Factor'])
    
    # Create cone for arrows
    arrow_cone = nodes.new('GeometryNodeMeshCone')
    arrow_cone.location = (100, -1100)
    arrow_cone.inputs['Radius Bottom'].default_value = 0.5
    arrow_cone.inputs['Radius Top'].default_value = 0.0
    arrow_cone.inputs['Depth'].default_value = 1.0
    arrow_cone.inputs['Vertices'].default_value = 8
    links.new(group_in.outputs['Arrow Size'], arrow_cone.inputs['Radius Bottom'])
    
    # Align arrow to tangent
    align_euler = nodes.new('FunctionNodeAlignEulerToVector')
    align_euler.axis = 'Z'
    align_euler.location = (300, -950)
    links.new(sample_curve.outputs['Tangent'], align_euler.inputs['Vector'])
    
    # Points for arrow instances (spline endpoints)
    curve_to_points = nodes.new('GeometryNodeCurveToPoints')
    curve_to_points.mode = 'EVALUATED'
    curve_to_points.location = (300, -800)
    links.new(mesh_to_curve.outputs['Curve'], curve_to_points.inputs['Curve'])
    
    # Instance arrows (conditional on Show Arrows)
    instance_arrows = nodes.new('GeometryNodeInstanceOnPoints')
    instance_arrows.location = (500, -900)
    links.new(curve_to_points.outputs['Points'], instance_arrows.inputs['Points'])
    links.new(arrow_cone.outputs['Mesh'], instance_arrows.inputs['Instance'])
    links.new(align_euler.outputs['Rotation'], instance_arrows.inputs['Rotation'])
    links.new(group_in.outputs['Show Arrows'], instance_arrows.inputs['Selection'])
    
    # Realize arrow instances
    realize_arrows = nodes.new('GeometryNodeRealizeInstances')
    realize_arrows.location = (700, -900)
    links.new(instance_arrows.outputs['Instances'], realize_arrows.inputs['Geometry'])
    
    # --------------------------------------------------------------------------
    # JOIN ALL GEOMETRY
    # --------------------------------------------------------------------------
    
    edge_geo_output = _strip_custom_attributes_from_geo(
        nodes, links, curve_to_mesh.outputs['Mesh'], obj.data, (500, -500)
    )
    arrow_geo_output = _strip_custom_attributes_from_geo(
        nodes, links, realize_arrows.outputs['Geometry'], obj.data, (900, -900)
    )
    
    join_geo = nodes.new('GeometryNodeJoinGeometry')
    join_geo.location = (1100, 0)
    links.new(realize_nodes.outputs['Geometry'], join_geo.inputs['Geometry'])
    links.new(edge_geo_output, join_geo.inputs['Geometry'])
    links.new(arrow_geo_output, join_geo.inputs['Geometry'])
    
    # Final output
    links.new(join_geo.outputs['Geometry'], group_out.inputs['Geometry'])
    
    log("Interactive Geometry Nodes visualization set up successfully")
    
    # Apply initial values from properties
    if bpy.context.scene.get('scigraphs_viz'):
        update_geometry_nodes_parameters(obj)
    
    return mod


# ============================================================================
# EDGE STYLE APPLICATION SYSTEM
# ============================================================================

def _save_custom_attributes(mesh, node_indices, has_intersection_attr):
    """
    Read all custom mesh attributes before a destructive bmesh rebuild.

    POINT-domain attributes are saved indexed by new vertex position
    (matching node_indices ordering). EDGE-domain attributes are saved
    keyed by canonical node pair so they can be restored on any edge
    topology.

    Returns:
        (point_attrs, edge_attrs) where each is a dict of
        {name: (data_type, values_mapping)}.
    """
    old_to_new = {old: new for new, old in enumerate(node_indices)}
    node_set = set(node_indices)

    _SKIP_POINT = {'is_intersection', 'position', '.corner_vert', '.corner_edge', '.edge_verts'}

    point_attrs = {}
    for attr in mesh.attributes:
        if attr.domain != 'POINT':
            continue
        if attr.name in _SKIP_POINT or attr.name.startswith('.'):
            continue
        values = []
        for old_idx in node_indices:
            values.append(attr.data[old_idx].value if old_idx < len(attr.data) else 0)
        point_attrs[attr.name] = (attr.data_type, values)

    edge_attrs = {}
    for attr in mesh.attributes:
        if attr.domain != 'EDGE' or attr.name.startswith('.'):
            continue

        values = {}

        if not has_intersection_attr:
            for e in mesh.edges:
                v1, v2 = e.vertices
                if v1 in old_to_new and v2 in old_to_new:
                    key = (min(old_to_new[v1], old_to_new[v2]),
                           max(old_to_new[v1], old_to_new[v2]))
                    if key not in values:
                        values[key] = attr.data[e.index].value
        else:
            adjacency = {}
            for e in mesh.edges:
                v1, v2 = e.vertices
                adjacency.setdefault(v1, []).append((v2, e.index))
                adjacency.setdefault(v2, []).append((v1, e.index))

            visited = set()
            for start_old in node_indices:
                for neighbour, first_edge_idx in adjacency.get(start_old, []):
                    if first_edge_idx in visited:
                        continue
                    chain_val = attr.data[first_edge_idx].value
                    visited.add(first_edge_idx)
                    cur, prev = neighbour, start_old
                    while cur not in node_set:
                        moved = False
                        for nxt, eidx in adjacency.get(cur, []):
                            if nxt != prev and eidx not in visited:
                                visited.add(eidx)
                                prev, cur = cur, nxt
                                moved = True
                                break
                        if not moved:
                            break
                    if cur in node_set:
                        a, b = old_to_new[start_old], old_to_new[cur]
                        key = (min(a, b), max(a, b))
                        if key not in values:
                            values[key] = chain_val

        if values:
            edge_attrs[attr.name] = (attr.data_type, values)

    return point_attrs, edge_attrs


def _create_bmesh_layers(bm, point_attrs, edge_attrs):
    """
    Create bmesh custom data layers for every saved attribute.

    Returns:
        (vert_layers, edge_layers) dicts mapping attr_name -> bmesh layer.
    """
    _LAYER_ACCESSORS = {
        'FLOAT': (lambda bm: bm.verts.layers.float, lambda bm: bm.edges.layers.float),
        'INT': (lambda bm: bm.verts.layers.int, lambda bm: bm.edges.layers.int),
    }

    vert_layers = {}
    for name, (dtype, _) in point_attrs.items():
        accessor = _LAYER_ACCESSORS.get(dtype)
        if accessor:
            vert_layers[name] = accessor[0](bm).new(name)

    edge_layers = {}
    for name, (dtype, _) in edge_attrs.items():
        accessor = _LAYER_ACCESSORS.get(dtype)
        if accessor:
            edge_layers[name] = accessor[1](bm).new(name)

    return vert_layers, edge_layers


def _set_edge_attrs(edge, edge_key, edge_layers, edge_attrs):
    """
    Write saved attribute values onto a single bmesh edge.
    """
    for attr_name, layer in edge_layers.items():
        values_map = edge_attrs[attr_name][1]
        if edge_key in values_map:
            edge[layer] = values_map[edge_key]


def apply_edge_style_to_graph(obj, style_params: dict = None):
    """
    Apply edge styling to a graph object by modifying its mesh geometry.

    This function:
    1. Reads the original node positions (is_intersection=1 vertices)
    2. Saves all custom attributes (vertex and edge domain)
    3. Rebuilds the mesh with the selected style
    4. Restores all saved attributes on the new geometry

    Args:
        obj: Blender mesh object with graph data.
        style_params: Dictionary of style parameters (if None, reads from
            scene properties).

    Returns:
        True if successful, False otherwise.
    """
    from . import edge_styles

    if obj is None or obj.type != 'MESH':
        log("Error: No valid mesh object provided")
        return False

    if "num_nodes" not in obj:
        log("Error: Object is not a SciGraphs graph")
        return False

    if style_params is None:
        props = bpy.context.scene.scigraphs
        style_params = edge_styles.get_style_params_from_props(props)

    is_osmnx = obj.get("is_osmnx", False)
    if is_osmnx and style_params.get('preserve_osmnx', True):
        log("OSMnx graph detected - preserving original street geometry")
        return True

    mesh = obj.data
    has_intersection_attr = "is_intersection" in mesh.attributes

    node_positions = []
    node_indices = []

    if has_intersection_attr:
        is_intersection_attr = mesh.attributes["is_intersection"]
        for i, v in enumerate(mesh.vertices):
            if is_intersection_attr.data[i].value == 1:
                node_positions.append(np.array(v.co))
                node_indices.append(i)
    else:
        for i, v in enumerate(mesh.vertices):
            node_positions.append(np.array(v.co))
            node_indices.append(i)

    num_nodes = len(node_positions)
    log(f"Processing {num_nodes} graph nodes")

    saved_point_attrs, saved_edge_attrs = _save_custom_attributes(
        mesh, node_indices, has_intersection_attr
    )
    log(f"  Saved {len(saved_point_attrs)} point attributes, {len(saved_edge_attrs)} edge attributes")

    edges_data = obj.get("edges_data", "")
    nodes_data = obj.get("nodes_data", "")

    if not edges_data or not nodes_data:
        log("Warning: No edge data stored, attempting to reconstruct from mesh")
        original_edges = []
        for e in mesh.edges:
            v1, v2 = e.vertices[0], e.vertices[1]
            if has_intersection_attr:
                is_int1 = mesh.attributes["is_intersection"].data[v1].value == 1
                is_int2 = mesh.attributes["is_intersection"].data[v2].value == 1
                if is_int1 and is_int2:
                    original_edges.append((v1, v2))
            else:
                original_edges.append((v1, v2))
    else:
        nodes_list = nodes_data.split(",")
        node_to_idx = {node: i for i, node in enumerate(nodes_list)}
        edges_flat = edges_data.split(",")
        original_edges = []
        for i in range(0, len(edges_flat) - 1, 2):
            src, tgt = edges_flat[i], edges_flat[i + 1]
            if src in node_to_idx and tgt in node_to_idx:
                original_edges.append((node_to_idx[src], node_to_idx[tgt]))

    log(f"Processing {len(original_edges)} edges with style: {style_params['style_type']}")

    if style_params.get('auto_offset_parallel', True):
        from . import edge_styles as es
        parallel_groups = es.identify_parallel_edges(original_edges)
    else:
        parallel_groups = {}

    bm = bmesh.new()

    is_intersection_layer = bm.verts.layers.int.new("is_intersection")
    vert_layers, edge_layers = _create_bmesh_layers(bm, saved_point_attrs, saved_edge_attrs)

    new_verts = []
    for new_idx, pos in enumerate(node_positions):
        v = bm.verts.new(pos)
        v[is_intersection_layer] = 1
        for attr_name, layer in vert_layers.items():
            values_list = saved_point_attrs[attr_name][1]
            if new_idx < len(values_list):
                v[layer] = values_list[new_idx]
        new_verts.append(v)

    bm.verts.ensure_lookup_table()

    if style_params['style_type'] == 'BUNDLED':
        _apply_bundled_edges(bm, new_verts, original_edges, node_positions,
                            style_params, is_intersection_layer,
                            edge_layers, saved_edge_attrs)
    else:
        _apply_styled_edges(bm, new_verts, original_edges, node_positions,
                           style_params, parallel_groups, is_intersection_layer,
                           edge_layers, saved_edge_attrs)

    bm.to_mesh(mesh)
    bm.free()

    obj["edge_style_applied"] = style_params['style_type']
    obj["num_mesh_verts"] = len(mesh.vertices)
    obj["num_curve_verts"] = len(mesh.vertices) - num_nodes

    mesh.update()

    _rebuild_visualization_if_present(obj)

    log(f"Edge style '{style_params['style_type']}' applied successfully")
    log(f"  Total vertices: {len(mesh.vertices)} ({num_nodes} nodes + {len(mesh.vertices) - num_nodes} curve points)")

    return True


_VIZ_REBUILD_HOOKS = []


def register_viz_rebuild_hook(callback):
    """Subscribe ``callback(obj)`` to the post-rebuild notification.

    Hooks fire right after :func:`_rebuild_visualization_if_present` has
    regenerated the SciGraphs_Viz Geometry Nodes tree. They are useful for
    decoupled subsystems (e.g. the coloring toolbar) that need to restore
    their patches against the freshly built tree without touching this
    module directly.

    Hooks must be cheap and exception-safe.
    """
    if callback not in _VIZ_REBUILD_HOOKS:
        _VIZ_REBUILD_HOOKS.append(callback)


def unregister_viz_rebuild_hook(callback):
    if callback in _VIZ_REBUILD_HOOKS:
        _VIZ_REBUILD_HOOKS.remove(callback)


def _notify_viz_rebuild(obj):
    for callback in list(_VIZ_REBUILD_HOOKS):
        try:
            callback(obj)
        except (RuntimeError, AttributeError, TypeError) as exc:
            log(f"  WARNING: viz rebuild hook {callback!r} failed: {exc}")


def _rebuild_visualization_if_present(obj):
    """
    Rebuild the SciGraphs_Viz Geometry Nodes tree if it exists.

    A full rebuild (rather than incremental patching) guarantees that
    the is_intersection selection filter AND the attribute-stripping
    chain on the edge tube geometry are up to date with the current
    set of mesh attributes.  This is called after edge-style
    application, node-attribute import, or any operation that changes
    the attribute set on the mesh.
    """
    mod = obj.modifiers.get("SciGraphs_Viz")
    if not mod or not mod.node_group:
        return

    tree_name = mod.node_group.name

    if tree_name.startswith("SciGraphs_Interactive"):
        setup_interactive_geometry_nodes(obj)
        log("  Rebuilt interactive GN tree (attribute stripping updated)")
    else:
        setup_geometry_nodes_visualization(obj)
        log("  Rebuilt simple GN tree (attribute stripping updated)")

    _notify_viz_rebuild(obj)


def _apply_styled_edges(bm, verts, edges, positions, params, parallel_groups,
                        is_int_layer, edge_layers, saved_edge_attrs):
    """
    Apply styled edges (non-bundled) to the bmesh.
    """
    from . import edge_styles as es

    style_type = params['style_type']
    curvature = params['curvature']
    segments = params['segments']
    direction = params['direction']
    orthogonal_style = params.get('orthogonal_style', 'CENTERED')
    self_loop_radius = params.get('self_loop_radius', 0.2)
    parallel_offset = params.get('parallel_offset', 0.05)

    for edge_idx, (src_idx, tgt_idx) in enumerate(edges):
        if src_idx >= len(verts) or tgt_idx >= len(verts):
            continue

        p0 = positions[src_idx]
        p1 = positions[tgt_idx]

        edge_key = (min(src_idx, tgt_idx), max(src_idx, tgt_idx))
        if edge_key in parallel_groups and len(parallel_groups[edge_key]) > 1:
            edge_num = parallel_groups[edge_key].index(edge_idx)
            total_parallel = len(parallel_groups[edge_key])
            p0, p1 = es.offset_parallel_edge(p0, p1, edge_num, total_parallel, parallel_offset)

        intermediate_points = es.compute_styled_edge_points(
            p0, p1,
            style_type=style_type,
            curvature=curvature,
            segments=segments,
            direction=direction,
            edge_index=edge_idx,
            orthogonal_style=orthogonal_style,
            self_loop_radius=self_loop_radius
        )

        if intermediate_points:
            prev_vert = verts[src_idx]
            for pt in intermediate_points:
                new_vert = bm.verts.new(pt)
                new_vert[is_int_layer] = 0
                bm.verts.ensure_lookup_table()
                try:
                    new_edge = bm.edges.new([prev_vert, new_vert])
                    _set_edge_attrs(new_edge, edge_key, edge_layers, saved_edge_attrs)
                except ValueError:
                    pass
                prev_vert = new_vert
            try:
                new_edge = bm.edges.new([prev_vert, verts[tgt_idx]])
                _set_edge_attrs(new_edge, edge_key, edge_layers, saved_edge_attrs)
            except ValueError:
                pass
        else:
            try:
                new_edge = bm.edges.new([verts[src_idx], verts[tgt_idx]])
                _set_edge_attrs(new_edge, edge_key, edge_layers, saved_edge_attrs)
            except ValueError:
                pass


def _apply_bundled_edges(bm, verts, edges, positions, params, is_int_layer,
                         edge_layers, saved_edge_attrs):
    """
    Apply force-directed edge bundling to the bmesh.
    """
    from . import edge_styles as es

    edge_points = [(positions[src], positions[tgt]) for src, tgt in edges]

    bundled_points = es.bundle_edges_fdeb(
        edge_points,
        strength=params.get('bundle_strength', 0.6),
        iterations=params.get('bundle_iterations', 6),
        segments=params.get('segments', 10),
        compatibility_threshold=params.get('bundle_compatibility_threshold', 0.6)
    )

    for edge_idx, ((src_idx, tgt_idx), intermediate_pts) in enumerate(zip(edges, bundled_points)):
        if src_idx >= len(verts) or tgt_idx >= len(verts):
            continue

        edge_key = (min(src_idx, tgt_idx), max(src_idx, tgt_idx))

        if intermediate_pts:
            prev_vert = verts[src_idx]
            for pt in intermediate_pts:
                new_vert = bm.verts.new(pt)
                new_vert[is_int_layer] = 0
                bm.verts.ensure_lookup_table()
                try:
                    new_edge = bm.edges.new([prev_vert, new_vert])
                    _set_edge_attrs(new_edge, edge_key, edge_layers, saved_edge_attrs)
                except ValueError:
                    pass
                prev_vert = new_vert
            try:
                new_edge = bm.edges.new([prev_vert, verts[tgt_idx]])
                _set_edge_attrs(new_edge, edge_key, edge_layers, saved_edge_attrs)
            except ValueError:
                pass
        else:
            try:
                new_edge = bm.edges.new([verts[src_idx], verts[tgt_idx]])
                _set_edge_attrs(new_edge, edge_key, edge_layers, saved_edge_attrs)
            except ValueError:
                pass


def reset_edge_style(obj):
    """
    Reset edges to straight lines by removing intermediate vertices.
    
    Args:
        obj: Blender mesh object with graph data
        
    Returns:
        True if successful
    """
    return apply_edge_style_to_graph(obj, {
        'style_type': 'STRAIGHT',
        'curvature': 0.0,
        'segments': 1,
        'direction': 'AUTO',
        'auto_offset_parallel': False,
        'preserve_osmnx': False,
    })

