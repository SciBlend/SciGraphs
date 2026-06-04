"""
Graph utility operators for City2Graph.

Provides operators for graph filtering, clipping, isochrone generation,
and removing isolated components.
"""

import bpy


def _mesh_to_planar_polygon(mesh_obj):
    """Build a shapely polygon from a mesh object's faces using world XY coordinates.

    The boundary vertices are transformed by the object's world matrix so the
    polygon shares the same world coordinate space as the graph nodes, which are
    also compared in world space.
    """
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    mesh = mesh_obj.data
    matrix_world = mesh_obj.matrix_world
    polygons = []
    for face in mesh.polygons:
        coords = []
        for i in face.vertices:
            world_co = matrix_world @ mesh.vertices[i].co
            coords.append((world_co.x, world_co.y))
        if len(coords) >= 3:
            polygons.append(Polygon(coords))

    if not polygons:
        return None
    return unary_union(polygons)


def _planar_geometry_to_object(geometry, name, collection_name, matrix_world):
    """Create a Blender mesh object from a shapely polygon in planar XY coordinates.

    Vertices are written directly from the geometry coordinates without any
    geographic reprojection, and the source object's world matrix is applied so
    the result overlays the originating graph.
    """
    import bmesh
    from shapely.geometry import Polygon, MultiPolygon

    if geometry is None or geometry.is_empty:
        return None

    geometries = list(geometry.geoms) if isinstance(geometry, MultiPolygon) else [geometry]

    mesh = bpy.data.meshes.new(f"{name}_mesh")
    bm = bmesh.new()
    for poly in geometries:
        if not isinstance(poly, Polygon) or poly.is_empty:
            continue
        coords = list(poly.exterior.coords)[:-1]
        if len(coords) < 3:
            continue
        verts = [bm.verts.new((x, y, 0.0)) for x, y in coords]
        try:
            bm.faces.new(verts)
        except ValueError:
            for v in verts:
                bm.verts.remove(v)
    bm.to_mesh(mesh)
    bm.free()

    if len(mesh.vertices) == 0:
        return None

    obj = bpy.data.objects.new(name, mesh)

    collection = bpy.data.collections.get(collection_name)
    if collection is None:
        collection = bpy.data.collections.new(collection_name)
        bpy.context.scene.collection.children.link(collection)
    collection.objects.link(obj)

    obj.matrix_world = matrix_world.copy()
    return obj


def _graph_with_2d_positions(graph):
    """Return a copy of the graph whose node 'pos' attributes are 2D (x, y).

    Several city2graph spatial helpers snap a 2D center point against a KD-tree
    built from node positions, which requires the stored positions to be 2D.
    """
    duplicate = graph.copy()
    for _node, data in duplicate.nodes(data=True):
        pos = data.get("pos")
        if pos is not None and len(pos) >= 2:
            data["pos"] = (pos[0], pos[1])
    return duplicate


class SCIGRAPHS_OT_C2G_GraphToolApply(bpy.types.Operator):
    """Apply a graph utility tool to the selected graph object."""
    bl_idname = "scigraphs.c2g_graph_tool_apply"
    bl_label = "Apply Graph Tool"
    bl_description = "Apply the selected graph utility (clip, filter, isochrone, remove isolated)"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.active_object is not None and context.active_object.type == 'MESH'

    def execute(self, context):
        props = context.scene.city2graph
        action = props.graph_tool_action
        obj = context.active_object

        if action == 'REMOVE_ISOLATED':
            return self._remove_isolated(context, obj)
        elif action == 'FILTER':
            return self._filter_by_distance(context, obj, props)
        elif action == 'CLIP':
            return self._clip_graph(context, obj, props)
        elif action == 'ISOCHRONE':
            return self._isochrone(context, obj, props)

        self.report({'ERROR'}, f"Unknown action: {action}")
        return {'CANCELLED'}

    def _remove_isolated(self, context, obj):
        from ....core.city2graph import utils as c2g_utils
        import networkx as nx

        G = c2g_utils.extract_graph_from_blender(obj)
        if G is None:
            self.report({'ERROR'}, "Could not extract graph from object")
            return {'CANCELLED'}

        if G.number_of_nodes() == 0:
            self.report({'ERROR'}, "Graph has no nodes")
            return {'CANCELLED'}

        self.report({'INFO'}, "Removing isolated components...")

        components = list(nx.connected_components(G))
        if not components:
            self.report({'ERROR'}, "Failed to remove isolated components")
            return {'CANCELLED'}

        largest = max(components, key=len)
        result = G.subgraph(largest).copy()

        from ....core.city2graph.morphology import create_graph_from_networkx
        new_obj = create_graph_from_networkx(result, name=f"{obj.name}_Clean", use_positions=True, osmnx_obj=None)

        if new_obj:
            new_obj.matrix_world = obj.matrix_world.copy()
            new_obj["is_city2graph"] = True
            context.view_layer.objects.active = new_obj
            removed = G.number_of_nodes() - result.number_of_nodes()
            self.report({'INFO'}, f"Removed {removed} isolated nodes, {result.number_of_nodes()} remain")
        return {'FINISHED'}

    def _filter_by_distance(self, context, obj, props):
        from ....core.city2graph import utils as c2g_utils
        from shapely.geometry import Point

        G = c2g_utils.extract_graph_from_blender(obj)
        if G is None:
            self.report({'ERROR'}, "Could not extract graph")
            return {'CANCELLED'}

        center_obj = props.graph_filter_center
        if center_obj and center_obj.type == 'MESH':
            loc = center_obj.location
            center = Point(loc.x, loc.y)
        else:
            center = Point(0, 0)

        threshold = props.graph_filter_threshold

        self.report({'INFO'}, f"Filtering graph by distance ({threshold})...")
        result = c2g_utils.filter_graph_by_distance(
            _graph_with_2d_positions(G), center=center, threshold=threshold,
        )

        if result is None:
            self.report({'ERROR'}, "Filter failed")
            return {'CANCELLED'}

        kept = [node for node in result.nodes() if node in G]
        if not kept:
            self.report({'ERROR'}, "Filter produced an empty graph")
            return {'CANCELLED'}

        subgraph = G.subgraph(kept).copy()

        from ....core.city2graph.morphology import create_graph_from_networkx
        new_obj = create_graph_from_networkx(subgraph, name=f"{obj.name}_Filtered", use_positions=True, osmnx_obj=None)

        if new_obj:
            new_obj.matrix_world = obj.matrix_world.copy()
            new_obj["is_city2graph"] = True
            context.view_layer.objects.active = new_obj
            self.report({'INFO'}, f"Filtered graph: {subgraph.number_of_nodes()} nodes")
        return {'FINISHED'}

    def _clip_graph(self, context, obj, props):
        from ....core.city2graph import utils as c2g_utils
        from shapely.geometry import Point
        from mathutils import Vector

        G = c2g_utils.extract_graph_from_blender(obj)
        if G is None:
            self.report({'ERROR'}, "Could not extract graph")
            return {'CANCELLED'}

        clip_obj = props.graph_filter_center
        if not clip_obj or clip_obj.type != 'MESH':
            self.report({'ERROR'}, "Select a polygon object as clip boundary in 'Center/Clip Object'")
            return {'CANCELLED'}

        polygon = _mesh_to_planar_polygon(clip_obj)
        if polygon is None or polygon.is_empty:
            self.report({'ERROR'}, "Could not extract clip polygon (the boundary object needs faces)")
            return {'CANCELLED'}

        self.report({'INFO'}, "Clipping graph to polygon...")
        graph_matrix = obj.matrix_world
        kept = []
        for node, data in G.nodes(data=True):
            pos = data.get("pos", (0, 0, 0))
            local_co = Vector((pos[0], pos[1], pos[2] if len(pos) > 2 else 0.0))
            world_co = graph_matrix @ local_co
            if polygon.contains(Point(world_co.x, world_co.y)):
                kept.append(node)
        if not kept:
            self.report({'ERROR'}, "Clip produced an empty graph (no nodes inside the boundary)")
            return {'CANCELLED'}

        subgraph = G.subgraph(kept).copy()

        from ....core.city2graph.morphology import create_graph_from_networkx
        new_obj = create_graph_from_networkx(subgraph, name=f"{obj.name}_Clipped", use_positions=True, osmnx_obj=None)

        if new_obj:
            new_obj.matrix_world = obj.matrix_world.copy()
            new_obj["is_city2graph"] = True
            context.view_layer.objects.active = new_obj
            self.report({'INFO'}, f"Clipped graph: {subgraph.number_of_nodes()} nodes")
        return {'FINISHED'}

    def _isochrone(self, context, obj, props):
        from ....core.city2graph import utils as c2g_utils
        from shapely.geometry import Point

        G = c2g_utils.extract_graph_from_blender(obj)
        if G is None:
            self.report({'ERROR'}, "Could not extract graph")
            return {'CANCELLED'}

        threshold = props.isochrone_threshold
        weight_attr = props.isochrone_weight_attr.strip() or "length"

        center_obj_prop = props.isochrone_center_object
        if center_obj_prop and center_obj_prop.type == 'MESH':
            loc = center_obj_prop.location
            center_point = Point(loc.x, loc.y)
        else:
            first_node = next(iter(G.nodes), None)
            if first_node is None:
                self.report({'ERROR'}, "Graph has no nodes")
                return {'CANCELLED'}
            pos = G.nodes[first_node].get('pos', (0, 0, 0))
            center_point = Point(pos[0], pos[1])

        self.report({'INFO'}, f"Creating isochrone (threshold={threshold}, weight={weight_attr})...")

        iso_gdf = c2g_utils.create_isochrone(
            _graph_with_2d_positions(G), center=center_point, threshold=threshold, weight=weight_attr,
        )

        if iso_gdf is None or len(iso_gdf) == 0:
            self.report({'ERROR'}, "Isochrone generation failed")
            return {'CANCELLED'}

        geometry = iso_gdf.geometry.unary_union
        iso_obj = _planar_geometry_to_object(
            geometry,
            name="Isochrone",
            collection_name="C2G_Tools",
            matrix_world=obj.matrix_world,
        )

        if iso_obj is not None:
            iso_obj["is_isochrone"] = True
            iso_obj["is_city2graph"] = True
            context.view_layer.objects.active = iso_obj
            self.report({'INFO'}, "Isochrone polygon created")
        else:
            self.report({'ERROR'}, "Failed to create isochrone object")
            return {'CANCELLED'}

        return {'FINISHED'}


classes = [
    SCIGRAPHS_OT_C2G_GraphToolApply,
]


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
