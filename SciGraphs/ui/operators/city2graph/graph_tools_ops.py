"""
Graph utility operators for City2Graph.

Provides operators for graph filtering, clipping, isochrone generation,
and removing isolated components.
"""

import bpy


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

        G = c2g_utils.extract_graph_from_blender(obj)
        if G is None:
            self.report({'ERROR'}, "Could not extract graph from object")
            return {'CANCELLED'}

        self.report({'INFO'}, "Removing isolated components...")

        result = c2g_utils.remove_isolated_components(G)
        if result is None:
            self.report({'ERROR'}, "Failed to remove isolated components")
            return {'CANCELLED'}

        from ....core.city2graph.morphology import create_graph_from_networkx
        ref = obj if obj.get("is_osmnx") or obj.get("is_city2graph") else None
        new_obj = create_graph_from_networkx(result, name=f"{obj.name}_Clean", use_positions=True, osmnx_obj=ref)

        if new_obj:
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
        result = c2g_utils.filter_graph_by_distance(G, center=center, threshold=threshold)

        if result is None:
            self.report({'ERROR'}, "Filter failed")
            return {'CANCELLED'}

        from ....core.city2graph.morphology import create_graph_from_networkx
        ref = obj if obj.get("is_osmnx") or obj.get("is_city2graph") else None
        new_obj = create_graph_from_networkx(result, name=f"{obj.name}_Filtered", use_positions=True, osmnx_obj=ref)

        if new_obj:
            new_obj["is_city2graph"] = True
            context.view_layer.objects.active = new_obj
            self.report({'INFO'}, f"Filtered graph: {result.number_of_nodes()} nodes")
        return {'FINISHED'}

    def _clip_graph(self, context, obj, props):
        from ....core.city2graph import utils as c2g_utils

        G = c2g_utils.extract_graph_from_blender(obj)
        if G is None:
            self.report({'ERROR'}, "Could not extract graph")
            return {'CANCELLED'}

        clip_obj = props.graph_filter_center
        if not clip_obj or clip_obj.type != 'MESH':
            self.report({'ERROR'}, "Select a polygon object as clip boundary in 'Center/Clip Object'")
            return {'CANCELLED'}

        clip_gdf = c2g_utils.blender_to_geopandas(clip_obj)
        if clip_gdf is None or len(clip_gdf) == 0:
            self.report({'ERROR'}, "Could not extract clip polygon")
            return {'CANCELLED'}

        polygon = clip_gdf.unary_union

        self.report({'INFO'}, "Clipping graph to polygon...")
        result = c2g_utils.clip_graph(G, polygon=polygon)

        if result is None:
            self.report({'ERROR'}, "Clip failed")
            return {'CANCELLED'}

        from ....core.city2graph.morphology import create_graph_from_networkx
        ref = obj if obj.get("is_osmnx") or obj.get("is_city2graph") else None
        new_obj = create_graph_from_networkx(result, name=f"{obj.name}_Clipped", use_positions=True, osmnx_obj=ref)

        if new_obj:
            new_obj["is_city2graph"] = True
            context.view_layer.objects.active = new_obj
            self.report({'INFO'}, f"Clipped graph: {result.number_of_nodes()} nodes")
        return {'FINISHED'}

    def _isochrone(self, context, obj, props):
        from ....core.city2graph import utils as c2g_utils

        G = c2g_utils.extract_graph_from_blender(obj)
        if G is None:
            self.report({'ERROR'}, "Could not extract graph")
            return {'CANCELLED'}

        threshold = props.isochrone_threshold
        weight_attr = props.isochrone_weight_attr.strip() or "length"

        center_node = 0
        center_obj_prop = props.isochrone_center_object
        if center_obj_prop and center_obj_prop.type == 'MESH':
            loc = center_obj_prop.location
            min_dist = float('inf')
            for n in G.nodes():
                pos = G.nodes[n].get('pos', (0, 0, 0))
                d = (pos[0] - loc.x) ** 2 + (pos[1] - loc.y) ** 2
                if d < min_dist:
                    min_dist = d
                    center_node = n

        self.report({'INFO'}, f"Creating isochrone (threshold={threshold}, weight={weight_attr})...")

        polygon = c2g_utils.create_isochrone(G, center=center_node, threshold=threshold, weight=weight_attr)

        if polygon is None:
            self.report({'ERROR'}, "Isochrone generation failed")
            return {'CANCELLED'}

        import geopandas as gpd
        iso_gdf = gpd.GeoDataFrame(geometry=[polygon], crs="EPSG:4326")

        ref = obj if obj.get("is_osmnx") or obj.get("is_city2graph") else None
        objects = c2g_utils.gdf_to_blender_mesh(
            iso_gdf,
            name="Isochrone",
            collection_name="C2G_Tools",
            osmnx_obj=ref,
        )

        if objects:
            for o in objects:
                o["is_isochrone"] = True
            context.view_layer.objects.active = objects[0]
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
