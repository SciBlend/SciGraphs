import bpy


class SCIGRAPHS_OT_C2G_GenerateTessellation(bpy.types.Operator):
    """Generate urban tessellation."""
    bl_idname = "scigraphs.c2g_generate_tessellation"
    bl_label = "Generate Tessellation"
    bl_description = "Create Voronoi tessellation from building footprints"
    bl_options = {'REGISTER', 'UNDO'}
    
    @classmethod
    def poll(cls, context):
        return context.active_object is not None and context.active_object.type == 'MESH'
    
    def execute(self, context):
        from ....core.city2graph import morphology
        
        buildings_obj = context.active_object
        
        barriers_obj = None
        if len(context.selected_objects) > 1:
            for obj in context.selected_objects:
                if obj != buildings_obj and obj.type == 'MESH':
                    barriers_obj = obj
                    break
        
        props = context.scene.city2graph
        
        self.report({'INFO'}, "Generating tessellation...")
        
        tessellation_obj = morphology.create_tessellation(
            buildings_obj,
            barriers_obj=barriers_obj,
            shrink=props.c2g_tessellation_shrink,
            segment_length=props.c2g_tessellation_segment
        )
        
        if tessellation_obj is None:
            self.report({'ERROR'}, "Failed to generate tessellation")
            return {'CANCELLED'}
        
        context.view_layer.objects.active = tessellation_obj
        
        self.report({'INFO'}, "Tessellation generated successfully")
        return {'FINISHED'}


class SCIGRAPHS_OT_C2G_MorphologicalGraph(bpy.types.Operator):
    """Create morphological graph."""
    bl_idname = "scigraphs.c2g_morphological_graph"
    bl_label = "Morphological Graph"
    bl_description = "Generate graph connecting buildings to streets (private-to-public)"
    bl_options = {'REGISTER', 'UNDO'}
    
    @classmethod
    def poll(cls, context):
        return len(context.selected_objects) >= 2
    
    def execute(self, context):
        from ....core.city2graph import morphology

        selected = context.selected_objects
        props = context.scene.city2graph

        if len(selected) < 2:
            self.report(
                {'ERROR'},
                "Select two objects: buildings (polygons) + street segments",
            )
            return {'CANCELLED'}

        # Heuristics to identify the role of each selected object:
        #  - Buildings = a mesh with face polygons (Polygon footprints).
        #  - Streets   = a mesh with edges but no faces (LineStrings).
        #  - The legacy 'is_tessellation' flag is also accepted as a
        #    "buildings stand-in" because the previous workflow asked
        #    the user to feed tessellations directly.
        buildings_obj = None
        street_network_obj = None
        for obj in selected:
            if obj.type != 'MESH' or obj.data is None:
                continue
            if (
                obj.get("is_tessellation")
                or obj.get("c2g_point_source") == 'building'
                or 'building' in obj.name.lower()
                or len(obj.data.polygons) > 0
            ):
                if buildings_obj is None:
                    buildings_obj = obj
                continue
            if (
                obj.get("is_osmnx")
                or obj.get("is_city2graph")
                or obj.get("c2g_point_source") == 'segment'
                or 'segment' in obj.name.lower()
                or 'street' in obj.name.lower()
                or (len(obj.data.edges) > 0 and len(obj.data.polygons) == 0)
            ):
                if street_network_obj is None:
                    street_network_obj = obj

        if buildings_obj is None or street_network_obj is None:
            # Fall back to positional order: first selected = buildings,
            # second = streets, just so an unguessable selection still
            # produces a meaningful error from the core function.
            buildings_obj = buildings_obj or selected[0]
            street_network_obj = street_network_obj or selected[1]

        # Resolve the centre for distance-based filtering.
        center_lat = None
        center_lon = None
        if props.morpho_use_center_from_osmnx:
            for cand in (street_network_obj, buildings_obj):
                if cand is None:
                    continue
                clat = cand.get("osmnx_center_lat") or cand.get("c2g_center_lat")
                clon = cand.get("osmnx_center_lon") or cand.get("c2g_center_lon")
                if clat is not None and clon is not None:
                    center_lat, center_lon = float(clat), float(clon)
                    break
        else:
            center_lat = float(props.morpho_center_lat)
            center_lon = float(props.morpho_center_lon)

        self.report(
            {'INFO'},
            f"Creating morphological graph: buildings='{buildings_obj.name}', "
            f"streets='{street_network_obj.name}'...",
        )

        if not (
            props.morpho_rel_priv_priv
            or props.morpho_rel_pub_pub
            or props.morpho_rel_priv_pub
        ):
            self.report(
                {'ERROR'},
                "Enable at least one relation type (priv↔priv, pub↔pub, priv↔pub)",
            )
            return {'CANCELLED'}

        result = morphology.create_morphological_graph(
            buildings_obj,
            street_network_obj,
            center_lat=center_lat,
            center_lon=center_lon,
            distance=float(props.morpho_distance) if props.morpho_distance > 0 else None,
            clipping_buffer=float(props.morpho_clipping_buffer),
            contiguity=props.morpho_contiguity,
            keep_buildings=bool(props.morpho_keep_buildings),
            keep_segments=bool(props.morpho_keep_segments),
            include_priv_priv=bool(props.morpho_rel_priv_priv),
            include_pub_pub=bool(props.morpho_rel_pub_pub),
            include_priv_pub=bool(props.morpho_rel_priv_pub),
        )

        if not result:
            self.report({'ERROR'}, "Failed to create morphological graph")
            return {'CANCELLED'}

        # ``result`` is a single object in full mode and a list of
        # objects in subset mode. Normalise for reporting / activation.
        if isinstance(result, list):
            for o in result:
                o.select_set(True)
            context.view_layer.objects.active = result[-1]
            names = ", ".join(o.name for o in result)
            self.report(
                {'INFO'},
                f"Morphological graphs created: {names}",
            )
        else:
            context.view_layer.objects.active = result
            self.report({'INFO'}, "Morphological graph created successfully")

        return {'FINISHED'}


class SCIGRAPHS_OT_C2G_SegmentsToGraph(bpy.types.Operator):
    """Convert Overture segments directly into a NetworkX graph."""
    bl_idname = "scigraphs.c2g_segments_to_graph"
    bl_label = "Segments to Graph"
    bl_description = "Convert processed Overture road segments to a graph"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.active_object is not None and context.active_object.type == 'MESH'

    def execute(self, context):
        from ....core.city2graph.get_c2g import get_city2graph
        from ....core.city2graph import utils as c2g_utils
        from ....core.city2graph.morphology import create_graph_from_networkx

        c2g = get_city2graph()
        if c2g is None:
            self.report({'ERROR'}, "city2graph library is not available")
            return {'CANCELLED'}

        obj = context.active_object
        segments_gdf = c2g_utils.blender_to_geopandas(obj)
        if segments_gdf is None or len(segments_gdf) == 0:
            self.report({'ERROR'}, "Could not extract segment geometries")
            return {'CANCELLED'}

        self.report({'INFO'}, "Converting segments to graph...")

        try:
            from city2graph.morphology import segments_to_graph as c2g_seg2graph
            # city2graph >= 0.3 returns a (nodes_gdf, edges_gdf) tuple
            # by default; ``as_nx=True`` collapses that into the
            # NetworkX graph the rest of this operator expects.
            graph = c2g_seg2graph(segments_gdf, as_nx=True)
        except Exception as e:
            self.report({'ERROR'}, f"segments_to_graph failed: {e}")
            return {'CANCELLED'}

        # Be tolerant of older c2g versions that still returned the
        # tuple even when as_nx=True wasn't supported.
        if isinstance(graph, tuple):
            try:
                import networkx as nx
                nodes_gdf, edges_gdf = graph
                G = nx.Graph()
                for nid, row in nodes_gdf.iterrows():
                    pt = row.geometry
                    G.add_node(nid, x=float(pt.x), y=float(pt.y))
                for (u, v), _ in edges_gdf.iterrows():
                    G.add_edge(u, v)
                graph = G
            except Exception as e:  # noqa: BLE001
                self.report({'ERROR'}, f"Unexpected segments_to_graph return: {e}")
                return {'CANCELLED'}

        if graph is None or graph.number_of_nodes() == 0:
            self.report({'ERROR'}, "Resulting graph is empty")
            return {'CANCELLED'}

        ref_obj = obj if obj.get("is_osmnx") or obj.get("is_city2graph") else None
        graph_obj = create_graph_from_networkx(graph, name="Segments_Graph", use_positions=True, osmnx_obj=ref_obj)

        if graph_obj:
            graph_obj["is_segments_graph"] = True
            context.view_layer.objects.active = graph_obj
            self.report({'INFO'}, f"Graph created: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
        else:
            self.report({'ERROR'}, "Failed to create graph object")
            return {'CANCELLED'}

        return {'FINISHED'}


classes = [
    SCIGRAPHS_OT_C2G_GenerateTessellation,
    SCIGRAPHS_OT_C2G_MorphologicalGraph,
    SCIGRAPHS_OT_C2G_SegmentsToGraph,
]


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)

