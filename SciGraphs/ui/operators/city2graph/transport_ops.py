import bpy


class SCIGRAPHS_OT_C2G_TravelSummaryGraph(bpy.types.Operator):
    """Create travel summary graph."""
    bl_idname = "scigraphs.c2g_travel_summary_graph"
    bl_label = "Travel Summary Graph"
    bl_description = "Create travel summary graph from GTFS data"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.scene.get("c2g_gtfs_loaded", False)

    def execute(self, context):
        from ....core.city2graph import transportation

        gtfs_data = transportation.get_active_gtfs()
        if gtfs_data is None:
            self.report(
                {'ERROR'},
                "No GTFS connection in memory — re-import the feed.",
            )
            return {'CANCELLED'}
        props = context.scene.city2graph

        cal_start = props.gtfs_calendar_start or None
        cal_end = props.gtfs_calendar_end or None

        self.report({'INFO'}, "Creating travel summary graph...")

        # Pass the active OSMnx graph (when there is one) so the GTFS
        # graph lands on top of the road network — matches the
        # behaviour of "Visualize GTFS Network".
        active = context.active_object
        osmnx_obj = active if (active and active.get("is_osmnx")) else None

        graph_obj = transportation.create_travel_summary_graph(
            gtfs_data,
            calendar_start=cal_start,
            calendar_end=cal_end,
            osmnx_obj=osmnx_obj,
        )

        if graph_obj is None:
            self.report({'ERROR'}, "Failed to create travel summary graph")
            return {'CANCELLED'}

        context.view_layer.objects.active = graph_obj
        self.report({'INFO'}, "Travel summary graph created successfully")
        return {'FINISHED'}


class SCIGRAPHS_OT_C2G_GetODPairs(bpy.types.Operator):
    """Extract Origin-Destination pairs from GTFS data."""
    bl_idname = "scigraphs.c2g_get_od_pairs"
    bl_label = "Extract OD Pairs"
    bl_description = "Extract origin-destination pairs from loaded GTFS data"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.scene.get("c2g_gtfs_loaded", False)

    def execute(self, context):
        from ....core.city2graph import transportation

        gtfs_data = transportation.get_active_gtfs()
        if gtfs_data is None:
            self.report(
                {'ERROR'},
                "No GTFS connection in memory — re-import the feed.",
            )
            return {'CANCELLED'}
        props = context.scene.city2graph

        cal_start = props.gtfs_calendar_start or None
        cal_end = props.gtfs_calendar_end or None
        directed = bool(props.gtfs_od_directed)
        top_n = int(props.gtfs_od_top_n) if props.gtfs_od_top_n > 0 else None

        active = context.active_object
        osmnx_obj = active if (active and active.get("is_osmnx")) else None

        self.report({'INFO'}, "Extracting OD pairs and building graph...")

        graph_obj = transportation.build_gtfs_od_graph(
            gtfs_data,
            osmnx_obj=osmnx_obj,
            start_date=cal_start,
            end_date=cal_end,
            directed=directed,
            top_n=top_n,
        )

        if graph_obj is None:
            self.report({'ERROR'}, "No OD pairs were extracted")
            return {'CANCELLED'}

        context.view_layer.objects.active = graph_obj
        self.report(
            {'INFO'},
            f"OD graph created: {graph_obj.get('num_nodes', 0)} stops, "
            f"{graph_obj.get('num_edges', 0)} pairs",
        )
        return {'FINISHED'}


classes = [
    SCIGRAPHS_OT_C2G_TravelSummaryGraph,
    SCIGRAPHS_OT_C2G_GetODPairs,
]


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)

