import bpy
from bpy.props import StringProperty


def _flatten_result_objects(result):
    """Return a flat list of objects from a feature-download result dict."""
    objects = []
    if not result:
        return objects
    for group in result.values():
        objects.extend(group)
    return objects


class SCIGRAPHS_OT_C2G_LoadOverture(bpy.types.Operator):
    """Download urban features as polygons/lines from Overture Maps.

    Replaces the old fragmented panel: a single dispatcher that takes
    the area resolved by :func:`area_resolver.resolve_area` (which
    mirrors the OSMnx area methods, plus a "From OSMnx Graph" mode
    that reuses the active graph's bbox + projection) and pushes the
    result through the Overture REST API.
    """
    bl_idname = "scigraphs.c2g_load_overture"
    bl_label = "Get as Polygons"
    bl_description = (
        "Download buildings / segments / land / water as polygon and line "
        "geometries from Overture Maps. The area follows the Area Method "
        "selected above (and aligns with the active OSMnx graph when one "
        "is selected)"
    )
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        from ....core.city2graph import data, area_resolver

        scene_props = context.scene.scigraphs

        try:
            area = area_resolver.resolve_area(context)
        except ValueError as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}

        bbox = area['bbox']
        osmnx_obj = area['osmnx_obj']

        self.report({'INFO'}, f"Area: {area['source']}")
        self.report({'INFO'}, f"Downloading {scene_props.feat_type} from {scene_props.feat_source}")

        place_name = area.get('place_name') if scene_props.feat_source == 'OSMNX' else None

        result = data.download_features(
            bbox,
            source=scene_props.feat_source,
            feature_type=scene_props.feat_type,
            custom_tags=scene_props.feat_custom_tags,
            osmnx_obj=osmnx_obj,
            limit=scene_props.feat_limit,
            nodes_only=scene_props.feat_nodes_only,
            place_name=place_name,
        )

        if result is None or len(result) == 0:
            self.report({'ERROR'}, "No data downloaded")
            return {'CANCELLED'}

        total_objects = len(_flatten_result_objects(result))
        self.report({'INFO'}, f"Created {total_objects} feature object(s)")

        return {'FINISHED'}


class SCIGRAPHS_OT_C2G_LoadOverturePoints(bpy.types.Operator):
    """Download POIs/places as native points from Overture Maps.

    Complement of :class:`SCIGRAPHS_OT_C2G_LoadOverture`: same area
    resolution and alignment, but the request goes to the *places*
    endpoint (which natively returns points) plus, optionally, any
    polygon feature reduced to its representative point so the result
    is always a point cloud — useful for proximity graphs or
    centrality-on-POIs analyses.
    """
    bl_idname = "scigraphs.c2g_load_overture_points"
    bl_label = "Get as Points"
    bl_description = (
        "Download Overture features as point geometries: 'Places' arrive "
        "natively as points; any other selected feature type is reduced "
        "to its representative point (centroid) so the whole result is "
        "point-only"
    )
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        from ....core.city2graph import data, area_resolver, utils

        scene_props = context.scene.scigraphs

        try:
            area = area_resolver.resolve_area(context)
        except ValueError as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}

        bbox = area['bbox']
        osmnx_obj = area['osmnx_obj']

        self.report({'INFO'}, f"Area: {area['source']}")
        self.report({'INFO'}, f"Downloading {scene_props.feat_type} as points from {scene_props.feat_source}")

        place_name = area.get('place_name') if scene_props.feat_source == 'OSMNX' else None

        full = data.download_features(
            bbox,
            source=scene_props.feat_source,
            feature_type=scene_props.feat_type,
            custom_tags=scene_props.feat_custom_tags,
            osmnx_obj=osmnx_obj,
            limit=scene_props.feat_limit,
            nodes_only=scene_props.feat_nodes_only,
            place_name=place_name,
        )
        objects = _flatten_result_objects(full)
        if not objects:
            self.report({'ERROR'}, "No point features were created")
            return {'CANCELLED'}

        point_objects = []
        for obj in list(objects):
            centroid_obj = utils.mesh_to_centroids(obj)
            if centroid_obj is None:
                continue
            centroid_obj["c2g_point_source"] = scene_props.feat_type
            bpy.data.objects.remove(obj, do_unlink=True)
            point_objects.append(centroid_obj)

        if not point_objects:
            self.report({'ERROR'}, "No point features were created")
            return {'CANCELLED'}

        total = len(point_objects)
        self.report({'INFO'}, f"Created {total} point object(s)")
        return {'FINISHED'}


class SCIGRAPHS_OT_C2G_ConvertToCentroids(bpy.types.Operator):
    """Reduce the active polygon/line object to its representative points.

    Operates on whatever the user has selected (typically a polygon
    object created by *Get as Polygons*) and creates a sibling object
    with one vertex per feature's centroid. Original object is kept,
    so the user can decide to delete it manually.
    """
    bl_idname = "scigraphs.c2g_convert_to_centroids"
    bl_label = "Convert Selected to Centroids"
    bl_description = (
        "Create a new point-only object with one vertex per polygon/line "
        "centroid of the active mesh. The original object is preserved"
    )
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj is not None and obj.type == 'MESH'

    def execute(self, context):
        from ....core.city2graph import utils

        obj = context.active_object
        centroid_obj = utils.mesh_to_centroids(obj)
        if centroid_obj is None:
            self.report({'ERROR'}, "Could not extract centroids from this object")
            return {'CANCELLED'}

        centroid_obj["c2g_point_source"] = obj.name
        # Activate the new object so the user sees the result immediately.
        bpy.ops.object.select_all(action='DESELECT')
        centroid_obj.select_set(True)
        context.view_layer.objects.active = centroid_obj

        self.report(
            {'INFO'},
            f"Created '{centroid_obj.name}' with "
            f"{len(centroid_obj.data.vertices)} centroid(s) from '{obj.name}'",
        )
        return {'FINISHED'}


class SCIGRAPHS_OT_C2G_ImportGTFS(bpy.types.Operator):
    """Import GTFS transit data."""
    bl_idname = "scigraphs.c2g_import_gtfs"
    bl_label = "Import GTFS Data"
    bl_description = "Import GTFS transit feed from zip file"
    bl_options = {'REGISTER', 'UNDO'}
    
    filepath: StringProperty(
        subtype='FILE_PATH',
    )
    
    def execute(self, context):
        from ....core.city2graph import transportation
        
        if not self.filepath:
            props = context.scene.city2graph
            self.filepath = bpy.path.abspath(props.c2g_gtfs_path)
        
        if not self.filepath:
            self.report({'ERROR'}, "No GTFS file specified")
            return {'CANCELLED'}
        
        self.report({'INFO'}, f"Loading GTFS data from {self.filepath}")
        
        gtfs_data = transportation.load_gtfs(self.filepath)

        if gtfs_data is None:
            self.report({'ERROR'}, "Failed to load GTFS data")
            return {'CANCELLED'}

        # Blender id-properties cannot store a DuckDBPyConnection, so
        # the connection itself lives in a module-level cache while the
        # scene only stores plain metadata that the panels poll.
        transportation.set_active_gtfs(gtfs_data, filepath=self.filepath)

        try:
            tables = sorted(
                row[0] for row in gtfs_data.execute("SHOW TABLES").fetchall()
            )
        except Exception:  # noqa: BLE001
            tables = []

        context.scene["c2g_gtfs_loaded"] = True
        context.scene["c2g_gtfs_path"] = self.filepath
        context.scene["c2g_gtfs_tables"] = tables
        # Legacy key kept for backwards compatibility with any panel
        # that still polls "c2g_gtfs_data". Safe truthy marker.
        context.scene["c2g_gtfs_data"] = self.filepath

        # Discover the set of service dates present in the feed so the
        # Calendar Start/End dropdowns can offer real choices instead
        # of asking the user to type YYYYMMDD by hand.
        try:
            dates = []
            if 'calendar' in tables:
                rows = gtfs_data.execute("""
                    SELECT MIN(start_date) AS s, MAX(end_date) AS e
                    FROM calendar
                """).fetchone() or (None, None)
                s, e = rows
                if s and e:
                    from datetime import datetime, timedelta
                    s_dt = datetime.strptime(str(s), "%Y%m%d")
                    e_dt = datetime.strptime(str(e), "%Y%m%d")
                    # Cap at 366 days so the dropdown stays usable on
                    # multi-year feeds.
                    span = min((e_dt - s_dt).days, 366)
                    dates = [
                        (s_dt + timedelta(days=i)).strftime("%Y%m%d")
                        for i in range(span + 1)
                    ]
            if not dates and 'calendar_dates' in tables:
                rows = gtfs_data.execute(
                    "SELECT DISTINCT date FROM calendar_dates ORDER BY date"
                ).fetchall()
                dates = [str(r[0]) for r in rows][:366]
            context.scene["c2g_gtfs_dates"] = dates
        except Exception as _e:  # noqa: BLE001
            context.scene["c2g_gtfs_dates"] = []

        self.report({'INFO'}, "GTFS data loaded successfully")
        return {'FINISHED'}
    
    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}


class SCIGRAPHS_OT_C2G_VisualizeGTFS(bpy.types.Operator):
    """Visualize GTFS network."""
    bl_idname = "scigraphs.c2g_visualize_gtfs"
    bl_label = "Visualize GTFS Network"
    bl_description = "Create 3D visualization of GTFS transit network"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        from ....core.city2graph import transportation
        
        gtfs_data = transportation.get_active_gtfs()
        if gtfs_data is None:
            self.report(
                {'ERROR'},
                "No GTFS data loaded (the cached connection was lost — re-import).",
            )
            return {'CANCELLED'}
        props = context.scene.city2graph
        
        osmnx_obj = context.active_object if context.active_object and context.active_object.get("is_osmnx") else None
        
        self.report({'INFO'}, "Creating GTFS visualization...")
        
        result = transportation.visualize_gtfs_network(
            gtfs_data,
            osmnx_obj=osmnx_obj,
            create_stops=props.c2g_gtfs_create_stops,
            create_routes=props.c2g_gtfs_create_routes
        )
        
        if result is None:
            self.report({'ERROR'}, "Failed to create GTFS visualization")
            return {'CANCELLED'}
        
        total = len(result.get('stops', [])) + len(result.get('routes', []))
        self.report({'INFO'}, f"Created {total} GTFS objects")
        
        return {'FINISHED'}


class SCIGRAPHS_OT_C2G_LoadFromFile(bpy.types.Operator):
    """Load urban data from file."""
    bl_idname = "scigraphs.c2g_load_file"
    bl_label = "Load from File"
    bl_description = "Load urban data from GeoJSON or Shapefile"
    bl_options = {'REGISTER', 'UNDO'}
    
    filepath: StringProperty(
        subtype='FILE_PATH',
    )
    
    filter_glob: StringProperty(
        default="*.geojson;*.json;*.shp;*.gpkg",
        options={'HIDDEN'},
    )
    
    def execute(self, context):
        from ....core.city2graph import data
        
        if not self.filepath:
            self.report({'ERROR'}, "No file specified")
            return {'CANCELLED'}
        
        osmnx_obj = context.active_object if context.active_object and context.active_object.get("is_osmnx") else None
        
        self.report({'INFO'}, f"Loading data from {self.filepath}")
        
        objects = data.load_data_from_file(self.filepath, osmnx_obj=osmnx_obj)
        
        if not objects:
            self.report({'ERROR'}, "Failed to load data from file")
            return {'CANCELLED'}
        
        self.report({'INFO'}, f"Created {len(objects)} object(s) from file")
        
        return {'FINISHED'}
    
    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}


class SCIGRAPHS_OT_C2G_GeocodeBoundaries(bpy.types.Operator):
    """Geocode a place name to obtain its polygon boundary."""
    bl_idname = "scigraphs.c2g_geocode_boundaries"
    bl_label = "Geocode Boundaries"
    bl_description = "Resolve a place name (e.g. 'Liverpool, UK') to a polygon boundary"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        from ....core.city2graph import data, utils

        props = context.scene.city2graph
        place = props.geocode_place_name.strip()

        if not place:
            self.report({'ERROR'}, "Enter a place name to geocode")
            return {'CANCELLED'}

        self.report({'INFO'}, f"Geocoding '{place}'...")

        boundary_gdf = data.get_boundaries(place)
        if boundary_gdf is None:
            self.report({'ERROR'}, f"No boundary found for '{place}'")
            return {'CANCELLED'}

        osmnx_obj = context.active_object if (
            context.active_object and context.active_object.get("is_osmnx")
        ) else None

        objects = utils.gdf_to_blender_mesh(
            boundary_gdf,
            name=f"Boundary_{place.split(',')[0].strip()}",
            collection_name="C2G_Boundaries",
            osmnx_obj=osmnx_obj,
        )

        if not objects:
            self.report({'ERROR'}, "Failed to create boundary object")
            return {'CANCELLED'}

        for obj in objects:
            obj["is_boundary"] = True
            obj["place_name"] = place

        self.report({'INFO'}, f"Boundary for '{place}' created")
        return {'FINISHED'}


class SCIGRAPHS_OT_C2G_LoadOverturePlaceName(bpy.types.Operator):
    """Download Overture data using city2graph API with a place name."""
    bl_idname = "scigraphs.c2g_load_overture_place"
    bl_label = "Download via Place Name"
    bl_description = "Download Overture Maps data using city2graph CLI with place name geocoding"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        from ....core.city2graph import data

        props = context.scene.city2graph
        place = props.geocode_place_name.strip()

        if not place:
            self.report({'ERROR'}, "Enter a place name")
            return {'CANCELLED'}

        types = []
        if props.c2g_overture_building:
            types.append('building')
        if props.c2g_overture_segment:
            types.append('segment')
        if props.c2g_overture_connector:
            types.append('connector')
        if props.c2g_overture_place:
            types.append('place')
        if not types:
            self.report({'ERROR'}, "No feature types selected")
            return {'CANCELLED'}

        osmnx_obj = context.active_object if (
            context.active_object and context.active_object.get("is_osmnx")
        ) else None

        self.report({'INFO'}, f"Downloading Overture data for '{place}': {', '.join(types)}")

        result = data.load_overture_data(
            types=types,
            osmnx_obj=osmnx_obj,
            use_city2graph_api=True,
            place_name=place,
        )

        if result is None or len(result) == 0:
            self.report({'ERROR'}, "No data downloaded")
            return {'CANCELLED'}

        total = sum(len(v) for v in result.values())
        self.report({'INFO'}, f"Created {total} object(s) from Overture Maps")
        return {'FINISHED'}


class SCIGRAPHS_OT_C2G_ProcessSegments(bpy.types.Operator):
    """Process Overture segments: split at connectors, extract barriers."""
    bl_idname = "scigraphs.c2g_process_segments"
    bl_label = "Process Overture Segments"
    bl_description = "Split segments by connectors and optionally extract barriers"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.active_object is not None and context.active_object.type == 'MESH'

    def execute(self, context):
        from ....core.city2graph import data, utils

        obj = context.active_object

        segments_gdf = utils.blender_to_geopandas(obj)
        if segments_gdf is None or len(segments_gdf) == 0:
            self.report({'ERROR'}, "Could not extract geometries from active object")
            return {'CANCELLED'}

        connectors_gdf = None
        for sel in context.selected_objects:
            if sel != obj and sel.type == 'MESH':
                connectors_gdf = utils.blender_to_geopandas(sel)
                break

        self.report({'INFO'}, "Processing Overture segments...")

        processed = data.process_overture_segments(
            segments_gdf,
            connectors_gdf=connectors_gdf,
            get_barriers=True,
        )

        if processed is None:
            self.report({'ERROR'}, "Segment processing failed")
            return {'CANCELLED'}

        ref_obj = obj if obj.get("is_osmnx") or obj.get("is_city2graph") else None
        objects = utils.gdf_to_blender_mesh(
            processed,
            name="Processed_Segments",
            collection_name="C2G_Segments",
            osmnx_obj=ref_obj,
        )

        if objects:
            for o in objects:
                o["is_processed_segments"] = True
            self.report({'INFO'}, f"Created {len(objects)} processed segment object(s)")
        else:
            self.report({'ERROR'}, "Failed to create segment objects")
            return {'CANCELLED'}

        return {'FINISHED'}


class SCIGRAPHS_OT_C2G_ExportGraph(bpy.types.Operator):
    """Export graph to JSON/GraphML."""
    bl_idname = "scigraphs.c2g_export_graph"
    bl_label = "Export Graph"
    bl_description = "Export graph to JSON or GraphML for external GNN training"
    bl_options = {'REGISTER'}
    
    filepath: StringProperty(
        subtype='FILE_PATH',
    )
    
    def execute(self, context):
        from ....core.city2graph import graph
        
        obj = context.active_object
        if obj is None or obj.type != 'MESH':
            self.report({'ERROR'}, "Select a graph object to export")
            return {'CANCELLED'}
        
        props = context.scene.city2graph
        
        if not self.filepath:
            self.filepath = bpy.path.abspath(props.c2g_export_path)
        
        if not self.filepath:
            self.report({'ERROR'}, "No export path specified")
            return {'CANCELLED'}
        
        format = props.c2g_export_format
        
        if not self.filepath.endswith(('.json', '.graphml')):
            if format == 'JSON':
                self.filepath += '.json'
            else:
                self.filepath += '.graphml'
        
        self.report({'INFO'}, f"Exporting graph to {format}...")
        
        success = graph.export_graph(obj, self.filepath, format=format)
        
        if not success:
            self.report({'ERROR'}, "Failed to export graph")
            return {'CANCELLED'}
        
        self.report({'INFO'}, f"Graph exported to {self.filepath}")
        
        return {'FINISHED'}
    
    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}


class SCIGRAPHS_OT_C2G_SaveLoaderScript(bpy.types.Operator):
    """Save PyG loader script."""
    bl_idname = "scigraphs.c2g_save_loader_script"
    bl_label = "Save PyG Loader Script"
    bl_description = "Save Python script for loading exports in PyTorch Geometric"
    bl_options = {'REGISTER'}
    
    filepath: StringProperty(
        subtype='FILE_PATH',
        default="load_scigraphs_export.py",
    )
    
    def execute(self, context):
        from ....core.city2graph import graph
        
        if not self.filepath:
            self.report({'ERROR'}, "No file path specified")
            return {'CANCELLED'}
        
        success = graph.save_external_loader_script(self.filepath)
        
        if not success:
            self.report({'ERROR'}, "Failed to save loader script")
            return {'CANCELLED'}
        
        self.report({'INFO'}, f"PyG loader script saved to {self.filepath}")
        
        return {'FINISHED'}
    
    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}


classes = [
    SCIGRAPHS_OT_C2G_LoadOverture,
    SCIGRAPHS_OT_C2G_LoadOverturePoints,
    SCIGRAPHS_OT_C2G_ConvertToCentroids,
    SCIGRAPHS_OT_C2G_ImportGTFS,
    SCIGRAPHS_OT_C2G_VisualizeGTFS,
    SCIGRAPHS_OT_C2G_LoadFromFile,
    SCIGRAPHS_OT_C2G_GeocodeBoundaries,
    SCIGRAPHS_OT_C2G_LoadOverturePlaceName,
    SCIGRAPHS_OT_C2G_ProcessSegments,
    SCIGRAPHS_OT_C2G_ExportGraph,
    SCIGRAPHS_OT_C2G_SaveLoaderScript,
]


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)

