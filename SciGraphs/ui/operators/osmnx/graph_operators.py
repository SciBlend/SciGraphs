import bpy
from bpy.props import StringProperty, FloatProperty, BoolProperty, EnumProperty
from ....core import osmnx_analysis
from .utils import (
    _get_osmnx_graph,
    _get_unprojected_graph,
    _store_osmnx_graph,
    _store_unprojected_graph,
    _transfer_edge_attribute_to_mesh,
)


def _iter_edge_data(G):
    """Iterate edge data dicts tolerantly across (Multi)DiGraph / (Multi)Graph."""
    if getattr(G, "is_multigraph", lambda: False)():
        for u, v, _k, data in G.edges(keys=True, data=True):
            yield u, v, data
    else:
        for u, v, data in G.edges(data=True):
            yield u, v, data


OSMNX_CRS_PRESETS = (
    ('AUTO_UTM', "Auto UTM (recommended)", "Let OSMnx choose the local UTM CRS for meter-based analysis"),
    ('WEB_MERCATOR', "EPSG:3857 Web Mercator", "Global web map projection; convenient for tiles, not equal-area"),
    ('WORLD_MERCATOR', "EPSG:3395 World Mercator", "Global conformal projection in meters"),
    ('EQUAL_EARTH', "EPSG:8857 Equal Earth", "Global equal-area projection"),
    ('EUROPE_LAEA', "EPSG:3035 Europe LAEA", "European equal-area projection"),
    ('EUROPE_MERCATOR', "EPSG:3034 Europe LCC", "European Lambert conformal conic projection"),
    ('SPAIN_UTM28', "EPSG:25828 Spain ETRS89 / UTM 28N", "Canary Islands and western Spain context"),
    ('SPAIN_UTM29', "EPSG:25829 Spain ETRS89 / UTM 29N", "Western Iberia"),
    ('SPAIN_UTM30', "EPSG:25830 Spain ETRS89 / UTM 30N", "Most of mainland Spain, including Valencia"),
    ('SPAIN_UTM31', "EPSG:25831 Spain ETRS89 / UTM 31N", "Eastern Spain and Balearic context"),
    ('WGS84_UTM28N', "EPSG:32628 WGS84 / UTM 28N", "UTM zone 28 north"),
    ('WGS84_UTM29N', "EPSG:32629 WGS84 / UTM 29N", "UTM zone 29 north"),
    ('WGS84_UTM30N', "EPSG:32630 WGS84 / UTM 30N", "UTM zone 30 north"),
    ('WGS84_UTM31N', "EPSG:32631 WGS84 / UTM 31N", "UTM zone 31 north"),
    ('UK_BNG', "EPSG:27700 British National Grid", "Great Britain national grid"),
    ('IRELAND_TM75', "EPSG:2157 Irish Transverse Mercator", "Ireland national projected CRS"),
    ('FRANCE_LAMBERT93', "EPSG:2154 France Lambert-93", "France mainland projected CRS"),
    ('NETHERLANDS_RD', "EPSG:28992 Netherlands RD New", "Netherlands national projected CRS"),
    ('SWISS_LV95', "EPSG:2056 Switzerland LV95", "Swiss projected CRS"),
    ('GERMANY_UTM32', "EPSG:25832 Germany ETRS89 / UTM 32N", "Western and central Germany"),
    ('GERMANY_UTM33', "EPSG:25833 Germany ETRS89 / UTM 33N", "Eastern Germany"),
    ('ITALY_UTM32', "EPSG:32632 Italy WGS84 / UTM 32N", "Western Italy"),
    ('ITALY_UTM33', "EPSG:32633 Italy WGS84 / UTM 33N", "Eastern Italy"),
    ('PORTUGAL_TM06', "EPSG:3763 Portugal TM06", "Portugal mainland projected CRS"),
    ('US_ALBERS', "EPSG:5070 USA Albers", "Contiguous US equal-area projection"),
    ('US_CONTIGUOUS_ALBERS', "ESRI:102003 USA Contiguous Albers", "ESRI US contiguous Albers equal-area"),
    ('CANADA_LAMBERT', "EPSG:3347 Canada Lambert", "Canada Statistics Lambert projected CRS"),
    ('AUSTRALIA_ALBERS', "EPSG:3577 Australia Albers", "Australia national equal-area projection"),
    ('WGS84_GEOGRAPHIC', "EPSG:4326 WGS84 geographic", "Latitude/longitude CRS; not suitable for meter-based buffers"),
    ('CUSTOM', "Custom EPSG/PROJ string", "Use the custom CRS field below"),
)

OSMNX_CRS_PRESET_VALUES = {
    'AUTO_UTM': "",
    'WEB_MERCATOR': "EPSG:3857",
    'WORLD_MERCATOR': "EPSG:3395",
    'EQUAL_EARTH': "EPSG:8857",
    'EUROPE_LAEA': "EPSG:3035",
    'EUROPE_MERCATOR': "EPSG:3034",
    'SPAIN_UTM28': "EPSG:25828",
    'SPAIN_UTM29': "EPSG:25829",
    'SPAIN_UTM30': "EPSG:25830",
    'SPAIN_UTM31': "EPSG:25831",
    'WGS84_UTM28N': "EPSG:32628",
    'WGS84_UTM29N': "EPSG:32629",
    'WGS84_UTM30N': "EPSG:32630",
    'WGS84_UTM31N': "EPSG:32631",
    'UK_BNG': "EPSG:27700",
    'IRELAND_TM75': "EPSG:2157",
    'FRANCE_LAMBERT93': "EPSG:2154",
    'NETHERLANDS_RD': "EPSG:28992",
    'SWISS_LV95': "EPSG:2056",
    'GERMANY_UTM32': "EPSG:25832",
    'GERMANY_UTM33': "EPSG:25833",
    'ITALY_UTM32': "EPSG:32632",
    'ITALY_UTM33': "EPSG:32633",
    'PORTUGAL_TM06': "EPSG:3763",
    'US_ALBERS': "EPSG:5070",
    'US_CONTIGUOUS_ALBERS': "ESRI:102003",
    'CANADA_LAMBERT': "EPSG:3347",
    'AUSTRALIA_ALBERS': "EPSG:3577",
    'WGS84_GEOGRAPHIC': "EPSG:4326",
}


class SCIGRAPHS_OT_ProjectGraph(bpy.types.Operator):
    """Project OSMnx graph to specified CRS or UTM."""
    bl_idname = "scigraphs.osmnx_project_graph"
    bl_label = "Project to CRS"
    bl_description = "Reproject graph to specified CRS or local UTM coordinate system"
    bl_options = {'REGISTER', 'UNDO'}
    
    target_crs: StringProperty(
        name="Custom CRS",
        description="Custom target CRS (e.g., 'EPSG:27700', '27700', 'ESRI:102003', or a PROJ string)",
        default=""
    )

    crs_preset: EnumProperty(
        name="Target CRS",
        description="Projection preset to use",
        items=OSMNX_CRS_PRESETS,
        default='AUTO_UTM',
    )
    
    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj and obj.get("is_osmnx", False)
    
    def execute(self, context):
        obj = context.active_object
        G = _get_osmnx_graph(obj)
        
        if G is None:
            self.report({'ERROR'}, "No OSMnx graph in memory. Re-import the network or load from GraphML.")
            return {'CANCELLED'}
        
        if not osmnx_analysis.is_graph_projected(G):
            _store_unprojected_graph(obj, G.copy())
        
        custom_crs = self.target_crs.strip()
        if self.crs_preset == 'CUSTOM' or (custom_crs and self.crs_preset == 'AUTO_UTM'):
            crs_param = custom_crs or None
        else:
            crs_value = OSMNX_CRS_PRESET_VALUES.get(self.crs_preset, "")
            crs_param = crs_value or None

        G_proj, crs_info = osmnx_analysis.project_graph(G, to_crs=crs_param)
        
        if G_proj is None:
            self.report({'ERROR'}, f"Projection failed: {crs_info}")
            return {'CANCELLED'}
        
        _store_osmnx_graph(obj, G_proj)
        obj["osmnx_crs"] = crs_info
        obj["osmnx_projected"] = osmnx_analysis.is_graph_projected(G_proj)
        
        self.report({'INFO'}, f"Graph projected to {crs_info}")
        return {'FINISHED'}
    
    def invoke(self, context, _event):
        return context.window_manager.invoke_props_dialog(self, width=520)

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False

        layout.prop(self, "crs_preset", text="Target CRS")
        if self.crs_preset == 'CUSTOM':
            layout.prop(self, "target_crs", text="Custom CRS")

        if self.crs_preset == 'WGS84_GEOGRAPHIC':
            box = layout.box()
            box.scale_y = 0.8
            box.label(text="WGS84 is not projected.", icon='ERROR')
            box.label(text="Use Auto UTM or a local projected CRS for meters.")


class SCIGRAPHS_OT_AddEdgeLengths(bpy.types.Operator):
    """Add length attribute to all edges."""
    bl_idname = "scigraphs.osmnx_add_edge_lengths"
    bl_label = "Add Edge Lengths"
    bl_description = "Calculate and add length attribute to all edges"
    bl_options = {'REGISTER', 'UNDO'}
    
    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj and obj.get("is_osmnx", False)
    
    def execute(self, context):
        obj = context.active_object
        G = _get_osmnx_graph(obj)
        
        if G is None:
            self.report({'ERROR'}, "No OSMnx graph found")
            return {'CANCELLED'}
        
        G = osmnx_analysis.add_edge_lengths(G)
        
        if G is None:
            self.report({'ERROR'}, "Failed to add edge lengths")
            return {'CANCELLED'}
        
        _store_osmnx_graph(obj, G)
        obj["osmnx_has_lengths"] = True
        
        edges_transferred = _transfer_edge_attribute_to_mesh(obj, G, 'length', 'edge_length')
        
        self.report({'INFO'}, f"Edge lengths added: {edges_transferred} mesh edges")
        return {'FINISHED'}


class SCIGRAPHS_OT_AddEdgeBearings(bpy.types.Operator):
    """Add bearing attribute (compass direction) to all edges."""
    bl_idname = "scigraphs.osmnx_add_edge_bearings"
    bl_label = "Add Edge Bearings"
    bl_description = "Calculate street orientation (0-360 degrees) for all edges"
    bl_options = {'REGISTER', 'UNDO'}
    
    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj and obj.get("is_osmnx", False)
    
    def execute(self, context):
        obj = context.active_object
        G = _get_osmnx_graph(obj)
        
        if G is None:
            self.report({'ERROR'}, "No OSMnx graph found")
            return {'CANCELLED'}
        
        unprojected_G = _get_unprojected_graph(obj)
        
        G = osmnx_analysis.add_edge_bearings(G, unprojected_G=unprojected_G)
        
        if G is None:
            self.report({'ERROR'}, "Failed to add edge bearings")
            return {'CANCELLED'}
        
        _store_osmnx_graph(obj, G)
        obj["osmnx_has_bearings"] = True
        
        edges_transferred = _transfer_edge_attribute_to_mesh(obj, G, 'bearing', 'edge_bearing')
        
        dist = osmnx_analysis.get_bearing_distribution(G)
        if dist:
            obj["osmnx_bearing_entropy"] = dist['normalized_entropy']
            dominant = dist.get('dominant_bearings', [])
            if dominant:
                obj["osmnx_dominant_bearings"] = str(dominant[:4])
        
        self.report({'INFO'}, f"Edge bearings added: {edges_transferred} mesh edges")
        return {'FINISHED'}


class SCIGRAPHS_OT_AddEdgeSpeeds(bpy.types.Operator):
    """Add speed estimates to all edges based on road type."""
    bl_idname = "scigraphs.osmnx_add_edge_speeds"
    bl_label = "Add Edge Speeds"
    bl_description = "Estimate travel speeds based on road type and OSM data"
    bl_options = {'REGISTER', 'UNDO'}
    
    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj and obj.get("is_osmnx", False)
    
    def execute(self, context):
        props = context.scene.scigraphs
        obj = context.active_object
        G = _get_osmnx_graph(obj)
        
        if G is None:
            self.report({'ERROR'}, "No OSMnx graph found")
            return {'CANCELLED'}
        
        fallback = props.osmnx_fallback_speed
        G = osmnx_analysis.add_edge_speeds(G, fallback_speed=fallback)
        
        if G is None:
            self.report({'ERROR'}, "Failed to add edge speeds")
            return {'CANCELLED'}

        verified = 0
        total_edges = 0
        for _u, _v, data in _iter_edge_data(G):
            total_edges += 1
            if "speed_kph" in data and data["speed_kph"] is not None:
                verified += 1

        if verified == 0:
            self.report({'ERROR'},
                        "OSMnx returned without errors but no edge received a "
                        "'speed_kph' value. The graph may lack 'highway' tags "
                        "and the fallback speed could not be applied.")
            return {'CANCELLED'}

        _store_osmnx_graph(obj, G)
        obj["osmnx_has_speeds"] = True
        
        edges_transferred = _transfer_edge_attribute_to_mesh(obj, G, 'speed_kph', 'edge_speed')
        
        self.report({'INFO'},
                    f"Edge speeds added: {verified}/{total_edges} graph edges, "
                    f"{edges_transferred} mesh edges updated (fallback: {fallback} km/h)")
        return {'FINISHED'}


class SCIGRAPHS_OT_AddEdgeTravelTimes(bpy.types.Operator):
    """Add travel time to all edges based on length and speed."""
    bl_idname = "scigraphs.osmnx_add_travel_times"
    bl_label = "Add Travel Times"
    bl_description = "Calculate travel time for each edge (requires speeds)"
    bl_options = {'REGISTER', 'UNDO'}
    
    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj and obj.get("is_osmnx", False) and obj.get("osmnx_has_speeds", False)
    
    def execute(self, context):
        obj = context.active_object
        G = _get_osmnx_graph(obj)
        
        if G is None:
            self.report({'ERROR'}, "No OSMnx graph found")
            return {'CANCELLED'}
        
        missing_length = 0
        missing_speed = 0
        total_edges = 0
        for _u, _v, data in _iter_edge_data(G):
            total_edges += 1
            if "length" not in data or data.get("length") is None:
                missing_length += 1
            if "speed_kph" not in data or data.get("speed_kph") is None:
                missing_speed += 1

        if total_edges == 0:
            self.report({'ERROR'}, "Graph has no edges")
            return {'CANCELLED'}

        if missing_length:
            self.report({'ERROR'},
                        f"'length' missing on {missing_length}/{total_edges} edges. "
                        f"Run 'Add Edge Lengths' first.")
            return {'CANCELLED'}

        if missing_speed:
            self.report({'ERROR'},
                        f"'speed_kph' missing on {missing_speed}/{total_edges} edges. "
                        f"Run 'Add Edge Speeds' first.")
            return {'CANCELLED'}

        G = osmnx_analysis.add_edge_travel_times(G)
        
        if G is None:
            self.report({'ERROR'}, "Failed to add travel times")
            return {'CANCELLED'}

        verified = 0
        for _u, _v, data in _iter_edge_data(G):
            if "travel_time" in data and data["travel_time"] is not None:
                verified += 1

        if verified == 0:
            self.report({'ERROR'},
                        "OSMnx returned without errors but no edge received a "
                        "'travel_time' value. Check that 'length' and 'speed_kph' "
                        "are numeric on every edge.")
            return {'CANCELLED'}

        _store_osmnx_graph(obj, G)
        obj["osmnx_has_travel_times"] = True
        
        edges_transferred = _transfer_edge_attribute_to_mesh(obj, G, 'travel_time', 'edge_travel_time')
        
        self.report({'INFO'},
                    f"Travel times added: {verified}/{total_edges} graph edges, "
                    f"{edges_transferred} mesh edges updated")
        return {'FINISHED'}


class SCIGRAPHS_OT_CalculateBasicStats(bpy.types.Operator):
    """Calculate basic network statistics."""
    bl_idname = "scigraphs.osmnx_basic_stats"
    bl_label = "Calculate Basic Stats"
    bl_description = "Calculate node count, edge count, total length, circuity, etc."
    bl_options = {'REGISTER', 'UNDO'}
    
    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj and obj.get("is_osmnx", False)
    
    def execute(self, context):
        props = context.scene.scigraphs
        obj = context.active_object
        G = _get_osmnx_graph(obj)
        
        if G is None:
            self.report({'ERROR'}, "No OSMnx graph found")
            return {'CANCELLED'}
        
        area_km2 = props.osmnx_network_area if props.osmnx_network_area > 0 else None
        
        if area_km2 is None:
            area_km2 = osmnx_analysis.estimate_network_area(G)
            if area_km2:
                props.osmnx_network_area = area_km2
        
        stats = osmnx_analysis.get_basic_stats(G, area_km2=area_km2)
        
        if stats is None:
            self.report({'ERROR'}, "Failed to calculate statistics")
            return {'CANCELLED'}
        
        for key, value in stats.items():
            if value is not None:
                obj[f"osmnx_stat_{key}"] = value
        
        obj["osmnx_stats_calculated"] = True
        
        msg = f"Nodes: {stats['n_nodes']}, Edges: {stats['n_edges']}, Length: {stats['total_length_km']:.1f} km"
        self.report({'INFO'}, msg)
        return {'FINISHED'}


class SCIGRAPHS_OT_EstimateNetworkArea(bpy.types.Operator):
    """Estimate the area covered by the network."""
    bl_idname = "scigraphs.osmnx_estimate_area"
    bl_label = "Estimate Area"
    bl_description = "Calculate convex hull area of the network"
    bl_options = {'REGISTER', 'UNDO'}
    
    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj and obj.get("is_osmnx", False)
    
    def execute(self, context):
        props = context.scene.scigraphs
        obj = context.active_object
        G = _get_osmnx_graph(obj)
        
        if G is None:
            self.report({'ERROR'}, "No OSMnx graph found")
            return {'CANCELLED'}
        
        area = osmnx_analysis.estimate_network_area(G)
        
        if area is None:
            self.report({'ERROR'}, "Failed to estimate area")
            return {'CANCELLED'}
        
        props.osmnx_network_area = area
        
        self.report({'INFO'}, f"Estimated network area: {area:.2f} km2")
        return {'FINISHED'}


class SCIGRAPHS_OT_ConvertToUndirected(bpy.types.Operator):
    """Convert directed graph to undirected."""
    bl_idname = "scigraphs.osmnx_to_undirected"
    bl_label = "Convert to Undirected"
    bl_description = "Convert MultiDiGraph to undirected MultiGraph"
    bl_options = {'REGISTER', 'UNDO'}
    
    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj and obj.get("is_osmnx", False)
    
    def execute(self, context):
        from ....core.osmnx import convert
        
        obj = context.active_object
        G = _get_osmnx_graph(obj)
        
        if G is None:
            self.report({'ERROR'}, "No OSMnx graph found")
            return {'CANCELLED'}
        
        G_undirected = convert.to_undirected(G)
        
        if G_undirected is None:
            self.report({'ERROR'}, "Failed to convert graph")
            return {'CANCELLED'}
        
        from ....core import importer
        if hasattr(importer, '_osmnx_graph_cache'):
            graph_id = obj.get("osmnx_graph_id", obj.name)
            importer._osmnx_graph_cache[graph_id] = G_undirected
        
        obj["is_directed"] = False
        self.report({'INFO'}, f"Converted to undirected graph")
        
        return {'FINISHED'}


class SCIGRAPHS_OT_ConvertToDiGraph(bpy.types.Operator):
    """Convert MultiDiGraph to simple DiGraph."""
    bl_idname = "scigraphs.osmnx_to_digraph"
    bl_label = "Convert to DiGraph"
    bl_description = "Convert to simple DiGraph by selecting minimum weight edges"
    bl_options = {'REGISTER', 'UNDO'}
    
    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj and obj.get("is_osmnx", False)
    
    def execute(self, context):
        from ....core.osmnx import convert
        
        obj = context.active_object
        props = context.scene.scigraphs
        G = _get_osmnx_graph(obj)
        
        if G is None:
            self.report({'ERROR'}, "No OSMnx graph found")
            return {'CANCELLED'}
        
        weight = "length"
        G_simple = convert.to_digraph(G, weight=weight)
        
        if G_simple is None:
            self.report({'ERROR'}, "Failed to convert graph")
            return {'CANCELLED'}
        
        from ....core import importer
        if hasattr(importer, '_osmnx_graph_cache'):
            graph_id = obj.get("osmnx_graph_id", obj.name)
            importer._osmnx_graph_cache[graph_id] = G_simple

        # Keep the Blender-side flag aligned with the actual graph type
        # (mirror of SCIGRAPHS_OT_ConvertToUndirected setting it to False).
        obj["is_directed"] = True

        self.report({'INFO'}, f"Converted to DiGraph (by {weight})")
        
        return {'FINISHED'}


class SCIGRAPHS_OT_SimplifyGraph(bpy.types.Operator):
    """Simplify graph by removing interstitial nodes."""
    bl_idname = "scigraphs.osmnx_simplify"
    bl_label = "Simplify Graph"
    bl_description = "Remove nodes that are not intersections"
    bl_options = {'REGISTER', 'UNDO'}
    
    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj and obj.get("is_osmnx", False)
    
    def execute(self, context):
        from ....core.osmnx import simplification
        import osmnx as ox
        
        obj = context.active_object
        G = _get_osmnx_graph(obj)
        
        if G is None:
            self.report({'ERROR'}, "No OSMnx graph found")
            return {'CANCELLED'}
        
        if G.graph.get("simplified", False):
            self.report({'WARNING'}, "Graph is already simplified. No changes made.")
            return {'CANCELLED'}
        
        nodes_before = G.number_of_nodes()
        
        try:
            G_simplified = simplification.simplify_graph(G)
        except Exception as e:
            if "already been simplified" in str(e):
                self.report({'WARNING'}, "Graph is already simplified. No changes made.")
                return {'CANCELLED'}
            else:
                self.report({'ERROR'}, f"Failed to simplify graph: {str(e)}")
                return {'CANCELLED'}
        
        if G_simplified is None:
            self.report({'ERROR'}, "Failed to simplify graph")
            return {'CANCELLED'}
        
        from ....core import importer
        if hasattr(importer, '_osmnx_graph_cache'):
            graph_id = obj.get("osmnx_graph_id", obj.name)
            importer._osmnx_graph_cache[graph_id] = G_simplified
        
        nodes_after = G_simplified.number_of_nodes()
        self.report({'INFO'}, f"Simplified: {nodes_before} -> {nodes_after} nodes")
        
        return {'FINISHED'}


class SCIGRAPHS_OT_ConsolidateIntersections(bpy.types.Operator):
    """Consolidate nearby intersections into single nodes."""
    bl_idname = "scigraphs.osmnx_consolidate"
    bl_label = "Consolidate Intersections"
    bl_description = "Merge nearby intersections within tolerance distance"
    bl_options = {'REGISTER', 'UNDO'}
    
    tolerance: FloatProperty(
        name="Tolerance",
        description="Distance threshold in meters",
        default=10.0,
        min=1.0,
        max=100.0,
    )
    
    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj and obj.get("is_osmnx", False)

    def invoke(self, context, _event):
        props = context.scene.scigraphs
        self.tolerance = props.osmnx_simplification_tolerance
        return context.window_manager.invoke_props_dialog(self)
    
    def execute(self, context):
        from ....core.osmnx import simplification
        
        obj = context.active_object
        G = _get_osmnx_graph(obj)
        
        if G is None:
            self.report({'ERROR'}, "No OSMnx graph found")
            return {'CANCELLED'}

        if not osmnx_analysis.is_graph_projected(G):
            self.report(
                {'ERROR'},
                "Project the graph first. Consolidation tolerance is in meters and cannot be applied safely in WGS84.",
            )
            return {'CANCELLED'}
        
        if not G.is_directed():
            self.report({'ERROR'}, "Graph must be directed (MultiDiGraph) for consolidation. Convert from undirected first or use original directed graph.")
            return {'CANCELLED'}
        
        if G.graph.get("streets_per_node_consolidated", False):
            self.report({'WARNING'}, "Graph intersections are already consolidated. No changes made.")
            return {'CANCELLED'}
        
        nodes_before = G.number_of_nodes()
        
        try:
            G_consolidated = simplification.consolidate_intersections(G, tolerance=self.tolerance)
        except Exception as e:
            if "already been consolidated" in str(e):
                self.report({'WARNING'}, "Graph intersections are already consolidated. No changes made.")
                return {'CANCELLED'}
            elif "undirected" in str(e).lower():
                self.report({'ERROR'}, "Graph must be directed for consolidation. Convert from undirected or use original directed graph.")
                return {'CANCELLED'}
            else:
                self.report({'ERROR'}, f"Failed to consolidate intersections: {str(e)}")
                return {'CANCELLED'}
        
        if G_consolidated is None:
            self.report({'ERROR'}, "Failed to consolidate intersections")
            return {'CANCELLED'}

        nodes_after = G_consolidated.number_of_nodes()
        if nodes_before > 1 and nodes_after <= 1:
            self.report(
                {'ERROR'},
                "Consolidation collapsed the graph to one node; result was discarded. Check CRS and tolerance.",
            )
            return {'CANCELLED'}
        
        from ....core import importer
        if hasattr(importer, '_osmnx_graph_cache'):
            graph_id = obj.get("osmnx_graph_id", obj.name)
            importer._osmnx_graph_cache[graph_id] = G_consolidated
        
        self.report({'INFO'}, f"Consolidated: {nodes_before} -> {nodes_after} nodes")
        
        return {'FINISHED'}


class SCIGRAPHS_OT_TruncateBBox(bpy.types.Operator):
    """Truncate graph to bounding box."""
    bl_idname = "scigraphs.osmnx_truncate_bbox"
    bl_label = "Truncate to BBox"
    bl_description = "Remove nodes outside bounding box"
    bl_options = {'REGISTER', 'UNDO'}
    
    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj and obj.get("is_osmnx", False)
    
    def execute(self, context):
        from ....core.osmnx import truncate
        
        obj = context.active_object
        props = context.scene.scigraphs
        G = _get_osmnx_graph(obj)
        
        if G is None:
            self.report({'ERROR'}, "No OSMnx graph found")
            return {'CANCELLED'}
        
        bbox = (props.osmnx_bbox_north, props.osmnx_bbox_south, 
                props.osmnx_bbox_east, props.osmnx_bbox_west)
        
        nodes_before = G.number_of_nodes()
        
        G_truncated = truncate.truncate_graph_bbox(G, bbox, truncate_by_edge=props.osmnx_truncate_by_edge)
        
        if G_truncated is None:
            self.report({'ERROR'}, "Failed to truncate graph")
            return {'CANCELLED'}
        
        from ....core import importer
        if hasattr(importer, '_osmnx_graph_cache'):
            graph_id = obj.get("osmnx_graph_id", obj.name)
            importer._osmnx_graph_cache[graph_id] = G_truncated
        
        nodes_after = G_truncated.number_of_nodes()
        self.report({'INFO'}, f"Truncated: {nodes_before} -> {nodes_after} nodes")
        
        return {'FINISHED'}


class SCIGRAPHS_OT_LargestComponent(bpy.types.Operator):
    """Extract largest connected component."""
    bl_idname = "scigraphs.osmnx_largest_component"
    bl_label = "Largest Component"
    bl_description = "Keep only the largest connected component"
    bl_options = {'REGISTER', 'UNDO'}
    
    strongly: BoolProperty(
        name="Strongly Connected",
        description="Use strongly connected components (vs. weakly)",
        default=False,
    )
    
    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj and obj.get("is_osmnx", False)
    
    def execute(self, context):
        from ....core.osmnx import truncate
        
        obj = context.active_object
        G = _get_osmnx_graph(obj)
        
        if G is None:
            self.report({'ERROR'}, "No OSMnx graph found")
            return {'CANCELLED'}
        
        nodes_before = G.number_of_nodes()
        
        G_component = truncate.largest_component(G, strongly=self.strongly)
        
        if G_component is None:
            self.report({'ERROR'}, "Failed to extract component")
            return {'CANCELLED'}
        
        from ....core import importer
        if hasattr(importer, '_osmnx_graph_cache'):
            graph_id = obj.get("osmnx_graph_id", obj.name)
            importer._osmnx_graph_cache[graph_id] = G_component
        
        nodes_after = G_component.number_of_nodes()
        self.report({'INFO'}, f"Largest component: {nodes_after} / {nodes_before} nodes")
        
        return {'FINISHED'}


class SCIGRAPHS_OT_GeocodeAddress(bpy.types.Operator):
    """Geocode address to coordinates."""
    bl_idname = "scigraphs.osmnx_geocode"
    bl_label = "Geocode Address"
    bl_description = "Convert address to latitude/longitude coordinates"
    bl_options = {'REGISTER', 'UNDO'}
    
    address: StringProperty(
        name="Address",
        description="Address or place name to geocode",
        default="",
    )
    
    def execute(self, context):
        from ....core.osmnx import geocoder
        
        if not self.address.strip():
            self.report({'ERROR'}, "Please enter an address")
            return {'CANCELLED'}
        
        coords = geocoder.geocode(self.address)
        
        if coords is None:
            self.report({'ERROR'}, "Failed to geocode address")
            return {'CANCELLED'}
        
        lat, lon = coords
        props = context.scene.scigraphs
        props.osmnx_latitude = lat
        props.osmnx_longitude = lon
        
        self.report({'INFO'}, f"Geocoded: ({lat:.6f}, {lon:.6f})")
        
        return {'FINISHED'}
    
    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)


class SCIGRAPHS_OT_OrientationEntropy(bpy.types.Operator):
    """Calculate orientation entropy of street network."""
    bl_idname = "scigraphs.osmnx_orientation_entropy"
    bl_label = "Orientation Entropy"
    bl_description = "Calculate Shannon entropy of street orientations"
    bl_options = {'REGISTER', 'UNDO'}
    
    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj and obj.get("is_osmnx", False)
    
    def execute(self, context):
        from ....core.osmnx import bearing
        
        obj = context.active_object
        G = _get_unprojected_graph(obj)
        
        if G is None:
            self.report({'ERROR'}, "No unprojected OSMnx graph found. Bearing analysis requires unprojected (WGS84) graph.")
            return {'CANCELLED'}
        
        entropy = bearing.orientation_entropy(G)
        
        if entropy is None:
            self.report({'ERROR'}, "Failed to calculate entropy")
            return {'CANCELLED'}
        
        obj["orientation_entropy"] = entropy
        self.report({'INFO'}, f"Orientation entropy: {entropy:.4f}")
        
        return {'FINISHED'}


class SCIGRAPHS_OT_CalculateCircuity(bpy.types.Operator):
    """Calculate average circuity of street network."""
    bl_idname = "scigraphs.osmnx_circuity"
    bl_label = "Calculate Circuity"
    bl_description = "Calculate ratio of network distance to straight-line distance"
    bl_options = {'REGISTER', 'UNDO'}
    
    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj and obj.get("is_osmnx", False)
    
    def execute(self, context):
        from ....core.osmnx import convert, stats
        
        obj = context.active_object
        G = _get_osmnx_graph(obj)
        
        if G is None:
            self.report({'ERROR'}, "No OSMnx graph found")
            return {'CANCELLED'}
        
        G_undirected = convert.to_undirected(G)
        if G_undirected is None:
            self.report({'ERROR'}, "Failed to convert to undirected")
            return {'CANCELLED'}
        
        circuity = stats.circuity_avg(G_undirected)
        
        if circuity is None:
            self.report({'ERROR'}, "Failed to calculate circuity")
            return {'CANCELLED'}
        
        obj["circuity_avg"] = circuity
        self.report({'INFO'}, f"Average circuity: {circuity:.3f}")
        
        return {'FINISHED'}


class SCIGRAPHS_OT_CalculateBearingPoints(bpy.types.Operator):
    """Calculate bearing between two coordinate points."""
    bl_idname = "scigraphs.osmnx_calculate_bearing_points"
    bl_label = "Calculate Bearing"
    bl_description = "Calculate compass bearing between two lat/lon points"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        from ....core.osmnx import bearing
        
        props = context.scene.scigraphs
        
        result = bearing.calculate_bearing(
            props.osmnx_bearing_lat1,
            props.osmnx_bearing_lon1,
            props.osmnx_bearing_lat2,
            props.osmnx_bearing_lon2
        )
        
        if result is None:
            self.report({'ERROR'}, "Failed to calculate bearing")
            return {'CANCELLED'}
        
        props.osmnx_bearing_result = result
        self.report({'INFO'}, f"Bearing: {result:.2f}°")
        
        return {'FINISHED'}


class SCIGRAPHS_OT_BearingsDistribution(bpy.types.Operator):
    """Calculate bearing distribution histogram."""
    bl_idname = "scigraphs.osmnx_bearings_distribution"
    bl_label = "Bearing Distribution"
    bl_description = "Calculate distribution of edge bearings in bins"
    bl_options = {'REGISTER', 'UNDO'}
    
    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj and obj.get("is_osmnx", False)
    
    def execute(self, context):
        from ....core.osmnx import bearing
        
        obj = context.active_object
        G = _get_unprojected_graph(obj)
        
        if G is None:
            self.report({'ERROR'}, "No unprojected OSMnx graph found. Bearing analysis requires unprojected (WGS84) graph.")
            return {'CANCELLED'}
        
        props = context.scene.scigraphs
        num_bins = props.osmnx_bearing_num_bins
        
        bearings = bearing.get_bearings_distribution(G, num_bins=num_bins)
        
        if bearings is None or bearings[0] is None:
            self.report({'ERROR'}, "Failed to calculate bearing distribution")
            return {'CANCELLED'}
        
        obj["bearing_distribution"] = str(list(bearings[0]))
        obj["bearing_num_bins"] = num_bins
        
        self.report({'INFO'}, f"Bearing distribution calculated ({num_bins} bins)")
        
        return {'FINISHED'}


class SCIGRAPHS_OT_CalcEuclidean(bpy.types.Operator):
    """Calculate Euclidean distance between two points."""
    bl_idname = "scigraphs.osmnx_calc_euclidean"
    bl_label = "Euclidean Distance"
    bl_description = "Calculate Euclidean distance between two points"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        from ....core.osmnx import distance
        
        props = context.scene.scigraphs
        
        dist = distance.euclidean(
            props.osmnx_dist_y1,
            props.osmnx_dist_x1,
            props.osmnx_dist_y2,
            props.osmnx_dist_x2
        )
        
        if dist is None:
            self.report({'ERROR'}, "Failed to calculate distance")
            return {'CANCELLED'}
        
        props.osmnx_distance_result = dist
        self.report({'INFO'}, f"Euclidean distance: {dist:.2f} units")
        
        return {'FINISHED'}


class SCIGRAPHS_OT_CalcGreatCircle(bpy.types.Operator):
    """Calculate great circle distance between two points."""
    bl_idname = "scigraphs.osmnx_calc_great_circle"
    bl_label = "Great Circle Distance"
    bl_description = "Calculate great circle (haversine) distance between two lat/lon points"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        from ....core.osmnx import distance
        
        props = context.scene.scigraphs
        
        dist = distance.great_circle(
            props.osmnx_dist_y1,
            props.osmnx_dist_x1,
            props.osmnx_dist_y2,
            props.osmnx_dist_x2
        )
        
        if dist is None:
            self.report({'ERROR'}, "Failed to calculate distance")
            return {'CANCELLED'}
        
        props.osmnx_distance_result = dist
        dist_km = dist / 1000.0
        self.report({'INFO'}, f"Great circle distance: {dist_km:.3f} km ({dist:.1f} m)")
        
        return {'FINISHED'}


class SCIGRAPHS_OT_GraphToGDFs(bpy.types.Operator):
    """Convert graph to GeoDataFrames."""
    bl_idname = "scigraphs.osmnx_graph_to_gdfs"
    bl_label = "Graph to GeoDataFrames"
    bl_description = "Export graph as node and edge GeoDataFrames (saved as files)"
    bl_options = {'REGISTER'}
    
    filepath: StringProperty(
        name="File Path",
        description="Base path for GeoDataFrame files",
        subtype='FILE_PATH',
    )
    
    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj and obj.get("is_osmnx", False)
    
    def invoke(self, context, event):
        import os

        obj = context.active_object
        place = obj.get("osmnx_place", "network")
        place_clean = place.replace(",", "").replace(" ", "_")[:30] if place else "network"

        # Blender 5.x emits a RuntimeWarning if a FILE_PATH property is set with the
        # blend-relative "//" prefix before the file selector opens.  Resolve to an
        # absolute default path (next to the .blend, or HOME if unsaved).
        if bpy.data.filepath:
            default_dir = os.path.dirname(bpy.data.filepath)
        else:
            default_dir = os.path.expanduser("~")
        self.filepath = os.path.join(default_dir, f"{place_clean}_nodes.geojson")

        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}
    
    def execute(self, context):
        from ....core.osmnx import convert
        
        obj = context.active_object
        G = _get_osmnx_graph(obj)
        
        if G is None:
            self.report({'ERROR'}, "No OSMnx graph found")
            return {'CANCELLED'}
        
        gdfs = convert.graph_to_gdfs(G, nodes=True, edges=True)
        
        if gdfs is None:
            self.report({'ERROR'}, "Failed to convert graph")
            return {'CANCELLED'}
        
        gdf_nodes, gdf_edges = gdfs
        
        filepath = bpy.path.abspath(self.filepath)
        base_path = filepath.rsplit('.', 1)[0] if '.' in filepath else filepath
        
        nodes_path = f"{base_path}_nodes.geojson"
        edges_path = f"{base_path}_edges.geojson"
        
        gdf_nodes.to_file(nodes_path, driver='GeoJSON')
        gdf_edges.to_file(edges_path, driver='GeoJSON')
        
        self.report({'INFO'}, f"GeoDataFrames saved: {len(gdf_nodes)} nodes, {len(gdf_edges)} edges")
        
        return {'FINISHED'}


class SCIGRAPHS_OT_GDFsToGraph(bpy.types.Operator):
    """Convert GeoDataFrames to graph."""
    bl_idname = "scigraphs.osmnx_gdfs_to_graph"
    bl_label = "GeoDataFrames to Graph"
    bl_description = "Import graph from node and edge GeoDataFrame files"
    bl_options = {'REGISTER', 'UNDO'}
    
    nodes_filepath: StringProperty(
        name="Nodes File",
        description="Path to nodes GeoDataFrame file",
        subtype='FILE_PATH',
    )
    
    edges_filepath: StringProperty(
        name="Edges File",
        description="Path to edges GeoDataFrame file",
        subtype='FILE_PATH',
    )
    
    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self, width=400)
    
    def draw(self, context):
        layout = self.layout
        layout.prop(self, "nodes_filepath", text="Nodes")
        layout.prop(self, "edges_filepath", text="Edges")
    
    def execute(self, context):
        import geopandas as gpd
        from ....core.osmnx import convert
        from ....core import geometry, importer
        
        nodes_path = bpy.path.abspath(self.nodes_filepath)
        edges_path = bpy.path.abspath(self.edges_filepath)
        
        gdf_nodes = gpd.read_file(nodes_path)
        gdf_edges = gpd.read_file(edges_path)
        
        G = convert.graph_from_gdfs(gdf_nodes, gdf_edges)
        
        if G is None:
            self.report({'ERROR'}, "Failed to create graph from GeoDataFrames")
            return {'CANCELLED'}
        
        graph_data, edge_geometries = importer.osmnx_to_graph_data(G, retain_geometry=True)
        
        if graph_data is None:
            self.report({'ERROR'}, "Failed to convert graph")
            return {'CANCELLED'}
        
        props = context.scene.scigraphs
        scale = props.osmnx_scale
        
        obj = geometry.create_osmnx_graph_object(
            graph_data, edge_geometries, scale=scale, retain_geometry=True
        )
        
        if obj:
            _store_osmnx_graph(obj, G)
            obj["osmnx_scale"] = scale
            
            self.report({'INFO'}, f"Graph created from GeoDataFrames: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
            return {'FINISHED'}
        
        self.report({'ERROR'}, "Failed to create graph object")
        return {'CANCELLED'}


classes = [
    SCIGRAPHS_OT_ProjectGraph,
    SCIGRAPHS_OT_AddEdgeLengths,
    SCIGRAPHS_OT_AddEdgeBearings,
    SCIGRAPHS_OT_AddEdgeSpeeds,
    SCIGRAPHS_OT_AddEdgeTravelTimes,
    SCIGRAPHS_OT_CalculateBasicStats,
    SCIGRAPHS_OT_EstimateNetworkArea,
    SCIGRAPHS_OT_ConvertToUndirected,
    SCIGRAPHS_OT_ConvertToDiGraph,
    SCIGRAPHS_OT_SimplifyGraph,
    SCIGRAPHS_OT_ConsolidateIntersections,
    SCIGRAPHS_OT_TruncateBBox,
    SCIGRAPHS_OT_LargestComponent,
    SCIGRAPHS_OT_GeocodeAddress,
    SCIGRAPHS_OT_OrientationEntropy,
    SCIGRAPHS_OT_CalculateCircuity,
    SCIGRAPHS_OT_CalculateBearingPoints,
    SCIGRAPHS_OT_BearingsDistribution,
    SCIGRAPHS_OT_CalcEuclidean,
    SCIGRAPHS_OT_CalcGreatCircle,
    SCIGRAPHS_OT_GraphToGDFs,
    SCIGRAPHS_OT_GDFsToGraph,
]


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)

