import bpy
from bpy.props import StringProperty, EnumProperty, FloatProperty, IntProperty, BoolProperty

from ....core.mesh.geo_mesh import create_feature_mesh_from_gdf as _create_feature_mesh_from_gdf


# Shared tag presets (used by From-Address / From-Polygon / From-XML).
# Covers the most common OSM categories referenced across the notebooks.
FEATURE_TAG_PRESETS = {
    'BUILDING':          {"building": True},
    'AMENITY':           {"amenity": True},
    'RESTAURANT':        {"amenity": ["restaurant"]},
    'SHOP':              {"shop": True},
    'LEISURE':           {"leisure": True},
    'PARKING':           {"amenity": ["parking"]},
    'BUS_STOP':          {"highway": ["bus_stop"]},
    'RAIL_STATION':      {"railway": ["station", "halt", "stop", "tram_stop"]},
    'PARK':              {"leisure": ["park", "garden", "nature_reserve", "playground"]},
    'EDUCATION':         {"amenity": ["school", "university", "college", "kindergarten", "library"]},
    'HEALTH':            {"amenity": ["hospital", "clinic", "doctors", "pharmacy", "dentist"]},
    'AMENITY_METAPATH':  {"amenity": ["cafe", "restaurant", "pub", "bar", "museum", "theatre", "cinema"]},
    'LANDUSE':           {"landuse": True},
    'NATURAL':           {"natural": True},
    'WATER':             {"natural": ["water"]},
    'HIGHWAY':           {"highway": True},
}


def _tags_from_preset(preset, custom_tags_str=""):
    """Resolve a preset enum (or 'CUSTOM') to an OSM tags dict."""
    if preset != 'CUSTOM':
        return dict(FEATURE_TAG_PRESETS.get(preset, {"building": True}))
    tags = {}
    for pair in custom_tags_str.split(','):
        if '=' in pair:
            key, value = pair.split('=', 1)
            tags[key.strip()] = value.strip()
    return tags


def _find_osmnx_object(context):
    """Return the active OSMnx street network object (active or any in scene)."""
    obj = context.active_object
    if obj and obj.get("is_osmnx", False):
        return obj
    for o in bpy.data.objects:
        if o.get("is_osmnx", False):
            return o
    return None


class SCIGRAPHS_OT_FeaturesFromPlace(bpy.types.Operator):
    """Download OSM features from a place."""
    bl_idname = "scigraphs.osmnx_features_place"
    bl_label = "Features from Place"
    bl_description = "Download buildings/POIs from a named place"
    bl_options = {'REGISTER', 'UNDO'}
    
    place: StringProperty(
        name="Place Name",
        description="Name of place (e.g., 'Manhattan, New York, USA')",
        default="",
    )
    
    feature_type: EnumProperty(
        name="Feature Type",
        items=[(k, k.replace('_', ' ').title(), "") for k in FEATURE_TAG_PRESETS]
            + [('CUSTOM', "Custom Tags", "Use custom OSM tags")],
        default='BUILDING',
    )
    
    custom_tags: StringProperty(
        name="Custom Tags",
        description="Custom OSM tags as key=value pairs (comma-separated)",
        default="",
    )
    
    filter_nodes_only: BoolProperty(
        name="Nodes Only",
        description="Filter to include only node elements (exclude ways/relations). Enable for notebook-exact results",
        default=False,
    )
    
    def execute(self, context):
        from ....core.osmnx import features
        
        if not self.place.strip():
            self.report({'ERROR'}, "Please enter a place name")
            return {'CANCELLED'}
        
        tags = _tags_from_preset(self.feature_type, self.custom_tags)
        
        if not tags:
            self.report({'ERROR'}, "No tags specified")
            return {'CANCELLED'}
        
        self.report({'INFO'}, f"Downloading features from {self.place}...")
        
        gdf = features.features_from_place(self.place, tags)
        
        if gdf is None or len(gdf) == 0:
            self.report({'ERROR'}, "No features found")
            return {'CANCELLED'}
        
        # Filter to nodes only if requested (matching notebook behavior)
        if self.filter_nodes_only and hasattr(gdf.index, 'names'):
            initial_count = len(gdf)
            # Find the element type level in the multiindex
            element_level_name = None
            for name in gdf.index.names:
                if name and 'element' in name.lower():
                    element_level_name = name
                    break
            
            if element_level_name:
                gdf = gdf[gdf.index.get_level_values(element_level_name) == "node"]
                filtered_count = len(gdf)
                self.report({'INFO'}, f"Filtered to nodes only: {initial_count} → {filtered_count} features")
                
                if filtered_count == 0:
                    self.report({'ERROR'}, "No node elements found after filtering")
                    return {'CANCELLED'}
        
        # Find OSMnx object in scene for coordinate transformation
        osmnx_obj = None
        if context.active_object and context.active_object.get("is_osmnx", False):
            osmnx_obj = context.active_object
        else:
            # Search for any OSMnx object in scene
            for obj in bpy.data.objects:
                if obj.get("is_osmnx", False):
                    osmnx_obj = obj
                    break
        
        if not osmnx_obj:
            self.report({'WARNING'}, "No OSMnx street network found - features may not align correctly. Download a street network first.")
        
        objects = _create_feature_mesh_from_gdf(gdf, name=f"{self.place}_{self.feature_type}", osmnx_obj=osmnx_obj)
        
        if not objects:
            self.report({'ERROR'}, "Failed to create mesh objects")
            return {'CANCELLED'}
        
        # Store place name and metadata in objects for later use (e.g., metapath analysis)
        for obj in objects:
            obj["place_name"] = self.place
            obj["feature_type"] = self.feature_type
            obj["crs"] = str(gdf.crs) if gdf.crs else "EPSG:4326"
            
            # Store coordinate transformation info from OSMnx object
            if osmnx_obj:
                obj["osmnx_center_lat"] = osmnx_obj.get("osmnx_center_lat")
                obj["osmnx_center_lon"] = osmnx_obj.get("osmnx_center_lon")
                obj["osmnx_scale"] = osmnx_obj.get("osmnx_scale", 0.001)
        
        if objects:
            context.view_layer.objects.active = objects[-1]
            objects[-1].select_set(True)
        
        self.report({'INFO'}, f"Created {len(objects)} object(s) with {len(gdf)} features")
        
        return {'FINISHED'}
    
    def invoke(self, context, event):
        props = context.scene.scigraphs
        self.place = props.osmnx_features_place or props.osmnx_place_name
        self.feature_type = props.osmnx_feature_type
        self.custom_tags = props.osmnx_custom_tags
        return context.window_manager.invoke_props_dialog(self)


class SCIGRAPHS_OT_FeaturesFromPoint(bpy.types.Operator):
    """Download OSM features near a point."""
    bl_idname = "scigraphs.osmnx_features_point"
    bl_label = "Features from Point"
    bl_description = "Download features within distance of a point"
    bl_options = {'REGISTER', 'UNDO'}
    
    latitude: FloatProperty(
        name="Latitude",
        description="Center latitude",
        default=0.0,
        min=-90.0,
        max=90.0,
        precision=6,
    )
    
    longitude: FloatProperty(
        name="Longitude",
        description="Center longitude",
        default=0.0,
        min=-180.0,
        max=180.0,
        precision=6,
    )
    
    distance: IntProperty(
        name="Distance (m)",
        description="Radius in meters",
        default=1000,
        min=100,
        max=10000,
    )
    
    feature_type: EnumProperty(
        name="Feature Type",
        items=[(k, k.replace('_', ' ').title(), "") for k in FEATURE_TAG_PRESETS]
            + [('CUSTOM', "Custom Tags", "Custom key=value,...")],
        default='BUILDING',
    )

    custom_tags: StringProperty(name="Custom Tags", default="")
    
    filter_nodes_only: BoolProperty(
        name="Nodes Only",
        description="Filter to include only node elements (exclude ways/relations). Enable for notebook-exact results",
        default=False,
    )
    
    def execute(self, context):
        from ....core.osmnx import features
        
        center_point = (self.latitude, self.longitude)
        
        tags = _tags_from_preset(self.feature_type, self.custom_tags)
        if not tags:
            self.report({'ERROR'}, "No tags specified")
            return {'CANCELLED'}
        
        self.report({'INFO'}, f"Downloading features...")
        
        gdf = features.features_from_point(center_point, tags, dist=self.distance)
        
        if gdf is None or len(gdf) == 0:
            self.report({'ERROR'}, "No features found")
            return {'CANCELLED'}
        
        # Filter to nodes only if requested (matching notebook behavior)
        if self.filter_nodes_only and hasattr(gdf.index, 'names'):
            initial_count = len(gdf)
            # Find the element type level in the multiindex
            element_level_name = None
            for name in gdf.index.names:
                if name and 'element' in name.lower():
                    element_level_name = name
                    break
            
            if element_level_name:
                gdf = gdf[gdf.index.get_level_values(element_level_name) == "node"]
                filtered_count = len(gdf)
                self.report({'INFO'}, f"Filtered to nodes only: {initial_count} → {filtered_count} features")
                
                if filtered_count == 0:
                    self.report({'ERROR'}, "No node elements found after filtering")
                    return {'CANCELLED'}
        
        # Find OSMnx object in scene for coordinate transformation
        osmnx_obj = None
        if context.active_object and context.active_object.get("is_osmnx", False):
            osmnx_obj = context.active_object
        else:
            # Search for any OSMnx object in scene
            for obj in bpy.data.objects:
                if obj.get("is_osmnx", False):
                    osmnx_obj = obj
                    break
        
        objects = _create_feature_mesh_from_gdf(gdf, name=f"OSM_{self.feature_type}", osmnx_obj=osmnx_obj)
        
        if not objects:
            self.report({'ERROR'}, "Failed to create mesh objects")
            return {'CANCELLED'}
        
        # Store metadata for later use
        for obj in objects:
            obj["place_name"] = f"Point ({self.latitude:.4f}, {self.longitude:.4f})"
            obj["feature_type"] = self.feature_type
            obj["center_lat"] = self.latitude
            obj["center_lon"] = self.longitude
            obj["search_radius"] = self.distance
            obj["crs"] = str(gdf.crs) if gdf.crs else "EPSG:4326"
            
            # Store coordinate transformation info from OSMnx object
            if osmnx_obj:
                obj["osmnx_center_lat"] = osmnx_obj.get("osmnx_center_lat")
                obj["osmnx_center_lon"] = osmnx_obj.get("osmnx_center_lon")
                obj["osmnx_scale"] = osmnx_obj.get("osmnx_scale", 0.001)
        
        if objects:
            context.view_layer.objects.active = objects[-1]
            objects[-1].select_set(True)
        
        self.report({'INFO'}, f"Created {len(objects)} object(s) with {len(gdf)} features")
        
        return {'FINISHED'}
    
    def invoke(self, context, event):
        props = context.scene.scigraphs
        self.latitude = props.osmnx_latitude
        self.longitude = props.osmnx_longitude
        self.distance = props.osmnx_features_distance
        self.feature_type = props.osmnx_feature_type
        self.custom_tags = props.osmnx_custom_tags
        return context.window_manager.invoke_props_dialog(self)


class SCIGRAPHS_OT_FeaturesFromBBox(bpy.types.Operator):
    """Download OSM features in bounding box."""
    bl_idname = "scigraphs.osmnx_features_bbox"
    bl_label = "Features from BBox"
    bl_description = "Download features within a bounding box"
    bl_options = {'REGISTER', 'UNDO'}
    
    feature_type: EnumProperty(
        name="Feature Type",
        items=[(k, k.replace('_', ' ').title(), "") for k in FEATURE_TAG_PRESETS]
            + [('CUSTOM', "Custom Tags", "Custom key=value,...")],
        default='BUILDING',
    )

    custom_tags: StringProperty(name="Custom Tags", default="")

    def invoke(self, context, event):
        props = context.scene.scigraphs
        self.feature_type = props.osmnx_feature_type
        self.custom_tags = props.osmnx_custom_tags
        return context.window_manager.invoke_props_dialog(self)
    
    def execute(self, context):
        from ....core.osmnx import features
        
        props = context.scene.scigraphs
        bbox = (props.osmnx_bbox_north, props.osmnx_bbox_south,
                props.osmnx_bbox_east, props.osmnx_bbox_west)
        
        tags = _tags_from_preset(self.feature_type, self.custom_tags)
        if not tags:
            self.report({'ERROR'}, "No tags specified")
            return {'CANCELLED'}
        
        self.report({'INFO'}, f"Downloading features...")
        
        gdf = features.features_from_bbox(bbox, tags)
        
        if gdf is None or len(gdf) == 0:
            self.report({'ERROR'}, "No features found")
            return {'CANCELLED'}
        
        # Find OSMnx object in scene for coordinate transformation
        osmnx_obj = None
        if context.active_object and context.active_object.get("is_osmnx", False):
            osmnx_obj = context.active_object
        else:
            # Search for any OSMnx object in scene
            for obj in bpy.data.objects:
                if obj.get("is_osmnx", False):
                    osmnx_obj = obj
                    break
        
        objects = _create_feature_mesh_from_gdf(gdf, name=f"OSM_{self.feature_type}", osmnx_obj=osmnx_obj)
        
        if not objects:
            self.report({'ERROR'}, "Failed to create mesh objects")
            return {'CANCELLED'}
        
        # Store metadata for later use
        for obj in objects:
            obj["place_name"] = f"BBox ({bbox[0]:.4f}, {bbox[1]:.4f}, {bbox[2]:.4f}, {bbox[3]:.4f})"
            obj["feature_type"] = self.feature_type
            obj["bbox_north"] = bbox[0]
            obj["bbox_south"] = bbox[1]
            obj["bbox_east"] = bbox[2]
            obj["bbox_west"] = bbox[3]
            obj["crs"] = str(gdf.crs) if gdf.crs else "EPSG:4326"
            
            # Store coordinate transformation info from OSMnx object
            if osmnx_obj:
                obj["osmnx_center_lat"] = osmnx_obj.get("osmnx_center_lat")
                obj["osmnx_center_lon"] = osmnx_obj.get("osmnx_center_lon")
                obj["osmnx_scale"] = osmnx_obj.get("osmnx_scale", 0.001)
        
        if objects:
            context.view_layer.objects.active = objects[-1]
            objects[-1].select_set(True)
        
        self.report({'INFO'}, f"Created {len(objects)} object(s) with {len(gdf)} features")
        
        return {'FINISHED'}


class SCIGRAPHS_OT_FeaturesFromAddress(bpy.types.Operator):
    """Download OSM features around a postal address."""
    bl_idname = "scigraphs.osmnx_features_address"
    bl_label = "Features from Address"
    bl_description = "Download features within a radius of a geocoded address"
    bl_options = {'REGISTER', 'UNDO'}

    address: StringProperty(
        name="Address",
        description="Postal address or landmark to geocode",
        default="",
    )

    distance: IntProperty(
        name="Distance (m)",
        description="Radius in meters",
        default=1000,
        min=100,
        max=10000,
    )

    feature_preset: EnumProperty(
        name="Feature Preset",
        items=[(k, k.replace('_', ' ').title(), "") for k in FEATURE_TAG_PRESETS]
            + [('CUSTOM', "Custom Tags", "Custom key=value,...")],
        default='BUILDING',
    )

    custom_tags: StringProperty(name="Custom Tags", default="")

    def invoke(self, context, event):
        props = context.scene.scigraphs
        self.address = props.osmnx_geocode_address or props.osmnx_address
        self.distance = props.osmnx_features_distance
        if props.osmnx_feature_type in FEATURE_TAG_PRESETS or props.osmnx_feature_type == 'CUSTOM':
            self.feature_preset = props.osmnx_feature_type
        self.custom_tags = props.osmnx_custom_tags
        return context.window_manager.invoke_props_dialog(self)

    def execute(self, context):
        from ....core.osmnx import features

        if not self.address.strip():
            self.report({'ERROR'}, "Please enter an address")
            return {'CANCELLED'}

        tags = _tags_from_preset(self.feature_preset, self.custom_tags)
        if not tags:
            self.report({'ERROR'}, "No tags specified")
            return {'CANCELLED'}

        gdf = features.features_from_address(self.address, tags, dist=self.distance)
        if gdf is None or len(gdf) == 0:
            self.report({'ERROR'}, "No features found")
            return {'CANCELLED'}

        osmnx_obj = _find_osmnx_object(context)
        name = f"{self.address[:40]}_{self.feature_preset}"
        objects = _create_feature_mesh_from_gdf(gdf, name=name, osmnx_obj=osmnx_obj)
        if not objects:
            self.report({'ERROR'}, "Failed to create meshes")
            return {'CANCELLED'}

        for o in objects:
            o["place_name"] = self.address
            o["feature_type"] = self.feature_preset
            o["crs"] = str(gdf.crs) if gdf.crs else "EPSG:4326"
            if osmnx_obj:
                o["osmnx_center_lat"] = osmnx_obj.get("osmnx_center_lat")
                o["osmnx_center_lon"] = osmnx_obj.get("osmnx_center_lon")
                o["osmnx_scale"] = osmnx_obj.get("osmnx_scale", 0.001)

        self.report({'INFO'}, f"Created {len(objects)} object(s) with {len(gdf)} features")
        return {'FINISHED'}


class SCIGRAPHS_OT_FeaturesFromPolygon(bpy.types.Operator):
    """Download OSM features inside a user-selected polygon object."""
    bl_idname = "scigraphs.osmnx_features_polygon"
    bl_label = "Features from Polygon"
    bl_description = "Download features inside a Blender mesh polygon (vertices as lon/lat)"
    bl_options = {'REGISTER', 'UNDO'}

    feature_preset: EnumProperty(
        name="Feature Preset",
        items=[(k, k.replace('_', ' ').title(), "") for k in FEATURE_TAG_PRESETS]
            + [('CUSTOM', "Custom Tags", "Custom key=value,...")],
        default='BUILDING',
    )

    custom_tags: StringProperty(name="Custom Tags", default="")

    @classmethod
    def poll(cls, context):
        return context.active_object is not None

    def invoke(self, context, event):
        props = context.scene.scigraphs
        if props.osmnx_feature_type in FEATURE_TAG_PRESETS or props.osmnx_feature_type == 'CUSTOM':
            self.feature_preset = props.osmnx_feature_type
        self.custom_tags = props.osmnx_custom_tags
        return context.window_manager.invoke_props_dialog(self)

    def execute(self, context):
        from ....core.osmnx import features
        try:
            from shapely.geometry import Polygon
        except ImportError:
            self.report({'ERROR'}, "Shapely required for POLYGON features")
            return {'CANCELLED'}

        poly_obj = context.active_object
        if poly_obj.type != 'MESH':
            self.report({'ERROR'}, "Active object must be a mesh (vertices as lon/lat)")
            return {'CANCELLED'}

        mesh = poly_obj.data
        if len(mesh.polygons) > 0:
            face = mesh.polygons[0]
            verts = [tuple(mesh.vertices[vi].co.xy) for vi in face.vertices]
        else:
            verts = [(v.co.x, v.co.y) for v in mesh.vertices]
        if len(verts) < 3:
            self.report({'ERROR'}, "Polygon must have ≥ 3 vertices")
            return {'CANCELLED'}
        polygon = Polygon(verts)

        tags = _tags_from_preset(self.feature_preset, self.custom_tags)
        if not tags:
            self.report({'ERROR'}, "No tags specified")
            return {'CANCELLED'}

        gdf = features.features_from_polygon(polygon, tags)
        if gdf is None or len(gdf) == 0:
            self.report({'ERROR'}, "No features found")
            return {'CANCELLED'}

        osmnx_obj = _find_osmnx_object(context)
        name = f"poly_{self.feature_preset}"
        objects = _create_feature_mesh_from_gdf(gdf, name=name, osmnx_obj=osmnx_obj)
        if not objects:
            self.report({'ERROR'}, "Failed to create meshes")
            return {'CANCELLED'}

        for o in objects:
            o["feature_type"] = self.feature_preset
            o["crs"] = str(gdf.crs) if gdf.crs else "EPSG:4326"
            if osmnx_obj:
                o["osmnx_center_lat"] = osmnx_obj.get("osmnx_center_lat")
                o["osmnx_center_lon"] = osmnx_obj.get("osmnx_center_lon")
                o["osmnx_scale"] = osmnx_obj.get("osmnx_scale", 0.001)

        self.report({'INFO'}, f"Created {len(objects)} object(s) with {len(gdf)} features")
        return {'FINISHED'}


class SCIGRAPHS_OT_FeaturesFromXML(bpy.types.Operator):
    """Download OSM features from a local .osm XML file."""
    bl_idname = "scigraphs.osmnx_features_xml"
    bl_label = "Features from XML"
    bl_description = "Parse features from a local .osm XML file"
    bl_options = {'REGISTER', 'UNDO'}

    filepath: StringProperty(subtype='FILE_PATH')

    feature_preset: EnumProperty(
        name="Feature Preset",
        items=[(k, k.replace('_', ' ').title(), "") for k in FEATURE_TAG_PRESETS]
            + [('CUSTOM', "Custom Tags", "Custom key=value,...")],
        default='BUILDING',
    )

    custom_tags: StringProperty(name="Custom Tags", default="")

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        from ....core.osmnx import features
        import os
        fp = bpy.path.abspath(self.filepath)
        if not fp or not os.path.exists(fp):
            self.report({'ERROR'}, f"File not found: {fp}")
            return {'CANCELLED'}

        tags = _tags_from_preset(self.feature_preset, self.custom_tags)
        gdf = features.features_from_xml(fp, tags)
        if gdf is None or len(gdf) == 0:
            self.report({'ERROR'}, "No features found in XML")
            return {'CANCELLED'}

        osmnx_obj = _find_osmnx_object(context)
        base = os.path.splitext(os.path.basename(fp))[0]
        objects = _create_feature_mesh_from_gdf(
            gdf, name=f"{base}_{self.feature_preset}", osmnx_obj=osmnx_obj,
        )
        if not objects:
            self.report({'ERROR'}, "Failed to create meshes")
            return {'CANCELLED'}

        for o in objects:
            o["feature_type"] = self.feature_preset
            o["source_xml"] = fp
            o["crs"] = str(gdf.crs) if gdf.crs else "EPSG:4326"
            if osmnx_obj:
                o["osmnx_center_lat"] = osmnx_obj.get("osmnx_center_lat")
                o["osmnx_center_lon"] = osmnx_obj.get("osmnx_center_lon")
                o["osmnx_scale"] = osmnx_obj.get("osmnx_scale", 0.001)

        self.report({'INFO'}, f"Loaded {len(gdf)} features from {base}")
        return {'FINISHED'}


class SCIGRAPHS_OT_SnapPOIsToNearestNodes(bpy.types.Operator):
    """Snap every POI point in a feature mesh to the nearest OSMnx graph node."""
    bl_idname = "scigraphs.osmnx_snap_pois"
    bl_label = "Snap POIs to Nearest Nodes"
    bl_description = "Attach each POI to its nearest graph node (attribute / move / connector)"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        if not obj or obj.type != 'MESH':
            return False
        return True

    def execute(self, context):
        props = context.scene.scigraphs
        obj = context.active_object

        osmnx_obj = _find_osmnx_object(context)
        if osmnx_obj is None:
            self.report({'ERROR'}, "No OSMnx street network in scene")
            return {'CANCELLED'}

        from ....core.osmnx.graph_cache import get_osmnx_graph, get_unprojected_graph
        G = get_osmnx_graph(osmnx_obj) or get_unprojected_graph(osmnx_obj)
        if G is None:
            self.report({'ERROR'}, "Street-network graph not found in memory")
            return {'CANCELLED'}

        from ....core.osmnx.spatial_queries import find_nearest_node
        from ....core.osmnx.metadata import get_graph_extent
        import math

        extent = get_graph_extent(G) or {"center_lat": 0.0, "center_lon": 0.0}
        center_lat = extent["center_lat"]
        center_lon = extent["center_lon"]
        scale = osmnx_obj.get("osmnx_scale", 0.001)

        EARTH_RADIUS = 6371000.0
        cos_lat = math.cos(math.radians(center_lat))
        mpd = math.pi / 180.0 * EARTH_RADIUS

        def blender_to_lonlat(x, y):
            lat = center_lat + (y / scale / mpd)
            lon = center_lon + (x / scale / (mpd * cos_lat))
            return lon, lat

        mode = props.osmnx_poi_snap_mode
        mesh = obj.data

        # Store nearest_node_id as POINT int attribute.
        attr_name = "nearest_node_id"
        if attr_name in mesh.attributes:
            mesh.attributes.remove(mesh.attributes[attr_name])
        nattr = mesh.attributes.new(name=attr_name, type='INT', domain='POINT')

        snapped = 0
        node_positions = {}  # node_id -> (x, y, z)
        nodes_str = osmnx_obj.get("nodes_data", "")
        if nodes_str and osmnx_obj.type == 'MESH':
            node_ids = nodes_str.split(",")
            om = osmnx_obj.data
            for i, nid in enumerate(node_ids[: len(om.vertices)]):
                v = om.vertices[i].co
                try:
                    node_positions[int(nid)] = (v.x, v.y, v.z)
                except ValueError:
                    continue

        connectors = []
        for i, v in enumerate(mesh.vertices):
            lon, lat = blender_to_lonlat(v.co.x, v.co.y)
            nid = find_nearest_node(G, lon, lat, is_projected=False)
            if nid is None:
                nattr.data[i].value = -1
                continue
            try:
                nattr.data[i].value = int(nid)
            except (TypeError, ValueError):
                nattr.data[i].value = -1

            if mode == 'MOVE_TO_NODE' and int(nid) in node_positions:
                nx, ny, nz = node_positions[int(nid)]
                v.co.x, v.co.y, v.co.z = nx, ny, nz

            if mode == 'ADD_CONNECTOR' and int(nid) in node_positions:
                connectors.append(((v.co.x, v.co.y, v.co.z), node_positions[int(nid)]))

            snapped += 1

        # Optional connector mesh.
        if mode == 'ADD_CONNECTOR' and connectors:
            cmesh = bpy.data.meshes.new(f"{obj.name}_connectors")
            verts = []
            edges = []
            for a, b in connectors:
                i0 = len(verts)
                verts.extend([a, b])
                edges.append((i0, i0 + 1))
            cmesh.from_pydata(verts, edges, [])
            cmesh.update()
            cobj = bpy.data.objects.new(f"{obj.name}_connectors", cmesh)
            context.scene.collection.objects.link(cobj)

        self.report({'INFO'}, f"Snapped {snapped} POIs to graph nodes ({mode})")
        return {'FINISHED'}


classes = [
    SCIGRAPHS_OT_FeaturesFromPlace,
    SCIGRAPHS_OT_FeaturesFromPoint,
    SCIGRAPHS_OT_FeaturesFromBBox,
    SCIGRAPHS_OT_FeaturesFromAddress,
    SCIGRAPHS_OT_FeaturesFromPolygon,
    SCIGRAPHS_OT_FeaturesFromXML,
    SCIGRAPHS_OT_SnapPOIsToNearestNodes,
]


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)

