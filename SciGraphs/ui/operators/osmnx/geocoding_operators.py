import bpy
from bpy.props import StringProperty, FloatProperty, IntProperty


class SCIGRAPHS_OT_GeocodeToGDF(bpy.types.Operator):
    """Geocode to GeoDataFrame."""
    bl_idname = "scigraphs.osmnx_geocode_to_gdf"
    bl_label = "Geocode to GeoDataFrame"
    bl_description = "Geocode a query and get geometry as GeoDataFrame"
    bl_options = {'REGISTER', 'UNDO'}
    
    query: StringProperty(
        name="Query",
        description="Place name or address to geocode",
        default="",
    )
    
    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)
    
    def execute(self, context):
        from ....core.osmnx import geocoder
        
        if not self.query.strip():
            self.report({'ERROR'}, "Please enter a query")
            return {'CANCELLED'}
        
        gdf = geocoder.geocode_to_gdf(self.query)
        
        if gdf is None or len(gdf) == 0:
            self.report({'ERROR'}, "Geocoding failed or no results")
            return {'CANCELLED'}
        
        self.report({'INFO'}, f"Geocoded: {len(gdf)} result(s)")
        
        return {'FINISHED'}


class SCIGRAPHS_OT_BBoxFromPoint(bpy.types.Operator):
    """Create bounding box from point."""
    bl_idname = "scigraphs.osmnx_bbox_from_point"
    bl_label = "BBox from Point"
    bl_description = "Create a bounding box around a point"
    bl_options = {'REGISTER', 'UNDO'}
    
    latitude: FloatProperty(
        name="Latitude",
        default=0.0,
        min=-90.0,
        max=90.0,
        precision=6,
    )
    
    longitude: FloatProperty(
        name="Longitude",
        default=0.0,
        min=-180.0,
        max=180.0,
        precision=6,
    )
    
    distance: FloatProperty(
        name="Distance (m)",
        default=500.0,
        min=10.0,
        max=50000.0,
    )
    
    def invoke(self, context, event):
        props = context.scene.scigraphs
        self.latitude = props.osmnx_latitude
        self.longitude = props.osmnx_longitude
        return context.window_manager.invoke_props_dialog(self)
    
    def execute(self, context):
        from ....core.osmnx import utils_geo
        
        point = (self.latitude, self.longitude)
        bbox = utils_geo.bbox_from_point(point, dist=self.distance)
        
        if bbox is None:
            self.report({'ERROR'}, "Failed to create bounding box")
            return {'CANCELLED'}
        
        props = context.scene.scigraphs
        if isinstance(bbox, tuple) and len(bbox) >= 4:
            props.osmnx_bbox_north = bbox[0]
            props.osmnx_bbox_south = bbox[1]
            props.osmnx_bbox_east = bbox[2]
            props.osmnx_bbox_west = bbox[3]
        
        self.report({'INFO'}, f"BBox created: {self.distance}m around point")
        
        return {'FINISHED'}


class SCIGRAPHS_OT_BBoxToPoly(bpy.types.Operator):
    """Convert bounding box to polygon."""
    bl_idname = "scigraphs.osmnx_bbox_to_poly"
    bl_label = "BBox to Polygon"
    bl_description = "Convert current bounding box to a Shapely polygon"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        from ....core.osmnx import utils_geo
        
        props = context.scene.scigraphs
        bbox = (props.osmnx_bbox_north, props.osmnx_bbox_south,
                props.osmnx_bbox_east, props.osmnx_bbox_west)
        
        poly = utils_geo.bbox_to_poly(bbox)
        
        if poly is None:
            self.report({'ERROR'}, "Failed to create polygon")
            return {'CANCELLED'}
        
        self.report({'INFO'}, f"Polygon created from BBox (area: {poly.area:.6f})")
        
        return {'FINISHED'}


class SCIGRAPHS_OT_BufferGeometry(bpy.types.Operator):
    """Buffer selected geometry."""
    bl_idname = "scigraphs.osmnx_buffer_geometry"
    bl_label = "Buffer Geometry"
    bl_description = "Create buffer around selected mesh geometry"
    bl_options = {'REGISTER', 'UNDO'}
    
    distance: FloatProperty(
        name="Buffer Distance (m)",
        default=10.0,
        min=0.1,
        max=10000.0,
    )
    
    @classmethod
    def poll(cls, context):
        return context.active_object and context.active_object.type == 'MESH'
    
    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)
    
    def execute(self, context):
        from ....core.osmnx import utils_geo
        from shapely.geometry import Point
        
        obj = context.active_object
        mesh = obj.data
        
        if len(mesh.vertices) == 0:
            self.report({'ERROR'}, "Object has no vertices")
            return {'CANCELLED'}
        
        point = Point(mesh.vertices[0].co.x, mesh.vertices[0].co.y)
        
        buffered = utils_geo.buffer_geometry(point, self.distance)
        
        if buffered is None:
            self.report({'ERROR'}, "Failed to buffer geometry")
            return {'CANCELLED'}
        
        self.report({'INFO'}, f"Geometry buffered by {self.distance}m")
        
        return {'FINISHED'}


class SCIGRAPHS_OT_InterpolatePoints(bpy.types.Operator):
    """Generate evenly spaced interpolated points along each edge of the active mesh."""
    bl_idname = "scigraphs.osmnx_interpolate_points"
    bl_label = "Interpolate Points"
    bl_description = "Generate evenly spaced points along edges of the active mesh (creates a new point-mesh)"
    bl_options = {'REGISTER', 'UNDO'}

    spacing: FloatProperty(
        name="Spacing (m)",
        default=10.0,
        min=0.1,
        max=1000.0,
    )

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj is not None and obj.type == 'MESH' and len(obj.data.edges) > 0

    def invoke(self, context, event):
        props = context.scene.scigraphs
        self.spacing = props.osmnx_interpolate_distance
        return context.window_manager.invoke_props_dialog(self)

    def execute(self, context):
        obj = context.active_object
        mesh = obj.data
        scale = obj.get("osmnx_scale", 0.001)
        spacing_units = self.spacing * scale  # spacing given in meters, mesh in blender units

        try:
            from ....core.osmnx import utils_geo  # noqa: F401 - keep lazy import in case we switch
        except Exception:
            pass

        out_verts = []
        for edge in mesh.edges:
            v0 = mesh.vertices[edge.vertices[0]].co
            v1 = mesh.vertices[edge.vertices[1]].co
            dx = v1.x - v0.x
            dy = v1.y - v0.y
            dz = v1.z - v0.z
            length = (dx * dx + dy * dy + dz * dz) ** 0.5
            if length <= 0:
                continue
            n = max(2, int(length / max(spacing_units, 1e-9)) + 1)
            for i in range(n):
                t = i / (n - 1) if n > 1 else 0.0
                out_verts.append((v0.x + dx * t, v0.y + dy * t, v0.z + dz * t))

        if not out_verts:
            self.report({'WARNING'}, "No points generated")
            return {'CANCELLED'}

        m = bpy.data.meshes.new(f"{obj.name}_interp_points")
        m.from_pydata(out_verts, [], [])
        m.update()
        o = bpy.data.objects.new(f"{obj.name}_interp_points", m)
        context.scene.collection.objects.link(o)

        self.report({'INFO'}, f"Interpolated {len(out_verts)} points @ {self.spacing}m")
        return {'FINISHED'}


class SCIGRAPHS_OT_SamplePoints(bpy.types.Operator):
    """Sample N points uniformly along the edges of the active OSMnx network."""
    bl_idname = "scigraphs.osmnx_sample_points"
    bl_label = "Sample Points"
    bl_description = "Generate N random points along the graph edges and snap each to the nearest node"
    bl_options = {'REGISTER', 'UNDO'}

    num_points: IntProperty(
        name="Number of Points",
        default=20,
        min=1,
        max=10000,
    )

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj is not None and obj.get("is_osmnx", False)

    def invoke(self, context, event):
        props = context.scene.scigraphs
        self.num_points = props.osmnx_sample_n
        return context.window_manager.invoke_props_dialog(self)

    def execute(self, context):
        import random
        obj = context.active_object
        mesh = obj.data

        edges = list(mesh.edges)
        if not edges:
            self.report({'ERROR'}, "Active object has no edges")
            return {'CANCELLED'}

        rng = random.Random()
        out_verts = []
        for _ in range(self.num_points):
            edge = rng.choice(edges)
            v0 = mesh.vertices[edge.vertices[0]].co
            v1 = mesh.vertices[edge.vertices[1]].co
            t = rng.random()
            out_verts.append(
                (v0.x + (v1.x - v0.x) * t,
                 v0.y + (v1.y - v0.y) * t,
                 v0.z + (v1.z - v0.z) * t)
            )

        m = bpy.data.meshes.new(f"{obj.name}_samples")
        m.from_pydata(out_verts, [], [])
        m.update()
        o = bpy.data.objects.new(f"{obj.name}_samples", m)
        context.scene.collection.objects.link(o)
        o["osmnx_samples_source"] = obj.name

        self.report({'INFO'}, f"Sampled {self.num_points} points on {obj.name}")
        return {'FINISHED'}


classes = [
    SCIGRAPHS_OT_GeocodeToGDF,
    SCIGRAPHS_OT_BBoxFromPoint,
    SCIGRAPHS_OT_BBoxToPoly,
    SCIGRAPHS_OT_BufferGeometry,
    SCIGRAPHS_OT_InterpolatePoints,
    SCIGRAPHS_OT_SamplePoints,
]


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)

