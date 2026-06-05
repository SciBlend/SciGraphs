import bpy
import os
from bpy.props import StringProperty, IntProperty, FloatProperty, BoolProperty, EnumProperty
from ....core import osmnx_analysis
from .utils import (
    _get_osmnx_graph,
    _get_unprojected_graph,
    _store_osmnx_graph,
    _store_unprojected_graph,
    _transfer_edge_attribute_to_mesh,
)


class SCIGRAPHS_OT_AddElevationsRaster(bpy.types.Operator):
    """Add elevation to nodes from a local DEM raster file."""
    bl_idname = "scigraphs.osmnx_add_elevations_raster"
    bl_label = "Add Elevations from Raster"
    bl_description = "Sample node elevations from a local DEM file (GeoTIFF)"
    bl_options = {'REGISTER', 'UNDO'}
    
    filepath: StringProperty(
        name="DEM File",
        description="Path to DEM raster file (GeoTIFF, etc.)",
        subtype='FILE_PATH',
    )
    
    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj and obj.get("is_osmnx", False)
    
    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}
    
    def execute(self, context):
        obj = context.active_object
        G = _get_osmnx_graph(obj)
        
        if G is None:
            self.report({'ERROR'}, "No OSMnx graph found")
            return {'CANCELLED'}
        
        filepath = bpy.path.abspath(self.filepath)
        
        if not os.path.exists(filepath):
            self.report({'ERROR'}, f"File not found: {filepath}")
            return {'CANCELLED'}
        
        G = osmnx_analysis.add_node_elevations_raster(G, filepath)
        
        if G is None:
            self.report({'ERROR'}, "Failed to add elevations from raster")
            return {'CANCELLED'}
        
        _store_osmnx_graph(obj, G)
        obj["osmnx_has_elevation"] = True
        
        stats = osmnx_analysis.get_elevation_stats(G)
        if stats:
            obj["osmnx_elev_min"] = stats['min_elevation']
            obj["osmnx_elev_max"] = stats['max_elevation']
            obj["osmnx_elev_range"] = stats['elevation_range']
        
        self.report({'INFO'}, f"Elevation added from raster ({stats['elevation_range']:.1f}m range)")
        return {'FINISHED'}


class SCIGRAPHS_OT_AddElevationsAPI(bpy.types.Operator):
    """Add elevation to nodes from an online API (Open-Elevation)."""
    bl_idname = "scigraphs.osmnx_add_elevations_api"
    bl_label = "Add Elevations from API"
    bl_description = "Query node elevations from Open-Elevation API (free, no key required)"
    bl_options = {'REGISTER', 'UNDO'}
    
    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj and obj.get("is_osmnx", False)
    
    def execute(self, context):
        props = context.scene.scigraphs
        obj = context.active_object
        
        G = _get_unprojected_graph(obj)
        if G is None:
            G = _get_osmnx_graph(obj)
        
        if G is None:
            self.report({'ERROR'}, "No OSMnx graph found")
            return {'CANCELLED'}
        
        if osmnx_analysis.is_graph_projected(G):
            self.report({'WARNING'}, "Using projected graph. Results may be less accurate.")
        
        self.report({'INFO'}, "Querying elevation API (this may take a while)...")

        try:
            from ....preferences import get_preferences
            prefs = get_preferences()
            if prefs and prefs.osmnx_elevation_url_template.strip():
                import osmnx as ox
                ox.settings.elevation_url_template = prefs.osmnx_elevation_url_template.strip()
        except Exception:
            pass

        G = osmnx_analysis.add_node_elevations_google(G, api_key=None)
        
        if G is None:
            self.report({'ERROR'}, "Failed to add elevations from API")
            return {'CANCELLED'}
        
        main_G = _get_osmnx_graph(obj)
        if main_G is not None:
            for node in G.nodes():
                if 'elevation' in G.nodes[node] and node in main_G.nodes:
                    main_G.nodes[node]['elevation'] = G.nodes[node]['elevation']
            _store_osmnx_graph(obj, main_G)
        
        _store_unprojected_graph(obj, G)
        
        obj["osmnx_has_elevation"] = True
        
        stats = osmnx_analysis.get_elevation_stats(G)
        if stats:
            obj["osmnx_elev_min"] = stats['min_elevation']
            obj["osmnx_elev_max"] = stats['max_elevation']
            obj["osmnx_elev_range"] = stats['elevation_range']
            self.report({'INFO'}, f"Elevation added ({stats['elevation_range']:.1f}m range)")
        else:
            self.report({'INFO'}, "Elevation added from API")
        
        return {'FINISHED'}


class SCIGRAPHS_OT_AddEdgeGrades(bpy.types.Operator):
    """Calculate edge grades (slopes) from node elevations."""
    bl_idname = "scigraphs.osmnx_add_edge_grades"
    bl_label = "Calculate Edge Grades"
    bl_description = "Calculate slope/grade for each edge based on node elevations"
    bl_options = {'REGISTER', 'UNDO'}
    
    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj and obj.get("is_osmnx", False) and obj.get("osmnx_has_elevation", False)
    
    def execute(self, context):
        obj = context.active_object
        G = _get_osmnx_graph(obj)
        
        if G is None:
            self.report({'ERROR'}, "No OSMnx graph found")
            return {'CANCELLED'}
        
        G = osmnx_analysis.add_edge_grades(G, add_absolute=True)
        
        if G is None:
            self.report({'ERROR'}, "Failed to calculate edge grades")
            return {'CANCELLED'}
        
        _store_osmnx_graph(obj, G)
        obj["osmnx_has_grades"] = True
        
        _transfer_edge_attribute_to_mesh(obj, G, 'grade', 'edge_grade')
        edges_transferred = _transfer_edge_attribute_to_mesh(obj, G, 'grade_abs', 'edge_grade_abs')
        
        stats = osmnx_analysis.get_grade_stats(G)
        if stats:
            obj["osmnx_grade_mean_abs"] = stats.get('mean_grade_abs', 0)
            obj["osmnx_grade_max_abs"] = stats.get('max_grade_abs', 0)
            obj["osmnx_steep_pct"] = stats.get('steep_edge_pct', 0)
            
            msg = f"Grades calculated: {edges_transferred} edges. Mean: {stats['mean_grade_abs']*100:.1f}%, Max: {stats['max_grade_abs']*100:.1f}%"
            self.report({'INFO'}, msg)
        else:
            self.report({'INFO'}, f"Edge grades calculated: {edges_transferred} edges")
        
        return {'FINISHED'}


class SCIGRAPHS_OT_ApplyElevation3D(bpy.types.Operator):
    """Apply elevation to transform the network into 3D."""
    bl_idname = "scigraphs.osmnx_apply_elevation_3d"
    bl_label = "Apply Elevation to 3D"
    bl_description = "Transform the flat network into 3D using node elevations"
    bl_options = {'REGISTER', 'UNDO'}
    
    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj and obj.get("is_osmnx", False) and obj.get("osmnx_has_elevation", False)
    
    def execute(self, context):
        props = context.scene.scigraphs
        obj = context.active_object

        mesh = obj.data
        # Either custom property may hold the scale depending on which
        # path created the object (see comment in
        # ``apply_dem_elevations_to_graph``).
        scale = obj.get("osmnx_scale")
        if scale is None:
            scale = obj.get("scale", 0.001)
        elev_scale = props.osmnx_elevation_scale
        elev_offset = props.osmnx_elevation_offset

        # The "elevation" mesh attribute is the single source of truth:
        # it is written by ``apply_dem_elevations_to_graph`` for every
        # vertex (intersections + curve points), so we can re-apply the
        # vertical scale/offset without depending on the cached graph or
        # on the BFS-based curve-point heuristic that used to leave
        # spurious vertices at ``(min+max)/2`` and produce vertical
        # spikes.
        attr = mesh.attributes.get("elevation")
        if attr is None:
            self.report(
                {'ERROR'},
                "Mesh has no per-vertex 'elevation' attribute. "
                "Run 'Get Elevation Data' first.",
            )
            return {'CANCELLED'}

        n_verts = len(mesh.vertices)
        elevations = [0.0] * n_verts
        attr.data.foreach_get("value", elevations)
        if not elevations:
            self.report({'ERROR'}, "Empty elevation attribute")
            return {'CANCELLED'}

        min_elev = float(min(elevations))
        max_elev = float(max(elevations))

        for vert_idx, vert in enumerate(mesh.vertices):
            elev = float(elevations[vert_idx])
            z = ((elev - min_elev) + elev_offset) * scale * elev_scale
            vert.co.z = z

        mesh.update()

        obj["osmnx_3d_applied"] = True
        obj["osmnx_elev_scale_used"] = elev_scale
        
        elev_range = max_elev - min_elev
        total_verts = len(mesh.vertices)
        self.report({'INFO'}, f"Applied 3D elevation + attribute to {total_verts} vertices (range: {elev_range:.1f}m)")
        return {'FINISHED'}


class SCIGRAPHS_OT_FlattenNetwork(bpy.types.Operator):
    """Reset network to flat (Z=0)."""
    bl_idname = "scigraphs.osmnx_flatten_network"
    bl_label = "Flatten to 2D"
    bl_description = "Reset the network to flat 2D (Z=0 for all vertices)"
    bl_options = {'REGISTER', 'UNDO'}
    
    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj and obj.get("is_osmnx", False)
    
    def execute(self, context):
        obj = context.active_object
        mesh = obj.data
        
        for vert in mesh.vertices:
            vert.co.z = 0.0
        
        mesh.update()
        obj["osmnx_3d_applied"] = False
        
        self.report({'INFO'}, "Network flattened to 2D")
        return {'FINISHED'}


class SCIGRAPHS_OT_ImportTerrain(bpy.types.Operator):
    """Import DEM from file and apply elevation to graph."""
    bl_idname = "scigraphs.osmnx_import_terrain"
    bl_label = "Import DEM from File"
    bl_description = "Load DEM file, apply elevation to network, and optionally show terrain"
    bl_options = {'REGISTER', 'UNDO'}
    
    filepath: StringProperty(
        name="DEM File",
        description="Path to DEM raster file (GeoTIFF)",
        subtype='FILE_PATH',
    )
    
    subsample: IntProperty(
        name="Subsample",
        description="Subsample factor for terrain mesh (1=full, 2=half, etc.)",
        default=1,
        min=1,
        max=10,
    )
    
    show_terrain: BoolProperty(
        name="Show Terrain Mesh",
        description="Create visible terrain mesh below the network",
        default=True,
    )
    
    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj and obj.get("is_osmnx", False)
    
    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}
    
    def execute(self, context):
        from ....core import terrain
        
        props = context.scene.scigraphs
        obj = context.active_object
        
        filepath = bpy.path.abspath(self.filepath)
        
        if not os.path.exists(filepath):
            self.report({'ERROR'}, f"File not found: {filepath}")
            return {'CANCELLED'}
        
        terrain_obj, success = terrain.import_dem_unified(
            obj,
            dem_source=filepath,
            source_type='file',
            vertical_scale=props.osmnx_elevation_scale,
            vertical_offset=props.osmnx_elevation_offset,
            show_terrain=self.show_terrain,
            subsample=self.subsample,
            padding=0.1
        )
        
        if not success:
            self.report({'ERROR'}, "Failed to import DEM. Check file and rasterio installation.")
            return {'CANCELLED'}
        
        msg = f"DEM applied to network"
        if terrain_obj:
            msg += f" + terrain ({len(terrain_obj.data.vertices)} vertices)"
        
        self.report({'INFO'}, msg)
        return {'FINISHED'}


class SCIGRAPHS_OT_ImportTerrainAPI(bpy.types.Operator):
    """Import DEM from API and apply elevation to graph."""
    bl_idname = "scigraphs.osmnx_import_terrain_api"
    bl_label = "Import DEM from API"
    bl_description = "Fetch elevation from API, apply to network, and optionally show terrain"
    bl_options = {'REGISTER', 'UNDO'}
    
    resolution: IntProperty(
        name="Resolution",
        description="Grid resolution (points per side). Higher = more detail but slower",
        default=50,
        min=10,
        max=300,
    )
    
    api: EnumProperty(
        name="API",
        description="Elevation data source",
        items=[
            ('open-elevation', "Open-Elevation", "Free API, global coverage"),
            ('opentopodata', "OpenTopoData (SRTM)", "Free API, SRTM 30m data"),
        ],
        default='open-elevation',
    )
    
    parallel_workers: IntProperty(
        name="Parallel Workers",
        description="Number of parallel API requests. Higher = faster but may cause rate limiting",
        default=5,
        min=1,
        max=50,
    )
    
    show_terrain: BoolProperty(
        name="Show Terrain Mesh",
        description="Create visible terrain mesh below the network",
        default=True,
    )
    
    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj and obj.get("is_osmnx", False)
    
    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self, width=350)
    
    def draw(self, context):
        layout = self.layout
        
        layout.prop(self, "resolution")
        layout.prop(self, "api")
        
        layout.separator()
        layout.label(text="Performance", icon='TIME')
        layout.prop(self, "parallel_workers")
        
        help_box = layout.box()
        help_box.scale_y = 0.8
        col = help_box.column(align=True)
        col.label(text="Safe: 3-5 workers", icon='CHECKMARK')
        col.label(text="Fast: 10-20 workers", icon='FORWARD')
        col.label(text="Turbo: 30-50 workers (may fail)", icon='ERROR')
        
        layout.separator()
        layout.prop(self, "show_terrain")
        
        layout.separator()
        
        total_points = self.resolution ** 2
        batches = (total_points + 99) // 100
        
        effective_parallelism = min(self.parallel_workers, batches)
        time_per_batch = 0.6 if self.api == 'open-elevation' else 1.2
        time_est = (batches / effective_parallelism) * time_per_batch
        
        box = layout.box()
        box.label(text=f"Grid: {self.resolution}x{self.resolution} = {total_points:,} points")
        box.label(text=f"Batches: {batches} ({self.parallel_workers} parallel)")
        
        if time_est < 60:
            box.label(text=f"Estimated time: ~{time_est:.0f} seconds")
        else:
            box.label(text=f"Estimated time: ~{time_est/60:.1f} minutes")
    
    def execute(self, context):
        from ....core import terrain
        
        props = context.scene.scigraphs
        obj = context.active_object
        
        self.report({'INFO'}, f"Fetching DEM from {self.api} ({self.parallel_workers} workers)...")

        try:
            terrain_obj, success = terrain.import_dem_unified(
                obj,
                dem_source=None,
                source_type='api',
                resolution=self.resolution,
                vertical_scale=props.osmnx_elevation_scale,
                vertical_offset=props.osmnx_elevation_offset,
                show_terrain=self.show_terrain,
                api=self.api,
                padding=0.1,
                max_workers=self.parallel_workers
            )
        except KeyboardInterrupt:
            self.report({'WARNING'}, "DEM fetch interrupted by user")
            return {'CANCELLED'}
        except Exception as exc:  # noqa: BLE001
            self.report({'ERROR'}, f"DEM fetch failed: {exc}")
            return {'CANCELLED'}

        if not success:
            self.report({'ERROR'}, "Failed to fetch DEM from API. Check internet connection.")
            return {'CANCELLED'}
        
        msg = f"DEM applied to network"
        if terrain_obj:
            msg += f" + terrain ({len(terrain_obj.data.vertices)} vertices)"
        
        self.report({'INFO'}, msg)
        return {'FINISHED'}


class SCIGRAPHS_OT_ToggleTerrain(bpy.types.Operator):
    """Toggle terrain visibility."""
    bl_idname = "scigraphs.osmnx_toggle_terrain"
    bl_label = "Toggle Terrain"
    bl_description = "Show or hide the terrain mesh"
    bl_options = {'REGISTER', 'UNDO'}
    
    @classmethod
    def poll(cls, context):
        obj = context.active_object
        if not obj or not obj.get("is_osmnx", False):
            return False
        terrain_name = obj.get("terrain_child", "")
        return terrain_name and terrain_name in bpy.data.objects
    
    def execute(self, context):
        from ....core import terrain
        
        obj = context.active_object
        terrain_name = obj.get("terrain_child", "")
        
        if terrain_name and terrain_name in bpy.data.objects:
            terrain_obj = bpy.data.objects[terrain_name]
            new_state = terrain_obj.hide_viewport
            terrain.toggle_terrain_visibility(obj, new_state)
            
            state_text = "visible" if new_state else "hidden"
            self.report({'INFO'}, f"Terrain {state_text}")
        
        return {'FINISHED'}


class SCIGRAPHS_OT_ExportAOI(bpy.types.Operator):
    """Export Area of Interest for Copernicus DEM download."""
    bl_idname = "scigraphs.osmnx_export_aoi"
    bl_label = "Export AOI for Copernicus"
    bl_description = "Export the network area as GeoJSON/KML/WKT for downloading DEM from Copernicus"
    bl_options = {'REGISTER'}
    
    filepath: StringProperty(
        name="File Path",
        description="Output file path",
        subtype='FILE_PATH',
    )
    
    format: EnumProperty(
        name="Format",
        description="Export format",
        items=[
            ('geojson', "GeoJSON", "GeoJSON format (recommended for Copernicus)"),
            ('kml', "KML", "KML format (Google Earth compatible)"),
            ('wkt', "WKT", "Well-Known Text format"),
        ],
        default='geojson',
    )
    
    padding: FloatProperty(
        name="Padding",
        description="Extra padding around network bounds (fraction, 0.1 = 10%)",
        default=0.1,
        min=0.0,
        max=0.5,
    )
    
    filter_glob: StringProperty(
        default="*.geojson;*.kml;*.wkt",
        options={'HIDDEN'},
    )
    
    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj and obj.get("is_osmnx", False)
    
    def invoke(self, context, event):
        obj = context.active_object
        place = obj.get("osmnx_place", "area")
        place_clean = place.replace(",", "").replace(" ", "_")[:30] if place else "area"
        
        ext = self.format
        self.filepath = f"{place_clean}_aoi.{ext}"
        
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}
    
    def draw(self, context):
        layout = self.layout
        layout.prop(self, "format")
        layout.prop(self, "padding")
        
        obj = context.active_object
        if obj:
            from ....core import terrain
            bounds = terrain.get_osmnx_bounds(obj, padding=self.padding)
            
            if bounds:
                import numpy as np
                lat_mid = (bounds['north'] + bounds['south']) / 2
                lon_km = 111.0 * np.cos(np.radians(lat_mid))
                lat_km = 111.0
                
                width_km = (bounds['east'] - bounds['west']) * lon_km
                height_km = (bounds['north'] - bounds['south']) * lat_km
                area_km2 = width_km * height_km
                
                box = layout.box()
                box.label(text="Area Preview:", icon='WORLD')
                col = box.column(align=True)
                col.label(text=f"N: {bounds['north']:.5f}")
                col.label(text=f"S: {bounds['south']:.5f}")
                col.label(text=f"E: {bounds['east']:.5f}")
                col.label(text=f"W: {bounds['west']:.5f}")
                col.separator()
                col.label(text=f"Size: {width_km:.1f} x {height_km:.1f} km")
                col.label(text=f"Area: {area_km2:.1f} km2")
    
    def execute(self, context):
        from ....core import terrain
        
        obj = context.active_object
        
        ext_map = {'geojson': '.geojson', 'kml': '.kml', 'wkt': '.wkt'}
        expected_ext = ext_map[self.format]
        
        filepath = self.filepath
        if not filepath.lower().endswith(expected_ext):
            filepath = filepath.rsplit('.', 1)[0] + expected_ext
        
        success, bounds_info = terrain.export_aoi_for_copernicus(
            obj,
            filepath,
            format=self.format,
            padding=self.padding
        )
        
        if not success:
            self.report({'ERROR'}, "Failed to export AOI. Check if graph is loaded.")
            return {'CANCELLED'}
        
        self.report({'INFO'}, 
            f"AOI exported: {bounds_info['width_km']:.1f} x {bounds_info['height_km']:.1f} km "
            f"({bounds_info['area_km2']:.1f} km2)"
        )
        return {'FINISHED'}


class SCIGRAPHS_OT_ImportTerrainPlane(bpy.types.Operator):
    """Import terrain plane from raster file."""
    bl_idname = "scigraphs.osmnx_import_terrain_plane"
    bl_label = "Import Terrain Plane"
    bl_description = "Import a georeferenced image as terrain plane (KMZ, GeoTIFF, PNG, JPG)"
    bl_options = {'REGISTER', 'UNDO'}
    
    filepath: StringProperty(
        name="File Path",
        description="Path to raster file",
        subtype='FILE_PATH',
    )
    
    source_crs: EnumProperty(
        name="Source CRS",
        description="Coordinate Reference System of the file",
        items=[
            ('EPSG:4326', "WGS 84 (EPSG:4326)", "GPS coordinates - lat/lon in degrees"),
            ('EPSG:3857', "Web Mercator (EPSG:3857)", "Google Maps, OpenStreetMap"),
            ('EPSG:32630', "UTM Zone 30N (EPSG:32630)", "Spain, UK, Portugal"),
        ],
        default='EPSG:4326',
    )
    
    filter_glob: StringProperty(
        default="*.kmz;*.kml;*.tif;*.tiff;*.geotiff;*.png;*.jpg;*.jpeg",
        options={'HIDDEN'},
    )
    
    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj and obj.get("is_osmnx", False)
    
    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}
    
    def draw(self, context):
        layout = self.layout
        
        layout.prop(self, "source_crs")
        
        box = layout.box()
        box.label(text="Supported formats:", icon='INFO')
        col = box.column(align=True)
        col.scale_y = 0.8
        col.label(text="KMZ - Google Earth overlay")
        col.label(text="GeoTIFF - 8/16/32 bit raster")
        col.label(text="PNG/JPG - with OSMnx alignment")
    
    def execute(self, context):
        from ....core import terrain
        
        obj = context.active_object
        
        terrain_obj, metadata = terrain.import_terrain_plane(
            self.filepath,
            source_crs=self.source_crs,
            target_crs='EPSG:4326',
            osmnx_obj=obj,
            name="Terrain_Plane"
        )
        
        if terrain_obj is None:
            self.report({'ERROR'}, "Failed to import terrain plane")
            return {'CANCELLED'}
        
        self.report({'INFO'}, 
            f"Terrain plane imported: {metadata['width_m']:.0f}m x {metadata['height_m']:.0f}m"
        )
        return {'FINISHED'}


class SCIGRAPHS_OT_UpdateTerrainPlaneOffset(bpy.types.Operator):
    """Update terrain plane position and scale."""
    bl_idname = "scigraphs.osmnx_update_terrain_plane"
    bl_label = "Update Terrain Plane"
    bl_description = "Update position and scale of terrain plane"
    bl_options = {'REGISTER', 'UNDO'}
    
    @classmethod
    def poll(cls, context):
        obj = context.active_object
        if obj and obj.get("is_osmnx", False):
            plane_name = obj.get("terrain_plane_child", "")
            return plane_name and plane_name in bpy.data.objects
        return False
    
    def execute(self, context):
        from ....core import terrain
        
        props = context.scene.scigraphs
        obj = context.active_object
        
        plane_name = obj.get("terrain_plane_child", "")
        if not plane_name or plane_name not in bpy.data.objects:
            self.report({'ERROR'}, "No terrain plane found")
            return {'CANCELLED'}
        
        plane_obj = bpy.data.objects[plane_name]
        
        terrain.update_terrain_plane_offset(
            plane_obj,
            offset_x=props.terrain_offset_x,
            offset_y=props.terrain_offset_y,
            offset_z=props.terrain_offset_z,
            scale_xy=props.terrain_scale_xy
        )
        
        terrain.update_terrain_plane_opacity(plane_obj, props.terrain_opacity)
        
        self.report({'INFO'}, "Terrain plane updated")
        return {'FINISHED'}


class SCIGRAPHS_OT_RemoveTerrainPlane(bpy.types.Operator):
    """Remove terrain plane."""
    bl_idname = "scigraphs.osmnx_remove_terrain_plane"
    bl_label = "Remove Terrain Plane"
    bl_description = "Remove the terrain plane"
    bl_options = {'REGISTER', 'UNDO'}
    
    @classmethod
    def poll(cls, context):
        obj = context.active_object
        if obj and obj.get("is_osmnx", False):
            plane_name = obj.get("terrain_plane_child", "")
            return plane_name and plane_name in bpy.data.objects
        return False
    
    def execute(self, context):
        obj = context.active_object
        plane_name = obj.get("terrain_plane_child", "")
        
        if plane_name and plane_name in bpy.data.objects:
            plane_obj = bpy.data.objects[plane_name]
            
            if plane_obj.data.materials:
                mat = plane_obj.data.materials[0]
                for node in mat.node_tree.nodes:
                    if node.type == 'TEX_IMAGE' and node.image:
                        bpy.data.images.remove(node.image)
                bpy.data.materials.remove(mat)
            
            mesh = plane_obj.data
            bpy.data.objects.remove(plane_obj, do_unlink=True)
            if mesh.users == 0:
                bpy.data.meshes.remove(mesh)
            
            del obj["terrain_plane_child"]
        
        self.report({'INFO'}, "Terrain plane removed")
        return {'FINISHED'}


class SCIGRAPHS_OT_ImportDEMDisplace(bpy.types.Operator):
    """Import DEM using Displace modifier method (fast)."""
    bl_idname = "scigraphs.import_dem_displace"
    bl_label = "Import DEM (Displace)"
    bl_description = "Import GeoTIFF DEM using subdivision and displace modifier (fast method)"
    bl_options = {'REGISTER', 'UNDO'}
    
    filepath: StringProperty(
        name="File Path",
        description="Path to GeoTIFF DEM file",
        subtype='FILE_PATH',
    )
    
    subdivision_levels: IntProperty(
        name="Subdivision Levels",
        description="Number of subdivision levels (higher = more detail, slower)",
        default=6,
        min=1,
        max=10,
    )
    
    scale: FloatProperty(
        name="Scale",
        description="Scale factor (0.001 = km to Blender units)",
        default=0.001,
        min=0.0001,
        max=1.0,
        precision=4,
    )
    
    apply_material: BoolProperty(
        name="Apply Material",
        description="Apply elevation-based material",
        default=True,
    )
    
    filter_glob: StringProperty(
        default="*.tif;*.tiff;*.geotiff",
        options={'HIDDEN'},
    )
    
    @classmethod
    def poll(cls, context):
        return True
    
    def invoke(self, context, event):
        from ....preferences import get_preferences
        prefs = get_preferences()
        if prefs:
            self.subdivision_levels = prefs.default_subdivision_levels
            self.apply_material = prefs.auto_apply_material
        
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}
    
    def draw(self, context):
        layout = self.layout
        layout.prop(self, "subdivision_levels")
        layout.prop(self, "scale")
        layout.prop(self, "apply_material")
        
        box = layout.box()
        box.scale_y = 0.8
        box.label(text="Displace Method:", icon='INFO')
        box.label(text="Fast, uses Blender modifiers")
        box.label(text="Good for visualization")
    
    def execute(self, context):
        from ....core.geo.georaster import load_georaster
        from ....core.geo.dem_processor import (
            create_raster_extent_mesh, 
            apply_displace_modifier,
            apply_elevation_material
        )
        
        georaster = load_georaster(self.filepath)
        if georaster is None:
            self.report({'ERROR'}, "Failed to load GeoTIFF")
            return {'CANCELLED'}
        
        obj = create_raster_extent_mesh(georaster, scale=self.scale, name="DEM_Displace")
        if obj is None:
            self.report({'ERROR'}, "Failed to create terrain mesh")
            return {'CANCELLED'}
        
        success = apply_displace_modifier(
            obj, georaster, 
            subdivision_levels=self.subdivision_levels,
            scale=self.scale
        )
        
        if not success:
            self.report({'ERROR'}, "Failed to apply displace modifier")
            return {'CANCELLED'}
        
        if self.apply_material:
            stats = georaster.get_statistics()
            if stats:
                obj["dem_elev_min"] = stats['min']
                obj["dem_elev_max"] = stats['max']
            apply_elevation_material(obj, style='ELEVATION')
        
        osmnx_obj = context.active_object
        if osmnx_obj and osmnx_obj.get("is_osmnx", False):
            obj.location = osmnx_obj.location.copy()
            obj.location.z -= 0.01
            obj["osmnx_parent"] = osmnx_obj.name
            osmnx_obj["dem_terrain_child"] = obj.name
        
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        context.view_layer.objects.active = obj
        
        self.report({'INFO'}, f"DEM imported (Displace method, {self.subdivision_levels} subdivisions)")
        return {'FINISHED'}


class SCIGRAPHS_OT_ImportDEMRawMesh(bpy.types.Operator):
    """Import DEM as raw mesh (accurate but slower)."""
    bl_idname = "scigraphs.import_dem_raw_mesh"
    bl_label = "Import DEM (Raw Mesh)"
    bl_description = "Import GeoTIFF DEM creating vertices from pixels (accurate method)"
    bl_options = {'REGISTER', 'UNDO'}
    
    filepath: StringProperty(
        name="File Path",
        description="Path to GeoTIFF DEM file",
        subtype='FILE_PATH',
    )
    
    subsample: IntProperty(
        name="Subsample Factor",
        description="Reduce resolution (1 = full, 2 = half, 4 = quarter)",
        default=1,
        min=1,
        max=16,
    )
    
    scale: FloatProperty(
        name="Scale",
        description="Scale factor (0.001 = km to Blender units)",
        default=0.001,
        min=0.0001,
        max=1.0,
        precision=4,
    )
    
    apply_material: BoolProperty(
        name="Apply Material",
        description="Apply elevation-based material",
        default=True,
    )
    
    filter_glob: StringProperty(
        default="*.tif;*.tiff;*.geotiff",
        options={'HIDDEN'},
    )
    
    @classmethod
    def poll(cls, context):
        return True
    
    def invoke(self, context, event):
        from ....preferences import get_preferences
        prefs = get_preferences()
        if prefs:
            self.apply_material = prefs.auto_apply_material
        
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}
    
    def draw(self, context):
        layout = self.layout
        layout.prop(self, "subsample")
        layout.prop(self, "scale")
        layout.prop(self, "apply_material")
        
        box = layout.box()
        box.scale_y = 0.8
        box.label(text="Raw Mesh Method:", icon='INFO')
        box.label(text="Accurate, one vertex per pixel")
        box.label(text="Slow for large DEMs")
        
        if self.subsample > 1:
            box.label(text=f"Resolution reduced to 1/{self.subsample}")
    
    def execute(self, context):
        from ....core.geo.georaster import load_georaster
        from ....core.geo.dem_processor import raster_to_mesh, apply_elevation_material
        
        georaster = load_georaster(self.filepath)
        if georaster is None:
            self.report({'ERROR'}, "Failed to load GeoTIFF")
            return {'CANCELLED'}
        
        obj = raster_to_mesh(
            georaster, 
            scale=self.scale, 
            subsample=self.subsample,
            name="DEM_RawMesh"
        )
        
        if obj is None:
            self.report({'ERROR'}, "Failed to create terrain mesh")
            return {'CANCELLED'}
        
        if self.apply_material:
            apply_elevation_material(obj, style='ELEVATION')
        
        osmnx_obj = context.active_object
        if osmnx_obj and osmnx_obj.get("is_osmnx", False):
            obj.location = osmnx_obj.location.copy()
            obj.location.z -= 0.01
            obj["osmnx_parent"] = osmnx_obj.name
            osmnx_obj["dem_terrain_child"] = obj.name
        
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        context.view_layer.objects.active = obj
        
        verts = len(obj.data.vertices)
        self.report({'INFO'}, f"DEM imported (Raw Mesh method, {verts} vertices)")
        return {'FINISHED'}


class SCIGRAPHS_OT_DownloadDEM(bpy.types.Operator):
    """Download DEM from OpenTopography."""
    bl_idname = "scigraphs.download_dem"
    bl_label = "Download DEM"
    bl_description = "Download elevation data from OpenTopography API"
    bl_options = {'REGISTER'}
    
    dataset: EnumProperty(
        name="Dataset",
        description="DEM dataset to download",
        items=[
            ('SRTMGL1', "SRTM GL1 (30m)", "NASA SRTM 1 arc-second"),
            ('SRTMGL3', "SRTM GL3 (90m)", "NASA SRTM 3 arc-second"),
            ('AW3D30', "ALOS World 3D (30m)", "JAXA ALOS"),
            ('NASADEM', "NASADEM (30m)", "NASA improved SRTM"),
            ('COP30', "Copernicus GLO-30", "EU Copernicus 30m"),
            ('COP90', "Copernicus GLO-90", "EU Copernicus 90m"),
        ],
        default='SRTMGL1',
    )
    
    import_method: EnumProperty(
        name="Import Method",
        description="How to create the terrain mesh",
        items=[
            ('DISPLACE', "Displace (Fast)", "Use subdivision + displace modifier"),
            ('RAW_MESH', "Raw Mesh (Accurate)", "Create mesh from raster pixels"),
        ],
        default='DISPLACE',
    )
    
    subdivision_levels: IntProperty(
        name="Subdivision Levels",
        description="For Displace method: subdivision levels",
        default=6,
        min=1,
        max=10,
    )
    
    subsample: IntProperty(
        name="Subsample Factor",
        description="For Raw Mesh method: reduce resolution",
        default=2,
        min=1,
        max=16,
    )
    
    padding: FloatProperty(
        name="Padding",
        description="Extra padding around network bounds (fraction)",
        default=0.1,
        min=0.0,
        max=0.5,
    )
    
    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj and obj.get("is_osmnx", False)
    
    def invoke(self, context, event):
        from ....preferences import get_preferences
        prefs = get_preferences()
        if prefs:
            self.dataset = prefs.opentopography_default_dataset
            if prefs.default_dem_method == 'DISPLACE':
                self.import_method = 'DISPLACE'
            else:
                self.import_method = 'RAW_MESH'
            self.subdivision_levels = prefs.default_subdivision_levels
        
        return context.window_manager.invoke_props_dialog(self, width=350)
    
    def draw(self, context):
        layout = self.layout
        
        from ....preferences import get_preferences, ADDON_PACKAGE
        prefs = get_preferences()
        has_key = prefs and prefs.opentopography_api_key
        
        if not has_key:
            box = layout.box()
            box.alert = True
            box.label(text="API Key not set!", icon='ERROR')
            box.operator("preferences.addon_show", text="Open Preferences").module = ADDON_PACKAGE
            return
        
        layout.prop(self, "dataset")
        layout.prop(self, "padding")
        
        layout.separator()
        layout.prop(self, "import_method")
        
        if self.import_method == 'DISPLACE':
            layout.prop(self, "subdivision_levels")
        else:
            layout.prop(self, "subsample")
        
        obj = context.active_object
        if obj:
            from ....core.geo.terrain import get_osmnx_bounds
            from ....core.geo.dem_download import estimate_download_size
            
            bounds = get_osmnx_bounds(obj, padding=self.padding)
            if bounds:
                size_mb = estimate_download_size(bounds, self.dataset)
                
                box = layout.box()
                box.scale_y = 0.8
                lat_range = bounds['north'] - bounds['south']
                lon_range = bounds['east'] - bounds['west']
                box.label(text=f"Area: {lat_range:.3f} x {lon_range:.3f} degrees")
                box.label(text=f"Estimated download: ~{size_mb:.1f} MB")
    
    def execute(self, context):
        from ....core.geo.terrain import get_osmnx_bounds
        from ....core.geo.dem_download import download_from_opentopography
        from ....core.geo.georaster import load_georaster
        from ....core.geo.dem_processor import (
            create_raster_extent_mesh,
            apply_displace_modifier,
            raster_to_mesh,
            apply_elevation_material,
            apply_georaster_elevations_to_graph
        )
        
        props = context.scene.scigraphs
        obj = context.active_object
        
        bounds = get_osmnx_bounds(obj, padding=self.padding)
        if bounds is None:
            self.report({'ERROR'}, "Could not get network bounds")
            return {'CANCELLED'}
        
        self.report({'INFO'}, f"Downloading {self.dataset} from OpenTopography...")
        
        dem_path = download_from_opentopography(bounds, dataset=self.dataset)
        
        if dem_path is None:
            self.report({'ERROR'}, "Download failed. Check API key and internet connection.")
            return {'CANCELLED'}
        
        georaster = load_georaster(dem_path)
        if georaster is None:
            self.report({'ERROR'}, "Failed to load downloaded DEM")
            return {'CANCELLED'}
        
        scale = obj.get("osmnx_scale", 0.001)
        vertical_scale = props.osmnx_elevation_scale
        vertical_offset = props.osmnx_elevation_offset
        
        self.report({'INFO'}, "Applying elevations to network...")
        
        graph_success = apply_georaster_elevations_to_graph(
            obj, georaster,
            vertical_scale=vertical_scale,
            vertical_offset=vertical_offset
        )
        
        if not graph_success:
            self.report({'WARNING'}, "Could not apply elevations to graph")
        
        if self.import_method == 'DISPLACE':
            terrain_obj = create_raster_extent_mesh(
                georaster, scale=scale, name="DEM_Terrain", osmnx_obj=obj
            )
            if terrain_obj:
                apply_displace_modifier(terrain_obj, georaster, 
                                       subdivision_levels=self.subdivision_levels,
                                       scale=scale * vertical_scale)
        else:
            terrain_obj = raster_to_mesh(
                georaster, 
                scale=scale, 
                subsample=self.subsample,
                name="DEM_Terrain",
                osmnx_obj=obj,
                vertical_scale=vertical_scale,
                vertical_offset=vertical_offset
            )
        
        if terrain_obj is None:
            self.report({'ERROR'}, "Failed to create terrain")
            return {'CANCELLED'}
        
        stats = georaster.get_statistics()
        if stats:
            terrain_obj["dem_elev_min"] = stats['min']
            terrain_obj["dem_elev_max"] = stats['max']
        apply_elevation_material(terrain_obj, style='ELEVATION')
        
        terrain_obj.location = obj.location.copy()
        terrain_obj["osmnx_parent"] = obj.name
        obj["dem_terrain_child"] = terrain_obj.name
        
        elev_range = stats['max'] - stats['min'] if stats else 0
        self.report({'INFO'}, f"DEM applied: terrain + network elevation ({elev_range:.0f}m range)")
        return {'FINISHED'}


class SCIGRAPHS_OT_UpdateTerrainScale(bpy.types.Operator):
    """Update terrain vertical scale."""
    bl_idname = "scigraphs.osmnx_update_terrain_scale"
    bl_label = "Update Terrain Scale"
    bl_description = "Update the vertical exaggeration of the terrain"
    bl_options = {'REGISTER', 'UNDO'}
    
    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj and obj.get("is_terrain", False)
    
    def execute(self, context):
        from ....core import terrain
        
        props = context.scene.scigraphs
        obj = context.active_object
        
        terrain.update_terrain_vertical_scale(obj, props.osmnx_elevation_scale)
        
        self.report({'INFO'}, f"Terrain scale updated to {props.osmnx_elevation_scale}")
        return {'FINISHED'}


class SCIGRAPHS_OT_RemoveTerrain(bpy.types.Operator):
    """Remove terrain object."""
    bl_idname = "scigraphs.osmnx_remove_terrain"
    bl_label = "Remove Terrain"
    bl_description = "Remove the terrain mesh"
    bl_options = {'REGISTER', 'UNDO'}
    
    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj and obj.get("is_terrain", False)
    
    def execute(self, context):
        from ....core import terrain
        
        obj = context.active_object
        terrain.remove_terrain(obj)
        
        self.report({'INFO'}, "Terrain removed")
        return {'FINISHED'}


class SCIGRAPHS_OT_TerrainMaterial(bpy.types.Operator):
    """Apply material to terrain."""
    bl_idname = "scigraphs.osmnx_terrain_material"
    bl_label = "Apply Terrain Material"
    bl_description = "Apply elevation-based material to terrain"
    bl_options = {'REGISTER', 'UNDO'}
    
    style: EnumProperty(
        name="Style",
        items=[
            ('ELEVATION', "Elevation Colors", "Color based on elevation"),
            ('SIMPLE', "Simple", "Single solid color"),
        ],
        default='ELEVATION',
    )
    
    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj and obj.get("is_terrain", False)
    
    def execute(self, context):
        from ....core import terrain
        
        obj = context.active_object
        mat = terrain.apply_terrain_material(obj, self.style)
        
        if mat:
            self.report({'INFO'}, f"Applied {self.style} material")
        else:
            self.report({'ERROR'}, "Failed to apply material")
            return {'CANCELLED'}
        
        return {'FINISHED'}


class SCIGRAPHS_OT_GetElevationData(bpy.types.Operator):
    """Unified entry point: pick a DEM source and decide whether to apply it
    to the graph nodes, build a terrain mesh, or both.

    Replaces the half-dozen overlapping buttons of the old panel. Internally
    delegates to the more specialised operators so existing pipelines and
    scripts that target them keep working.
    """

    bl_idname = "scigraphs.osmnx_get_elevation"
    bl_label = "Get Elevation Data"
    bl_description = (
        "Fetch a DEM from the chosen source and apply it to nodes and/or as "
        "a terrain mesh, in a single step"
    )
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj and obj.get("is_osmnx", False)

    def _safe_call(self, op_fn, **kwargs):
        """Call a sub-operator and convert any failure into a clean
        ``{'CANCELLED'}`` instead of letting RuntimeError / KeyboardInterrupt
        bubble all the way to Blender's red-popup handler.
        """
        try:
            res = op_fn('EXEC_DEFAULT', **kwargs)
        except KeyboardInterrupt:
            self.report({'WARNING'}, "Interrupted by user")
            return {'CANCELLED'}
        except RuntimeError as exc:
            # Sub-operator already reported the original cause; surface a
            # short message so the user sees something contextual.
            self.report({'ERROR'}, f"Sub-operation failed: {exc}")
            return {'CANCELLED'}
        except Exception as exc:  # noqa: BLE001
            self.report({'ERROR'}, f"Unexpected error: {exc}")
            return {'CANCELLED'}
        return res

    def execute(self, context):
        props = context.scene.scigraphs
        obj = context.active_object

        if not (props.osmnx_dem_apply_to_nodes or props.osmnx_dem_create_terrain):
            self.report(
                {'ERROR'},
                "Enable at least one of 'Apply to nodes' or 'Create terrain mesh'.",
            )
            return {'CANCELLED'}

        # Always start from a clean slate: every elevation pipeline
        # ultimately creates a *new* terrain object instead of editing
        # the previous one in place, so we have to remove the old one
        # ourselves to avoid stacking duplicates each time the user
        # re-runs "Get Elevation Data". Also clears any basemap material
        # that was draped onto the old terrain — it would point to a
        # disappeared mesh otherwise.
        for slot in ("dem_terrain_child", "terrain_child"):
            old_name = obj.get(slot)
            if old_name and old_name in bpy.data.objects:
                old = bpy.data.objects[old_name]
                # Remove its basemap material first if any.
                if old.get("basemap_image"):
                    mat_name = f"SciGraphs_Basemap_{old.name}"
                    mat = bpy.data.materials.get(mat_name)
                    if mat is not None:
                        try:
                            bpy.data.materials.remove(mat, do_unlink=True)
                        except RuntimeError:
                            pass
                bpy.data.objects.remove(old, do_unlink=True)
            obj.pop(slot, None)

        source = props.osmnx_dem_source

        # ---- Source 1: OpenTopography (high quality, requires API key) ----
        if source == 'OPENTOPOGRAPHY':
            from ....preferences import get_preferences
            prefs = get_preferences()
            if not (prefs and prefs.opentopography_api_key):
                self.report(
                    {'ERROR'},
                    "OpenTopography API key required. Set it in addon preferences.",
                )
                return {'CANCELLED'}

            method = 'DISPLACE' if props.osmnx_dem_terrain_method == 'DISPLACE' else 'RAW_MESH'
            res = self._safe_call(
                bpy.ops.scigraphs.download_dem,
                dataset=prefs.opentopography_default_dataset,
                import_method=method,
                subdivision_levels=props.osmnx_dem_subdivision_levels,
                subsample=props.osmnx_dem_subsample,
                padding=props.osmnx_dem_padding,
            )
            if 'CANCELLED' in res:
                return {'CANCELLED'}

            if not props.osmnx_dem_create_terrain:
                terrain_name = obj.get("dem_terrain_child") or obj.get("terrain_child")
                if terrain_name and terrain_name in bpy.data.objects:
                    bpy.data.objects.remove(bpy.data.objects[terrain_name], do_unlink=True)
                    obj.pop("dem_terrain_child", None)
                    obj.pop("terrain_child", None)
            self._normalize_terrain_slots(obj)
            self._restore_active(context, obj)
            return {'FINISHED'}

        # ---- Source 2: Open-Elevation API (free, no key) ----
        if source == 'OPEN_ELEVATION':
            # Cheap, unscientific guard against runaway requests: warn the
            # user before issuing thousands of API calls that will block the
            # main Blender thread for minutes.
            res_total = props.osmnx_dem_api_resolution ** 2
            if res_total > 10000:
                self.report(
                    {'WARNING'},
                    f"Open-Elevation will fetch {res_total:,} points; this may "
                    f"take several minutes and freeze the UI. Lower 'Resolution' "
                    f"if you want to abort.",
                )

            res = self._safe_call(
                bpy.ops.scigraphs.osmnx_import_terrain_api,
                resolution=props.osmnx_dem_api_resolution,
                api=props.osmnx_dem_api_provider,
                parallel_workers=props.osmnx_dem_api_workers,
                show_terrain=props.osmnx_dem_create_terrain,
            )
            if 'CANCELLED' in res:
                return {'CANCELLED'}

            if not props.osmnx_dem_apply_to_nodes:
                obj["osmnx_has_elevation"] = False
            self._normalize_terrain_slots(obj)
            self._restore_active(context, obj)
            return {'FINISHED'}

        # ---- Source 3: Local GeoTIFF ----
        if source == 'LOCAL_GEOTIFF':
            path = bpy.path.abspath(props.osmnx_dem_local_path or "")
            if not path or not os.path.exists(path):
                self.report({'ERROR'}, f"GeoTIFF not found: {path or '(empty path)'}")
                return {'CANCELLED'}

            apply_nodes = props.osmnx_dem_apply_to_nodes
            create_terrain = props.osmnx_dem_create_terrain

            if apply_nodes and create_terrain:
                res = self._safe_call(
                    bpy.ops.scigraphs.osmnx_import_terrain,
                    filepath=path,
                    subsample=props.osmnx_dem_subsample,
                    show_terrain=True,
                )
            elif apply_nodes and not create_terrain:
                res = self._safe_call(
                    bpy.ops.scigraphs.osmnx_add_elevations_raster,
                    filepath=path,
                )
            elif create_terrain and not apply_nodes:
                if props.osmnx_dem_terrain_method == 'DISPLACE':
                    res = self._safe_call(
                        bpy.ops.scigraphs.import_dem_displace,
                        filepath=path,
                        subdivision_levels=props.osmnx_dem_subdivision_levels,
                    )
                else:
                    res = self._safe_call(
                        bpy.ops.scigraphs.import_dem_raw_mesh,
                        filepath=path,
                        subsample=props.osmnx_dem_subsample,
                    )
            else:
                res = {'CANCELLED'}

            if 'FINISHED' not in res:
                return {'CANCELLED'}
            self._normalize_terrain_slots(obj)
            self._restore_active(context, obj)
            return {'FINISHED'}

        self.report({'ERROR'}, f"Unknown elevation source: {source}")
        return {'CANCELLED'}

    @staticmethod
    def _normalize_terrain_slots(osmnx_obj):
        """Make ``terrain_child`` and ``dem_terrain_child`` agree.

        Sub-operators historically set one slot or the other. Mirror them
        so downstream code (panel UI, basemap operator, etc.) finds the
        terrain regardless of which key it queries.
        """
        name = osmnx_obj.get("terrain_child") or osmnx_obj.get("dem_terrain_child")
        if name and name in bpy.data.objects:
            osmnx_obj["terrain_child"] = name
            osmnx_obj["dem_terrain_child"] = name
        else:
            osmnx_obj.pop("terrain_child", None)
            osmnx_obj.pop("dem_terrain_child", None)

    @staticmethod
    def _restore_active(context, osmnx_obj):
        """Sub-operators leave the terrain selected; restore the network
        as the active object so the user keeps interacting with the
        graph after a Get Elevation Data call."""
        try:
            for o in bpy.data.objects:
                o.select_set(False)
            osmnx_obj.select_set(True)
            context.view_layer.objects.active = osmnx_obj
        except Exception:  # noqa: BLE001
            pass


class SCIGRAPHS_OT_RemoveTerrainChild(bpy.types.Operator):
    """Remove the terrain mesh attached to an OSMnx network.

    The legacy ``osmnx_remove_terrain`` only worked on a terrain object
    selected directly (poll requires ``is_terrain=True``), so it could not
    be triggered from the network-side panel. This wrapper handles the
    common case "I'm on the OSMnx object, kill its terrain child".
    """

    bl_idname = "scigraphs.osmnx_remove_terrain_child"
    bl_label = "Remove Terrain"
    bl_description = "Delete the terrain mesh linked to the active OSMnx network"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        if not obj or not obj.get("is_osmnx", False):
            return False
        terrain_name = obj.get("terrain_child") or obj.get("dem_terrain_child")
        return bool(terrain_name and terrain_name in bpy.data.objects)

    def execute(self, context):
        obj = context.active_object
        terrain_name = obj.get("terrain_child") or obj.get("dem_terrain_child")
        if not terrain_name or terrain_name not in bpy.data.objects:
            self.report({'WARNING'}, "No terrain mesh linked to this network")
            return {'CANCELLED'}

        terrain_obj = bpy.data.objects[terrain_name]
        bpy.data.objects.remove(terrain_obj, do_unlink=True)
        obj.pop("terrain_child", None)
        obj.pop("dem_terrain_child", None)
        self.report({'INFO'}, f"Terrain '{terrain_name}' removed")
        return {'FINISHED'}


def _resolve_terrain_object(osmnx_obj):
    """Return the Blender terrain mesh attached to an OSMnx network, or None."""
    name = osmnx_obj.get("terrain_child") or osmnx_obj.get("dem_terrain_child")
    if not name:
        return None
    return bpy.data.objects.get(name)


def _terrain_bounds_wgs84(terrain_obj):
    """Return the WGS84 bbox stored on a SciGraphs terrain mesh, or None.

    Terrain meshes built by ``create_terrain_mesh`` (Displace and Raw Mesh
    paths) record the exact DEM bounding box in custom properties. These
    bounds are the right basis for both fetching imagery and projecting
    UVs, because the terrain XY extent corresponds *exactly* to that
    rectangle in WGS84 (the terrain uses the same equirectangular local
    projection as the rest of the addon).
    """
    keys = ('dem_bounds_north', 'dem_bounds_south', 'dem_bounds_east', 'dem_bounds_west')
    if not all(k in terrain_obj for k in keys):
        return None
    return {
        'north': float(terrain_obj['dem_bounds_north']),
        'south': float(terrain_obj['dem_bounds_south']),
        'east': float(terrain_obj['dem_bounds_east']),
        'west': float(terrain_obj['dem_bounds_west']),
    }


def _terrain_xy_to_latlon_factory(terrain_obj):
    """Build a callable ``(world_x, world_y) -> (lat, lon)`` for a terrain.

    The terrain mesh lives in the same equirectangular-local projection
    that :func:`SciGraphs.core.geo.terrain.create_terrain_mesh` used at
    construction time, with these reversible parameters stored on the
    object as custom properties:

    * ``dem_center_lat``, ``dem_center_lon`` — projection origin.
    * ``dem_scale`` — meters → Blender units factor.

    Inverting that projection per vertex is what makes the basemap drape
    follow the *real* geography rather than just the bbox of the mesh.
    The implementation is intentionally factored as a closure so a
    future terrain stored in UTM (or any other CRS) can plug a
    ``pyproj.Transformer`` here without touching the UV code.
    """
    import math as _math

    center_lat = terrain_obj.get("dem_center_lat")
    center_lon = terrain_obj.get("dem_center_lon")
    scale = terrain_obj.get("dem_scale")
    print(
        f"[SciGraphs] xy_to_latlon('{terrain_obj.name}'): "
        f"dem_center_lat={center_lat} dem_center_lon={center_lon} "
        f"dem_scale={scale}"
    )

    # Backwards compatibility: terrains created by older builds didn't
    # store the projection origin. Try to recover it from the parent
    # OSMnx network, then from the DEM bbox as a last resort. This
    # avoids forcing the user to re-import a perfectly valid terrain
    # just because a custom property is missing.
    if center_lat is None or center_lon is None or not scale:
        import bpy as _bpy
        parent_name = (
            terrain_obj.get("osmnx_parent")
            or terrain_obj.get("parent_osmnx")
            or terrain_obj.get("source_osmnx")
        )
        parent = _bpy.data.objects.get(parent_name) if parent_name else None
        if parent is None:
            # Fallback: any OSMnx network that points to this terrain.
            for o in _bpy.data.objects:
                if o.get("is_osmnx") and (
                    o.get("dem_terrain_child") == terrain_obj.name
                    or o.get("terrain_child") == terrain_obj.name
                ):
                    parent = o
                    break

        if parent is not None:
            if center_lat is None:
                center_lat = parent.get("osmnx_center_lat")
            if center_lon is None:
                center_lon = parent.get("osmnx_center_lon")
            if not scale:
                scale = parent.get("osmnx_scale") or parent.get("scale")

        # Last-resort: derive centre from the DEM bbox (assumes the
        # mesh is centred on the bbox midpoint, which is true for the
        # plane created by ``create_dem_plane`` when no OSMnx parent
        # exists).
        if (center_lat is None or center_lon is None) and all(
            k in terrain_obj
            for k in ("dem_bounds_north", "dem_bounds_south",
                      "dem_bounds_east", "dem_bounds_west")
        ):
            center_lat = (
                float(terrain_obj["dem_bounds_north"]) +
                float(terrain_obj["dem_bounds_south"])
            ) / 2.0
            center_lon = (
                float(terrain_obj["dem_bounds_east"]) +
                float(terrain_obj["dem_bounds_west"])
            ) / 2.0

        if not scale:
            # Reasonable default consistent with create_terrain_mesh /
            # plane_from_dem (1 unit ≈ 1 km).
            scale = 0.001

        print(
            f"[SciGraphs] xy_to_latlon recovered: "
            f"center_lat={center_lat} center_lon={center_lon} scale={scale} "
            f"(parent={parent.name if parent else None}, "
            f"has_bounds={all(k in terrain_obj for k in ('dem_bounds_north','dem_bounds_south','dem_bounds_east','dem_bounds_west'))}, "
            f"all_props={list(terrain_obj.keys())})"
        )

        # Persist what we recovered so the next call is O(1).
        if center_lat is not None and center_lon is not None and scale:
            terrain_obj["dem_center_lat"] = float(center_lat)
            terrain_obj["dem_center_lon"] = float(center_lon)
            terrain_obj["dem_scale"] = float(scale)
        else:
            return None

    earth_radius = 6_371_000.0
    cos_lat = _math.cos(_math.radians(float(center_lat)))
    meters_per_deg_lat = (_math.pi / 180.0) * earth_radius
    meters_per_deg_lon = meters_per_deg_lat * cos_lat
    if abs(meters_per_deg_lon) < 1e-9:
        meters_per_deg_lon = meters_per_deg_lat  # safety at the poles

    inv_scale = 1.0 / float(scale)

    def to_latlon(world_x: float, world_y: float) -> tuple[float, float]:
        x_m = world_x * inv_scale
        y_m = world_y * inv_scale
        lat = float(center_lat) + (y_m / meters_per_deg_lat)
        lon = float(center_lon) + (x_m / meters_per_deg_lon)
        return lat, lon

    return to_latlon


def _project_uv_geographic(terrain_obj, metadata):
    """Per-vertex UV unwrap that respects the **projection of the basemap**.

    Each terrain vertex is converted from Blender world XY back to its
    real ``(lat, lon)`` (using the inverse of the equirectangular-local
    projection that created the mesh), then forwarded to
    :func:`SciGraphs.core.geo.imagery.latlon_to_image_uv`, which knows
    whether the basemap is Web Mercator (XYZ tiles) or WGS84-linear
    (WMS) and produces an exact pixel-accurate UV.

    This is the single point where geographic alignment between graph
    and basemap is enforced, so the result is correct even when the
    Displace modifier deforms Z (UVs only depend on XY) and even when
    we drape a Web Mercator image onto an equirectangular mesh (the
    XY→latlon→pixel chain is exact in both projections).
    """
    from ....core.geo import imagery

    mesh = terrain_obj.data
    if not mesh or len(mesh.vertices) == 0:
        return False, "Terrain has no geometry"

    to_latlon = _terrain_xy_to_latlon_factory(terrain_obj)
    if to_latlon is None:
        return False, (
            "Terrain is missing DEM metadata (dem_center_lat/lon/scale). "
            "Re-import the terrain through 'Get Elevation Data'."
        )

    if not mesh.uv_layers:
        mesh.uv_layers.new(name="BasemapUV")
    uv_layer = mesh.uv_layers.active or mesh.uv_layers[0]

    matrix_world = terrain_obj.matrix_world
    vertex_uvs: list[tuple[float, float]] = []
    for v in mesh.vertices:
        co = matrix_world @ v.co
        lat, lon = to_latlon(co.x, co.y)
        vertex_uvs.append(imagery.latlon_to_image_uv(lat, lon, metadata))

    uv_data = uv_layer.uv if hasattr(uv_layer, "uv") else uv_layer.data
    use_attr = hasattr(uv_layer, "uv")

    for poly in mesh.polygons:
        for loop_idx in poly.loop_indices:
            vert_idx = mesh.loops[loop_idx].vertex_index
            u, v = vertex_uvs[vert_idx]
            if use_attr:
                uv_data[loop_idx].vector = (u, v)
            else:
                uv_data[loop_idx].uv = (u, v)
    return True, None


def _ensure_basemap_material(terrain_obj, image_path, attribution):
    """Create or update a 'SciGraphs_Basemap' material on the terrain."""
    mat_name = f"SciGraphs_Basemap_{terrain_obj.name}"
    mat = bpy.data.materials.get(mat_name)
    if mat is None:
        mat = bpy.data.materials.new(mat_name)
        mat.use_nodes = True

    nt = mat.node_tree
    nt.nodes.clear()

    out = nt.nodes.new('ShaderNodeOutputMaterial')
    out.location = (400, 0)
    bsdf = nt.nodes.new('ShaderNodeBsdfPrincipled')
    bsdf.location = (100, 0)
    if 'Roughness' in bsdf.inputs:
        bsdf.inputs['Roughness'].default_value = 0.85
    if 'Specular IOR Level' in bsdf.inputs:
        bsdf.inputs['Specular IOR Level'].default_value = 0.1
    elif 'Specular' in bsdf.inputs:
        bsdf.inputs['Specular'].default_value = 0.1

    tex = nt.nodes.new('ShaderNodeTexImage')
    tex.location = (-300, 0)
    img_name = os.path.basename(image_path)
    image = bpy.data.images.get(img_name)
    if image is not None:
        try:
            image.filepath = image_path
            image.reload()
        except Exception:  # noqa: BLE001
            image = None
    if image is None:
        image = bpy.data.images.load(image_path, check_existing=True)
    tex.image = image

    nt.links.new(tex.outputs['Color'], bsdf.inputs['Base Color'])
    nt.links.new(bsdf.outputs['BSDF'], out.inputs['Surface'])

    if mat.users == 0 or not terrain_obj.data.materials:
        terrain_obj.data.materials.clear()
        terrain_obj.data.materials.append(mat)
    else:
        terrain_obj.data.materials[0] = mat

    terrain_obj["basemap_attribution"] = attribution
    terrain_obj["basemap_image"] = image_path
    return mat


class SCIGRAPHS_OT_FetchBasemap(bpy.types.Operator):
    """Download a basemap (satellite / map) for the active OSMnx graph
    bbox and apply it as a texture to the terrain mesh.
    """

    bl_idname = "scigraphs.osmnx_fetch_basemap"
    bl_label = "Fetch & Apply Basemap"
    bl_description = (
        "Download an aerial / satellite / map image covering the graph bbox "
        "and drape it onto the terrain mesh as a texture"
    )
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        if not obj or not obj.get("is_osmnx", False):
            return False
        return _resolve_terrain_object(obj) is not None

    def execute(self, context):
        from ....core.geo import imagery, terrain as terrain_module

        props = context.scene.scigraphs
        obj = context.active_object
        terrain_obj = _resolve_terrain_object(obj)
        if terrain_obj is None:
            self.report({'ERROR'}, "No terrain mesh attached to this network")
            return {'CANCELLED'}

        # Prefer the terrain's own DEM bbox: that is the rectangle the
        # terrain mesh actually covers, so the imagery will line up
        # perfectly with the streets even when graph and terrain were
        # created with different padding values. Fall back to the graph
        # bbox only when the terrain has no recorded bounds (legacy data).
        bounds = _terrain_bounds_wgs84(terrain_obj)
        bounds_origin = "terrain"
        if bounds is None:
            bounds = terrain_module.get_osmnx_bounds(
                obj, padding=props.osmnx_basemap_padding
            )
            bounds_origin = "graph"
        if bounds is None:
            self.report({'ERROR'}, "Could not determine geographic bounds")
            return {'CANCELLED'}
        print(
            f"[SciGraphs] Basemap bbox source: {bounds_origin} → "
            f"N={bounds['north']:.5f} S={bounds['south']:.5f} "
            f"E={bounds['east']:.5f} W={bounds['west']:.5f}"
        )

        source = imagery.resolve_source(props.osmnx_basemap_source)

        api_key = None
        wms_url = None
        wms_layer = None
        if source == 'WMS':
            wms_url = (props.osmnx_wms_url or "").strip()
            wms_layer = (props.osmnx_wms_layer or "").strip()
            if not wms_url or not wms_layer:
                self.report({'ERROR'}, "WMS source needs both URL and Layer name")
                return {'CANCELLED'}
        else:
            cfg = imagery.TILE_SOURCES.get(source, {})
            if cfg.get('needs_key'):
                from ....preferences import get_preferences
                prefs = get_preferences()
                key_pref = cfg.get('key_pref')
                api_key = ""
                if prefs and key_pref:
                    api_key = (getattr(prefs, key_pref, "") or "").strip()
                if not api_key:
                    provider = cfg.get('provider', source)
                    self.report(
                        {'ERROR'},
                        f"{provider} requires an API key in Preferences > Add-ons > SciGraphs",
                    )
                    return {'CANCELLED'}

        # Pre-flight tile estimate so the user does not nuke their disk.
        if source != 'WMS':
            est = imagery.estimate_tiles(bounds, props.osmnx_basemap_zoom)
            if est > 400:
                self.report(
                    {'ERROR'},
                    f"Zoom {props.osmnx_basemap_zoom} would need {est} tiles "
                    "(limit 400). Lower the zoom level.",
                )
                return {'CANCELLED'}
            self.report({'INFO'}, f"Fetching ~{est} tiles…")

        try:
            image_path, metadata = imagery.fetch_basemap(
                bounds=bounds,
                source=source,
                zoom=props.osmnx_basemap_zoom,
                api_key=api_key,
                wms_url=wms_url,
                wms_layer=wms_layer,
            )
        except KeyboardInterrupt:
            self.report({'WARNING'}, "Basemap fetch interrupted by user")
            return {'CANCELLED'}
        except (ValueError, RuntimeError) as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        except Exception as exc:  # noqa: BLE001
            self.report({'ERROR'}, f"Unexpected error: {exc}")
            return {'CANCELLED'}

        ok, err = _project_uv_geographic(terrain_obj, metadata)
        if not ok:
            self.report({'ERROR'}, err or "Could not project UVs onto terrain mesh")
            return {'CANCELLED'}

        _ensure_basemap_material(terrain_obj, image_path, metadata['attribution'])

        terrain_obj["basemap_source"] = metadata['source_name']
        terrain_obj["basemap_zoom"] = metadata.get('zoom') or 0

        self.report(
            {'INFO'},
            f"Basemap applied: {metadata['source_name']} "
            f"({metadata['image_size'][0]}×{metadata['image_size'][1]} px) — "
            f"{metadata['attribution']}",
        )
        return {'FINISHED'}


class SCIGRAPHS_OT_ClearBasemap(bpy.types.Operator):
    """Remove the basemap material from the terrain mesh."""

    bl_idname = "scigraphs.osmnx_clear_basemap"
    bl_label = "Clear Basemap"
    bl_description = "Detach the basemap texture from the terrain"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        if not obj or not obj.get("is_osmnx", False):
            return False
        terrain_obj = _resolve_terrain_object(obj)
        if terrain_obj is None:
            return False
        return bool(terrain_obj.get("basemap_image"))

    def execute(self, context):
        obj = context.active_object
        terrain_obj = _resolve_terrain_object(obj)
        if terrain_obj is None:
            self.report({'WARNING'}, "No terrain mesh attached")
            return {'CANCELLED'}

        mat_name = f"SciGraphs_Basemap_{terrain_obj.name}"
        mat = bpy.data.materials.get(mat_name)
        if mat is not None:
            for slot in list(terrain_obj.data.materials):
                if slot is mat:
                    idx = list(terrain_obj.data.materials).index(slot)
                    terrain_obj.data.materials.pop(index=idx)
            try:
                bpy.data.materials.remove(mat, do_unlink=True)
            except RuntimeError:
                pass

        for key in ("basemap_attribution", "basemap_image",
                    "basemap_source", "basemap_zoom"):
            terrain_obj.pop(key, None)

        self.report({'INFO'}, "Basemap cleared")
        return {'FINISHED'}


classes = [
    SCIGRAPHS_OT_AddElevationsRaster,
    SCIGRAPHS_OT_AddElevationsAPI,
    SCIGRAPHS_OT_AddEdgeGrades,
    SCIGRAPHS_OT_ApplyElevation3D,
    SCIGRAPHS_OT_FlattenNetwork,
    SCIGRAPHS_OT_ImportTerrain,
    SCIGRAPHS_OT_ImportTerrainAPI,
    SCIGRAPHS_OT_ToggleTerrain,
    SCIGRAPHS_OT_ExportAOI,
    # Legacy "Georef Image Plane" operators kept registered for backwards
    # compatibility with old .blend files but no longer surfaced in the UI.
    SCIGRAPHS_OT_ImportTerrainPlane,
    SCIGRAPHS_OT_UpdateTerrainPlaneOffset,
    SCIGRAPHS_OT_RemoveTerrainPlane,
    SCIGRAPHS_OT_ImportDEMDisplace,
    SCIGRAPHS_OT_ImportDEMRawMesh,
    SCIGRAPHS_OT_DownloadDEM,
    SCIGRAPHS_OT_UpdateTerrainScale,
    SCIGRAPHS_OT_RemoveTerrain,
    SCIGRAPHS_OT_TerrainMaterial,
    SCIGRAPHS_OT_GetElevationData,
    SCIGRAPHS_OT_RemoveTerrainChild,
    SCIGRAPHS_OT_FetchBasemap,
    SCIGRAPHS_OT_ClearBasemap,
]


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)

