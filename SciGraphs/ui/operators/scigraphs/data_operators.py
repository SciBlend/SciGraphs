# Data import and configuration operators

import bpy
import numpy as np
from ....core import importer, geometry, graph, layout as graph_layout
from ...view_utils import focus_graph_in_top_view


class SCIGRAPHS_AutoLayoutOnImport:
    """Shared auto-layout behavior for graph creation operators."""

    @classmethod
    def apply(cls, context, obj, operator=None, skip_reason=None):
        props = context.scene.scigraphs

        if not getattr(props, "auto_layout_on_import", True):
            return False

        if skip_reason:
            if operator:
                operator.report({'INFO'}, f"Auto-layout skipped: {skip_reason}")
            return False

        if obj is None or "num_nodes" not in obj:
            return False

        success = graph_layout.apply_graph_layout(
            obj,
            algorithm=props.layout_algorithm,
            iterations=props.iterations,
            scale=props.layout_scale,
            props=props,
        )

        if not success:
            if operator:
                operator.report({'WARNING'}, "Auto-layout failed; graph was still imported")
            return False

        geometry.update_node_positions_from_property(obj)
        geometry.rebuild_edges(obj)
        obj["auto_layout_algorithm"] = props.layout_algorithm

        if operator:
            operator.report({'INFO'}, f"Auto-layout applied: {props.layout_algorithm}")

        return True


class SCIGRAPHS_OT_LoadColumns(bpy.types.Operator):
    """Load column names from CSV file and auto-detect data types."""
    bl_idname = "scigraphs.load_columns"
    bl_label = "Load Columns"
    bl_description = "Load column names from the CSV file"
    
    def execute(self, context):
        import pandas as pd
        
        props = context.scene.scigraphs
        
        if not props.filepath:
            self.report({'WARNING'}, "Please select a file first")
            return {'CANCELLED'}

        if importer.is_graph_file(props.filepath):
            graph_data = importer.load_native_graph_file(props.filepath)
            if graph_data is None or graph_data.dataframe is None:
                self.report({'ERROR'}, "Could not read graph file")
                return {'CANCELLED'}
            df = graph_data.dataframe
            columns = list(df.columns)
        else:
            columns = importer.get_columns_from_file(props.filepath, props.csv_delimiter)
        
        if not columns:
            self.report({'ERROR'}, "Could not read columns from file")
            return {'CANCELLED'}
        
        # Read a sample to detect column types
        if not importer.is_graph_file(props.filepath):
            try:
                df = pd.read_csv(props.filepath, nrows=10, delimiter=props.csv_delimiter)
            except Exception as e:
                self.report({'ERROR'}, f"Could not read file: {e}")
                return {'CANCELLED'}
        
        # Clear existing column list
        props.available_csv_columns.clear()
        
        # Populate column list with type information
        for col in columns:
            item = props.available_csv_columns.add()
            item.name = col
            
            # Determine column type
            if col in df.columns:
                dtype = df[col].dtype
                
                # Try to detect if column is actually numeric even if detected as object
                if pd.api.types.is_numeric_dtype(dtype):
                    item.column_type = "numeric"
                    item.import_as_attribute = True
                elif pd.api.types.is_datetime64_any_dtype(dtype):
                    item.column_type = "datetime"
                    item.import_as_attribute = True
                else:
                    # For object columns, try to convert to numeric
                    # If most values can be converted, it's probably numeric
                    try:
                        converted = pd.to_numeric(df[col], errors='coerce')
                        non_null_count = converted.notna().sum()
                        total_count = len(df[col])
                        
                        # If at least 50% of values are numeric, treat as numeric
                        if total_count > 0 and (non_null_count / total_count) > 0.5:
                            item.column_type = "numeric"
                            item.import_as_attribute = True
                        else:
                            item.column_type = "text"
                            item.import_as_attribute = False
                    except:
                        item.column_type = "text"
                        item.import_as_attribute = False
        
        # Auto-detect geospatial and temporal data for tabular files only.
        if not importer.is_graph_file(props.filepath):
            bpy.ops.scigraphs.detect_geospatial()
        
        self.report({'INFO'}, f"Loaded {len(columns)} columns")
        return {'FINISHED'}


class SCIGRAPHS_OT_DetectGeospatial(bpy.types.Operator):
    """Auto-detect geospatial and temporal columns in data."""
    bl_idname = "scigraphs.detect_geospatial"
    bl_label = "Auto-Detect Geospatial Data"
    bl_description = "Automatically detect geospatial and temporal columns"
    
    def execute(self, context):
        from ....core import geospatial
        import pandas as pd
        
        props = context.scene.scigraphs
        
        if not props.filepath:
            return {'CANCELLED'}
        
        try:
            # Read a sample of the file for detection
            df = pd.read_csv(props.filepath, nrows=100, delimiter=props.csv_delimiter)
            
            # Detect lat/lon columns
            lat_col, lon_col = geospatial.detect_geospatial_columns(df)
            if lat_col and lon_col:
                props.use_geospatial = True
                props.latitude_column = str(df.columns.get_loc(lat_col))
                props.longitude_column = str(df.columns.get_loc(lon_col))
                self.report({'INFO'}, f"Detected geospatial columns: {lat_col}, {lon_col}")
            
            # Detect country columns
            elif geospatial.detect_country_columns(df):
                props.use_geospatial = True
                props.geocode_columns = True
                self.report({'INFO'}, "Detected country data - geocoding enabled")
            
            # Detect temporal columns
            time_col = geospatial.detect_temporal_columns(df)
            if time_col:
                props.has_temporal_data = True
                props.time_column = str(df.columns.get_loc(time_col))
                self.report({'INFO'}, f"Detected temporal column: {time_col}")
            
            # Try to detect weight column (look for 'value', 'weight', 'count' etc.)
            weight_patterns = ['value', 'weight', 'count', 'amount']
            for col in df.columns:
                col_lower = col.lower()
                if any(pattern in col_lower for pattern in weight_patterns):
                    if pd.api.types.is_numeric_dtype(df[col]):
                        props.weight_column = str(df.columns.get_loc(col))
                        self.report({'INFO'}, f"Detected weight column: {col}")
                        break
            
        except Exception as e:
            print(f"Detection error: {e}")
        
        return {'FINISHED'}


class SCIGRAPHS_OT_CreateGraph(bpy.types.Operator):
    """Create graph object from loaded data."""
    bl_idname = "scigraphs.create_graph"
    bl_label = "Create Graph"
    bl_description = "Generate the graph from the specified data"
    
    def execute(self, context):
        from ....core import geospatial
        
        props = context.scene.scigraphs
        
        if not props.filepath:
            self.report({'ERROR'}, "No file selected")
            return {'CANCELLED'}
        
        try:
            source_col = int(props.source_column)
            target_col = int(props.target_column)
        except ValueError:
            self.report({'ERROR'}, "Invalid column selection")
            return {'CANCELLED'}
        
        # Handle geospatial mode
        if props.use_geospatial:
            self.report({'INFO'}, "Creating geospatial graph...")
            
            # Parse column indices
            lat_col = None
            lon_col = None
            
            # Only use lat/lon columns if NOT using geocoding
            if not props.geocode_columns:
                if props.latitude_column != '0' and props.longitude_column != '0':
                    try:
                        lat_col = int(props.latitude_column)
                        lon_col = int(props.longitude_column)
                    except ValueError:
                        pass
            
            time_col = None
            if props.has_temporal_data and props.time_column != '0':
                try:
                    time_col = int(props.time_column)
                except ValueError:
                    pass
            
            weight_col = None
            if props.weight_column != '0':
                try:
                    weight_col = int(props.weight_column)
                except ValueError:
                    pass
            
            # Inform user if geocoding will be performed
            if props.geocode_columns:
                self.report({'INFO'}, "Geocoding country/city names to coordinates... (check console for progress)")
            
            # Load geospatial graph data
            graph_data = importer.load_geospatial_graph(
                props.filepath,
                source_col,
                target_col,
                lat_col=lat_col,
                lon_col=lon_col,
                geocode_mode=props.geocode_columns,
                time_col=time_col,
                time_agg=props.time_aggregation,
                time_start=props.time_range_start if props.time_range_start else None,
                time_end=props.time_range_end if props.time_range_end else None,
                weight_col=weight_col,
                delimiter=props.csv_delimiter,
            )
            
            if not graph_data:
                self.report({'ERROR'}, "Could not load graph data. Check file and columns.")
                return {'CANCELLED'}
            
            if not hasattr(graph_data, 'node_coordinates') or not graph_data.node_coordinates:
                if props.geocode_columns:
                    self.report({'ERROR'}, "Geocoding failed. Check console for details. Country names may not be recognized.")
                else:
                    self.report({'ERROR'}, "No geographic coordinates found. Check lat/lon columns.")
                return {'CANCELLED'}
            
            # Calculate 3D positions from coordinates
            positions_3d = geospatial.calculate_sphere_positions(
                graph_data.node_coordinates,
                props.globe_radius
            )
            
            # Get selected attributes to import
            selected_attrs = [item.name for item in props.available_csv_columns if item.import_as_attribute]
            
            # Create geospatial graph
            obj = geometry.create_geospatial_graph_object(
                graph_data,
                positions_3d,
                edge_style=props.edge_style,
                is_directed=props.is_directed,
                selected_attributes=selected_attrs if selected_attrs else None,
                remove_self_loops=props.remove_self_loops,
            )
            
            if props.show_globe:
                globe_obj = geospatial.create_globe_mesh(
                    radius=props.globe_radius,
                    subdivisions=props.globe_subdivisions,
                    material_style=props.globe_material,
                    map_resolution=props.map_resolution,
                    feature_type=props.map_feature_type,
                    globe_theme=props.globe_theme_api,
                    texture_resolution=props.globe_texture_resolution,
                    water_specular=props.globe_water_specular,
                    water_roughness=props.globe_water_roughness,
                    land_roughness=props.globe_land_roughness,
                    bump_strength=props.globe_bump_strength
                )
                
                if props.globe_theme_api != 'NONE':
                    self.report({'INFO'}, f"Created Earth globe with {props.globe_theme_api} texture")
                else:
                    self.report({'INFO'}, f"Created Earth globe with {props.globe_material} material")
            
            focus_graph_in_top_view(context, obj)
            self.report({'INFO'}, f"Geospatial graph created: {len(graph_data.nodes)} nodes, {len(graph_data.edges)} edges")
            
        else:
            # Standard non-geospatial graph
            graph_data = importer.load_graph_from_file(
                props.filepath,
                source_col,
                target_col,
                delimiter=props.csv_delimiter
            )
            
            if not graph_data:
                self.report({'ERROR'}, "Could not load graph data. Check file and columns.")
                return {'CANCELLED'}
            
            selected_attrs = [
                item.name for item in props.available_csv_columns
                if item.import_as_attribute
            ]
            
            obj = geometry.create_graph_object(
                graph_data,
                is_directed=props.is_directed,
                selected_attributes=selected_attrs if selected_attrs else None,
                remove_self_loops=props.remove_self_loops,
            )

            auto_layout_applied = SCIGRAPHS_AutoLayoutOnImport.apply(context, obj, self)
            focus_graph_in_top_view(context, obj)
            suffix = f" ({props.layout_algorithm} layout applied)" if auto_layout_applied else ""
            self.report({'INFO'}, f"Graph created: {len(graph_data.nodes)} nodes, {len(graph_data.edges)} edges{suffix}")
        
        return {'FINISHED'}


class SCIGRAPHS_OT_ToggleAllAttributes(bpy.types.Operator):
    """Toggle all CSV column attribute checkboxes on or off."""
    bl_idname = "scigraphs.toggle_all_attributes"
    bl_label = "Toggle All Attributes"
    bl_description = "Select or deselect all columns for attribute import"
    
    enable: bpy.props.BoolProperty(default=True)
    
    def execute(self, context):
        props = context.scene.scigraphs
        for item in props.available_csv_columns:
            item.import_as_attribute = self.enable
        return {'FINISHED'}


class SCIGRAPHS_OT_ImportNodeAttributes(bpy.types.Operator):
    """Import vertex-only attributes from an external file onto the active graph."""
    bl_idname = "scigraphs.import_node_attributes"
    bl_label = "Import Node Attributes"
    bl_description = "Load a file with per-node values and store them as POINT attributes"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj is not None and obj.type == 'MESH' and "num_nodes" in obj

    def execute(self, context):
        props = context.scene.scigraphs

        if not props.node_attr_filepath:
            self.report({'WARNING'}, "Please select a node attribute file")
            return {'CANCELLED'}

        import os
        resolved = bpy.path.abspath(props.node_attr_filepath)
        if not os.path.isfile(resolved):
            self.report({'ERROR'}, f"File not found: {resolved}")
            return {'CANCELLED'}

        obj = context.active_object

        num_attrs, num_matched = geometry.import_node_attributes_from_file(
            obj,
            resolved,
            delimiter=props.node_attr_delimiter,
            has_header=props.node_attr_has_header,
        )

        if num_attrs == 0:
            self.report({'ERROR'}, "No attributes could be imported. Check file format.")
            return {'CANCELLED'}

        geometry._rebuild_visualization_if_present(obj)

        num_nodes = obj.get("num_nodes", 0)
        self.report(
            {'INFO'},
            f"Imported {num_attrs} attribute(s): {num_matched}/{num_nodes} nodes matched"
        )
        return {'FINISHED'}


class SCIGRAPHS_OT_SetupVisualization(bpy.types.Operator):
    """Setup Geometry Nodes for graph visualization."""
    bl_idname = "scigraphs.setup_visualization"
    bl_label = "Setup Visualization"
    bl_description = "Add Geometry Nodes modifier for visual representation"
    
    def execute(self, context):
        obj = context.active_object
        
        if not obj or "num_nodes" not in obj:
            self.report({'ERROR'}, "No graph object selected")
            return {'CANCELLED'}
        
        # Setup Geometry Nodes modifier
        geometry.setup_geometry_nodes_visualization(obj)
        
        self.report({'INFO'}, "Geometry Nodes modifier added")
        return {'FINISHED'}


class SCIGRAPHS_OT_ImportOSMGraph(bpy.types.Operator):
    """Import street network from OpenStreetMap using OSMnx."""
    bl_idname = "scigraphs.import_osm_graph"
    bl_label = "Import OSM Graph"
    bl_description = "Download and import street network from OpenStreetMap"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        props = context.scene.scigraphs
        method = props.osmnx_download_method

        if method == 'PLACE':
            if not props.osmnx_place_name.strip():
                self.report({'ERROR'}, "Please enter a place name")
                return {'CANCELLED'}

        elif method == 'MULTI_PLACE':
            if not props.osmnx_place_list.strip():
                self.report({'ERROR'}, "Enter one place name per line in Place List")
                return {'CANCELLED'}

        elif method == 'ADDRESS':
            if not props.osmnx_address.strip():
                self.report({'ERROR'}, "Please enter an address")
                return {'CANCELLED'}

        elif method == 'BBOX':
            if props.osmnx_bbox_north <= props.osmnx_bbox_south:
                self.report({'ERROR'}, "North latitude must be greater than South")
                return {'CANCELLED'}
            if props.osmnx_bbox_east <= props.osmnx_bbox_west:
                self.report({'ERROR'}, "East longitude must be greater than West")
                return {'CANCELLED'}

        elif method == 'XML':
            if not props.osmnx_xml_filepath or not props.osmnx_xml_filepath.strip():
                self.report({'ERROR'}, "Pick an .osm XML file")
                return {'CANCELLED'}

        # Resolve POLYGON source from a Blender mesh object.
        polygon = None
        if method == 'POLYGON':
            obj_name = props.osmnx_polygon_object.strip()
            if not obj_name:
                self.report({'ERROR'}, "Pick a Blender mesh object to use as polygon")
                return {'CANCELLED'}
            poly_obj = bpy.data.objects.get(obj_name)
            if poly_obj is None or poly_obj.type != 'MESH':
                self.report({'ERROR'}, f"Object '{obj_name}' is not a mesh")
                return {'CANCELLED'}
            try:
                from shapely.geometry import Polygon
            except ImportError:
                self.report({'ERROR'}, "Shapely is required for POLYGON downloads")
                return {'CANCELLED'}
            # Interpret mesh vertices as lon/lat (user responsibility).
            # Accept any mesh; flatten Z and build a simple polygon from first face.
            mesh = poly_obj.data
            if len(mesh.polygons) > 0:
                face = mesh.polygons[0]
                verts = [tuple(mesh.vertices[vi].co.xy) for vi in face.vertices]
            else:
                verts = [(v.co.x, v.co.y) for v in mesh.vertices]
            if len(verts) < 3:
                self.report({'ERROR'}, "Polygon object must have at least 3 vertices")
                return {'CANCELLED'}
            polygon = Polygon(verts)

        place_list = None
        if method == 'MULTI_PLACE':
            place_list = [p.strip() for p in props.osmnx_place_list.splitlines() if p.strip()]
            if not place_list:
                self.report({'ERROR'}, "Place list is empty")
                return {'CANCELLED'}

        custom_filter = props.osmnx_custom_filter_text.strip() or props.osmnx_custom_filter_preset
        which_result = props.osmnx_which_result if props.osmnx_which_result > 0 else None

        self.report({'INFO'}, f"Downloading network from OpenStreetMap ({method})...")

        xml_fp = bpy.path.abspath(props.osmnx_xml_filepath) if props.osmnx_xml_filepath else ''

        graph_data, edge_geometries = importer.load_osmnx_graph(
            method=method,
            place_name=props.osmnx_place_name,
            latitude=props.osmnx_latitude,
            longitude=props.osmnx_longitude,
            distance=props.osmnx_distance,
            address=props.osmnx_address,
            bbox_north=props.osmnx_bbox_north,
            bbox_south=props.osmnx_bbox_south,
            bbox_east=props.osmnx_bbox_east,
            bbox_west=props.osmnx_bbox_west,
            polygon=polygon,
            xml_filepath=xml_fp,
            place_list=place_list,
            network_type=props.osmnx_network_type,
            simplify=props.osmnx_simplify,
            retain_geometry=props.osmnx_retain_geometry,
            truncate_by_edge=props.osmnx_truncate_by_edge,
            retain_all=props.osmnx_retain_all,
            custom_filter=custom_filter,
            which_result=which_result,
        )
        
        if graph_data is None:
            self.report({'ERROR'}, "Failed to download network. Check console for details.")
            return {'CANCELLED'}
        
        # Create the graph object in Blender
        obj = geometry.create_osmnx_graph_object(
            graph_data,
            edge_geometries,
            scale=props.osmnx_scale,
            retain_geometry=props.osmnx_retain_geometry,
        )
        
        if obj is None:
            self.report({'ERROR'}, "Failed to create graph object")
            return {'CANCELLED'}
        
        # Store the OSMnx graph in cache for analysis operators
        if hasattr(graph_data, 'osmnx_graph') and graph_data.osmnx_graph is not None:
            import uuid
            if not hasattr(importer, '_osmnx_graph_cache'):
                importer._osmnx_graph_cache = {}
            
            # Generate unique ID for this graph
            graph_id = str(uuid.uuid4())
            obj["osmnx_graph_id"] = graph_id
            
            # Store the original graph
            importer._osmnx_graph_cache[graph_id] = graph_data.osmnx_graph
            
            # Also store as unprojected version (original is always unprojected)
            # This ensures spatial queries work even if user projects the graph
            importer._osmnx_graph_cache[graph_id + "_unprojected"] = graph_data.osmnx_graph.copy()
            
            obj["osmnx_scale"] = props.osmnx_scale
            
            # Store metadata for cache filename generation and reload
            obj["osmnx_method"] = method
            obj["osmnx_network_type"] = props.osmnx_network_type
            
            if method == 'PLACE':
                obj["osmnx_query_name"] = props.osmnx_place_name
            elif method == 'MULTI_PLACE':
                first = (place_list or [''])[0]
                obj["osmnx_query_name"] = f"multi_{first}_{len(place_list or [])}"
            elif method == 'ADDRESS':
                obj["osmnx_query_name"] = props.osmnx_address
            elif method == 'POINT':
                obj["osmnx_query_name"] = f"{props.osmnx_latitude:.4f}_{props.osmnx_longitude:.4f}"
            elif method == 'BBOX':
                obj["osmnx_query_name"] = f"bbox_{props.osmnx_bbox_north:.2f}_{props.osmnx_bbox_south:.2f}"
            elif method == 'POLYGON':
                obj["osmnx_query_name"] = f"polygon_{props.osmnx_polygon_object}"
            elif method == 'XML':
                import os as _os
                obj["osmnx_query_name"] = f"xml_{_os.path.splitext(_os.path.basename(xml_fp))[0]}"
            
            # Automatically save to cache
            from ....core.osmnx import cache
            success, filepath, message = cache.save_graph_to_cache(obj, graph_data.osmnx_graph)
            if success:
                import os
                filename = os.path.basename(filepath) if filepath else ""
                self.report({'INFO'}, f"Graph cached as: {filename}")
            else:
                # Log warning but don't interrupt workflow
                from ....utils.logger import log
                log(f"Warning: Failed to auto-save graph to cache: {message}")
        
        # Calculate stats
        num_nodes = len(graph_data.nodes)
        num_edges = len(graph_data.edges)
        total_length = sum(getattr(graph_data, 'edge_lengths', [0])) / 1000  # Convert to km
        
        focus_graph_in_top_view(context, obj)
        self.report(
            {'INFO'}, 
            f"Imported: {num_nodes} intersections, {num_edges} street segments, {total_length:.1f} km total"
        )
        return {'FINISHED'}


class SCIGRAPHS_OT_ClearTextureCache(bpy.types.Operator):
    """Clear downloaded satellite textures from cache."""
    bl_idname = "scigraphs.clear_texture_cache"
    bl_label = "Clear Texture Cache"
    bl_description = "Remove all cached satellite/map textures to free disk space"
    bl_options = {'REGISTER'}
    
    def execute(self, context):
        from ....core import texture_api
        
        size_bytes, size_str = texture_api.get_cache_size()
        deleted = texture_api.clear_texture_cache()
        
        if deleted > 0:
            self.report({'INFO'}, f"Cleared {deleted} cached textures ({size_str} freed)")
        else:
            self.report({'INFO'}, "Texture cache is already empty")
        
        return {'FINISHED'}
    
    def invoke(self, context, event):
        return context.window_manager.invoke_confirm(self, event)


class SCIGRAPHS_OT_DownloadGlobeTexture(bpy.types.Operator):
    """Pre-download a globe texture without creating the globe."""
    bl_idname = "scigraphs.download_globe_texture"
    bl_label = "Download Globe Texture"
    bl_description = "Download and cache the selected globe texture"
    bl_options = {'REGISTER'}
    
    def execute(self, context):
        from ....core import texture_api
        
        props = context.scene.scigraphs
        
        if props.globe_theme_api == 'NONE':
            self.report({'WARNING'}, "Select a globe theme first")
            return {'CANCELLED'}
        
        self.report({'INFO'}, f"Downloading {props.globe_theme_api} texture ({props.globe_texture_resolution})...")
        
        texture_path = texture_api.download_texture(
            props.globe_theme_api,
            props.globe_texture_resolution,
            force_download=False
        )
        
        if texture_path:
            self.report({'INFO'}, f"Texture ready: {texture_path}")
        else:
            self.report({'ERROR'}, "Failed to download texture")
            return {'CANCELLED'}
        
        return {'FINISHED'}


def register():
    bpy.utils.register_class(SCIGRAPHS_OT_LoadColumns)
    bpy.utils.register_class(SCIGRAPHS_OT_DetectGeospatial)
    bpy.utils.register_class(SCIGRAPHS_OT_CreateGraph)
    bpy.utils.register_class(SCIGRAPHS_OT_ToggleAllAttributes)
    bpy.utils.register_class(SCIGRAPHS_OT_ImportNodeAttributes)
    bpy.utils.register_class(SCIGRAPHS_OT_SetupVisualization)
    bpy.utils.register_class(SCIGRAPHS_OT_ImportOSMGraph)
    bpy.utils.register_class(SCIGRAPHS_OT_ClearTextureCache)
    bpy.utils.register_class(SCIGRAPHS_OT_DownloadGlobeTexture)


def unregister():
    bpy.utils.unregister_class(SCIGRAPHS_OT_DownloadGlobeTexture)
    bpy.utils.unregister_class(SCIGRAPHS_OT_ClearTextureCache)
    bpy.utils.unregister_class(SCIGRAPHS_OT_ImportOSMGraph)
    bpy.utils.unregister_class(SCIGRAPHS_OT_SetupVisualization)
    bpy.utils.unregister_class(SCIGRAPHS_OT_ImportNodeAttributes)
    bpy.utils.unregister_class(SCIGRAPHS_OT_ToggleAllAttributes)
    bpy.utils.unregister_class(SCIGRAPHS_OT_CreateGraph)
    bpy.utils.unregister_class(SCIGRAPHS_OT_DetectGeospatial)
    bpy.utils.unregister_class(SCIGRAPHS_OT_LoadColumns)

