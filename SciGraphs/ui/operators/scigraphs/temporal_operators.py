import bpy
import json
from bpy.props import IntProperty, BoolProperty


class SCIGRAPHS_OT_AnalyzeTimeColumn(bpy.types.Operator):
    """Analyze the selected time column to find available time periods."""
    bl_idname = "scigraphs.analyze_time_column"
    bl_label = "Analyze Time Column"
    bl_description = "Scan CSV to find unique time values in the selected column"
    bl_options = {'REGISTER'}
    
    def execute(self, context):
        import pandas as pd
        
        props = context.scene.scigraphs
        
        if not props.filepath:
            self.report({'ERROR'}, "No file selected")
            return {'CANCELLED'}
        
        try:
            time_col_idx = int(props.time_column)
        except ValueError:
            self.report({'ERROR'}, "Select a time column first")
            return {'CANCELLED'}
        
        try:
            df = pd.read_csv(props.filepath, low_memory=False, delimiter=props.csv_delimiter)
            time_col_name = df.columns[time_col_idx]
            time_series = df[time_col_name]
            
            if pd.api.types.is_numeric_dtype(time_series):
                unique_values = sorted(time_series.dropna().unique())
                time_values = [str(int(v)) if float(v).is_integer() else str(v) for v in unique_values]
            else:
                unique_values = sorted(time_series.dropna().unique(), key=str)
                time_values = [str(v) for v in unique_values]
            
            if not time_values:
                self.report({'ERROR'}, "No valid time values found in column")
                return {'CANCELLED'}
            
            props.temporal_available_values_json = json.dumps(time_values)
            props.temporal_analyzed = True
            props.temporal_selected_start_idx = 0
            props.temporal_selected_end_idx = len(time_values) - 1
            
            self.report({'INFO'}, f"Found {len(time_values)} time periods: {time_values[0]} to {time_values[-1]}")
            
        except Exception as e:
            self.report({'ERROR'}, f"Error analyzing time column: {e}")
            return {'CANCELLED'}
        
        return {'FINISHED'}


class SCIGRAPHS_OT_CreateTemporalGraphs(bpy.types.Operator):
    """Create multiple graph objects, one for each selected time period."""
    bl_idname = "scigraphs.create_temporal_graphs"
    bl_label = "Create Temporal Graphs"
    bl_description = "Generate separate graph objects for each time period in the selected range"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        import pandas as pd
        from ....core import importer, geometry, graph as graph_module
        from ....core import geospatial
        
        props = context.scene.scigraphs
        
        if not props.temporal_analyzed:
            self.report({'ERROR'}, "Analyze time column first")
            return {'CANCELLED'}
        
        try:
            time_values = json.loads(props.temporal_available_values_json)
        except:
            self.report({'ERROR'}, "No time values available")
            return {'CANCELLED'}
        
        start_idx = props.temporal_selected_start_idx
        end_idx = props.temporal_selected_end_idx
        
        if start_idx > end_idx:
            start_idx, end_idx = end_idx, start_idx
        
        selected_times = time_values[start_idx:end_idx + 1]
        
        if not selected_times:
            self.report({'ERROR'}, "No time periods selected")
            return {'CANCELLED'}
        
        try:
            source_col = int(props.source_column)
            target_col = int(props.target_column)
            time_col_idx = int(props.time_column)
        except ValueError:
            self.report({'ERROR'}, "Invalid column selection")
            return {'CANCELLED'}
        
        try:
            df = pd.read_csv(props.filepath, low_memory=False, delimiter=props.csv_delimiter)
            source_col_name = df.columns[source_col]
            target_col_name = df.columns[target_col]
            time_col_name = df.columns[time_col_idx]
        except Exception as e:
            self.report({'ERROR'}, f"Error reading CSV: {e}")
            return {'CANCELLED'}
        
        collection = bpy.data.collections.new("SciGraphs_Temporal")
        context.scene.collection.children.link(collection)
        
        created_objects = []
        
        for i, time_val in enumerate(selected_times):
            self.report({'INFO'}, f"Creating graph for {time_val} ({i+1}/{len(selected_times)})...")
            
            if pd.api.types.is_numeric_dtype(df[time_col_name]):
                try:
                    numeric_val = float(time_val)
                    if numeric_val.is_integer():
                        numeric_val = int(numeric_val)
                    mask = df[time_col_name] == numeric_val
                except:
                    mask = df[time_col_name].astype(str) == time_val
            else:
                mask = df[time_col_name].astype(str) == time_val
            
            filtered_df = df[mask].copy()
            
            if len(filtered_df) == 0:
                print(f"  No data for time period: {time_val}")
                continue
            
            source_values = filtered_df[source_col_name].values
            target_values = filtered_df[target_col_name].values
            edges = list(zip(source_values, target_values))
            
            import numpy as np
            all_node_values = np.concatenate([source_values, target_values])
            try:
                nodes = np.unique(all_node_values)
            except TypeError:
                nodes = pd.Series(all_node_values).dropna().unique()
            
            graph_data = graph_module.GraphData(nodes, edges, filtered_df)
            
            if props.use_geospatial and props.geocode_columns:
                graph_data = self._add_geospatial_data(graph_data, props)
                
                if hasattr(graph_data, 'node_coordinates') and graph_data.node_coordinates:
                    positions_3d = geospatial.calculate_sphere_positions(
                        graph_data.node_coordinates,
                        props.globe_radius
                    )
                    obj = geometry.create_geospatial_graph_object(
                        graph_data,
                        positions_3d,
                        edge_style=props.edge_style,
                        is_directed=props.is_directed
                    )
                else:
                    obj = geometry.create_graph_object(graph_data, is_directed=props.is_directed)
            else:
                obj = geometry.create_graph_object(graph_data, is_directed=props.is_directed)
            
            obj.name = f"Graph_{time_val}"
            obj["temporal_time_value"] = time_val
            obj["temporal_index"] = i
            
            for old_col in obj.users_collection:
                old_col.objects.unlink(obj)
            collection.objects.link(obj)
            
            if i > 0:
                obj.hide_viewport = True
                obj.hide_render = True
            
            created_objects.append(obj)
            
            print(f"  Created: {obj.name} ({len(graph_data.nodes)} nodes, {len(graph_data.edges)} edges)")
        
        if props.use_geospatial and props.show_globe:
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
            for old_col in globe_obj.users_collection:
                old_col.objects.unlink(globe_obj)
            collection.objects.link(globe_obj)
        
        props.temporal_values_json = json.dumps(selected_times)
        props.temporal_max_index = len(created_objects) - 1
        props.temporal_graph_loaded = True
        props.temporal_current_index = 0
        
        self.report({'INFO'}, f"Created {len(created_objects)} temporal graphs")
        return {'FINISHED'}
    
    def _add_geospatial_data(self, graph_data, props):
        """Add geocoded coordinates to graph data."""
        from ....core import geospatial
        
        all_locations = list(set(graph_data.nodes))
        
        geocoded = geospatial.geocode_locations(all_locations, use_cache=True)
        
        if geocoded:
            graph_data.node_coordinates = geocoded
        
        return graph_data


class SCIGRAPHS_OT_TemporalNext(bpy.types.Operator):
    """Move to the next time period."""
    bl_idname = "scigraphs.temporal_next"
    bl_label = "Next Time"
    bl_description = "Move to the next time period"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        props = context.scene.scigraphs
        
        if not props.temporal_graph_loaded:
            self.report({'WARNING'}, "No temporal graph loaded")
            return {'CANCELLED'}
        
        try:
            temporal_values = json.loads(props.temporal_values_json)
            max_index = len(temporal_values) - 1
            
            if props.temporal_current_index < max_index:
                props.temporal_current_index += 1
            elif props.temporal_loop_animation:
                props.temporal_current_index = 0
            
        except Exception as e:
            self.report({'ERROR'}, f"Error: {e}")
            return {'CANCELLED'}
        
        return {'FINISHED'}


class SCIGRAPHS_OT_TemporalPrevious(bpy.types.Operator):
    """Move to the previous time period."""
    bl_idname = "scigraphs.temporal_previous"
    bl_label = "Previous Time"
    bl_description = "Move to the previous time period"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        props = context.scene.scigraphs
        
        if not props.temporal_graph_loaded:
            self.report({'WARNING'}, "No temporal graph loaded")
            return {'CANCELLED'}
        
        try:
            temporal_values = json.loads(props.temporal_values_json)
            
            if props.temporal_current_index > 0:
                props.temporal_current_index -= 1
            elif props.temporal_loop_animation:
                props.temporal_current_index = len(temporal_values) - 1
            
        except Exception as e:
            self.report({'ERROR'}, f"Error: {e}")
            return {'CANCELLED'}
        
        return {'FINISHED'}


class SCIGRAPHS_OT_TemporalFirst(bpy.types.Operator):
    """Jump to the first time period."""
    bl_idname = "scigraphs.temporal_first"
    bl_label = "First Time"
    bl_description = "Jump to the first time period"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        props = context.scene.scigraphs
        
        if not props.temporal_graph_loaded:
            self.report({'WARNING'}, "No temporal graph loaded")
            return {'CANCELLED'}
        
        props.temporal_current_index = 0
        return {'FINISHED'}


class SCIGRAPHS_OT_TemporalLast(bpy.types.Operator):
    """Jump to the last time period."""
    bl_idname = "scigraphs.temporal_last"
    bl_label = "Last Time"
    bl_description = "Jump to the last time period"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        props = context.scene.scigraphs
        
        if not props.temporal_graph_loaded:
            self.report({'WARNING'}, "No temporal graph loaded")
            return {'CANCELLED'}
        
        try:
            temporal_values = json.loads(props.temporal_values_json)
            props.temporal_current_index = len(temporal_values) - 1
        except:
            pass
        
        return {'FINISHED'}


class SCIGRAPHS_OT_TemporalPlay(bpy.types.Operator):
    """Play/pause temporal animation."""
    bl_idname = "scigraphs.temporal_play"
    bl_label = "Play/Pause"
    bl_description = "Start or stop temporal animation playback"
    bl_options = {'REGISTER'}
    
    _timer = None
    
    def modal(self, context, event):
        props = context.scene.scigraphs
        
        if not props.temporal_animation_playing:
            self.cancel(context)
            return {'CANCELLED'}
        
        if event.type == 'TIMER':
            try:
                temporal_values = json.loads(props.temporal_values_json)
                max_index = len(temporal_values) - 1
                
                if props.temporal_current_index < max_index:
                    props.temporal_current_index += 1
                elif props.temporal_loop_animation:
                    props.temporal_current_index = 0
                else:
                    props.temporal_animation_playing = False
                    self.cancel(context)
                    return {'CANCELLED'}
                
            except Exception as e:
                print(f"Animation error: {e}")
                props.temporal_animation_playing = False
                self.cancel(context)
                return {'CANCELLED'}
        
        return {'PASS_THROUGH'}
    
    def execute(self, context):
        props = context.scene.scigraphs
        
        if not props.temporal_graph_loaded:
            self.report({'WARNING'}, "No temporal graph loaded")
            return {'CANCELLED'}
        
        if props.temporal_animation_playing:
            props.temporal_animation_playing = False
            return {'FINISHED'}
        
        props.temporal_animation_playing = True
        
        fps = max(1, props.temporal_animation_fps)
        interval = 1.0 / fps
        
        wm = context.window_manager
        self._timer = wm.event_timer_add(interval, window=context.window)
        wm.modal_handler_add(self)
        
        return {'RUNNING_MODAL'}
    
    def cancel(self, context):
        if self._timer:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None
        
        if hasattr(context, 'area') and context.area:
            context.area.header_text_set(None)


class SCIGRAPHS_OT_TemporalGoTo(bpy.types.Operator):
    """Jump to a specific time index."""
    bl_idname = "scigraphs.temporal_goto"
    bl_label = "Go To Time"
    bl_description = "Jump to a specific time period"
    bl_options = {'REGISTER', 'UNDO'}
    
    index: IntProperty(
        name="Time Index",
        default=0,
        min=0,
    )
    
    def execute(self, context):
        props = context.scene.scigraphs
        
        if not props.temporal_graph_loaded:
            return {'CANCELLED'}
        
        try:
            temporal_values = json.loads(props.temporal_values_json)
            max_index = len(temporal_values) - 1
            props.temporal_current_index = min(max(0, self.index), max_index)
        except:
            pass
        
        return {'FINISHED'}


class SCIGRAPHS_OT_TemporalRefresh(bpy.types.Operator):
    """Refresh the temporal visualization."""
    bl_idname = "scigraphs.temporal_refresh"
    bl_label = "Refresh Temporal View"
    bl_description = "Manually refresh the graph for the current time period"
    bl_options = {'REGISTER'}
    
    def execute(self, context):
        props = context.scene.scigraphs
        
        if not props.temporal_graph_loaded:
            self.report({'WARNING'}, "No temporal graph loaded")
            return {'CANCELLED'}
        
        graph_obj = context.scene.objects.get("SciGraph_Object")
        if not graph_obj:
            graph_obj = context.scene.objects.get("SciGraph_Geospatial")
        
        if not graph_obj:
            self.report({'ERROR'}, "No graph object found")
            return {'CANCELLED'}
        
        try:
            temporal_values = json.loads(props.temporal_values_json)
            current_index = min(props.temporal_current_index, len(temporal_values) - 1)
            current_time = temporal_values[current_index]
            
            from ....core import temporal_filter
            success = temporal_filter.update_graph_for_time(graph_obj, current_time, context)
            
            if success:
                self.report({'INFO'}, f"Updated to time: {current_time}")
            else:
                self.report({'WARNING'}, "Update failed")
            
        except Exception as e:
            self.report({'ERROR'}, f"Error: {e}")
            return {'CANCELLED'}
        
        return {'FINISHED'}


class SCIGRAPHS_OT_TemporalClear(bpy.types.Operator):
    """Clear temporal data and show all time periods."""
    bl_idname = "scigraphs.temporal_clear"
    bl_label = "Show All Time"
    bl_description = "Clear temporal filter and show all data"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        from ....core import temporal_filter
        
        temporal_filter.clear_temporal_state(context)
        
        graph_obj = context.scene.objects.get("SciGraph_Object")
        if not graph_obj:
            graph_obj = context.scene.objects.get("SciGraph_Geospatial")
        
        if graph_obj:
            mesh = graph_obj.data
            if "temporal_visible" in mesh.attributes:
                visible_attr = mesh.attributes["temporal_visible"]
                for i in range(len(visible_attr.data)):
                    visible_attr.data[i].value = 1.0
                mesh.update()
        
        self.report({'INFO'}, "Showing all time periods")
        return {'FINISHED'}


classes = [
    SCIGRAPHS_OT_AnalyzeTimeColumn,
    SCIGRAPHS_OT_CreateTemporalGraphs,
    SCIGRAPHS_OT_TemporalNext,
    SCIGRAPHS_OT_TemporalPrevious,
    SCIGRAPHS_OT_TemporalFirst,
    SCIGRAPHS_OT_TemporalLast,
    SCIGRAPHS_OT_TemporalPlay,
    SCIGRAPHS_OT_TemporalGoTo,
    SCIGRAPHS_OT_TemporalRefresh,
    SCIGRAPHS_OT_TemporalClear,
]


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)

