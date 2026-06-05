# Data import and source configuration panels

import bpy

class SCIGRAPHS_PT_data(bpy.types.Panel):
    """Main data panel for importing and configuring data sources."""
    bl_label = "Data"
    bl_parent_id = "SCIGRAPHS_PT_main"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        props = context.scene.scigraphs

        layout.use_property_split = True
        layout.use_property_decorate = False

        # Step 1: Select Data Source
        box = layout.box()
        box.label(text="Step 1: Select Data Source", icon='FILE_FOLDER')
        
        # Data source toggle. Disable property-split so each enum button keeps
        # its label (with split on, the first button renders without text).
        toggle_col = box.column(align=True)
        toggle_col.use_property_split = False
        row = toggle_col.row(align=True)
        row.prop_enum(props, "data_source", 'FILE')
        row.prop_enum(props, "data_source", 'DATABASE')
        row.prop_enum(props, "data_source", 'SUITESPARSE')
        row.prop_enum(props, "data_source", 'REPRO')
        
        box.separator()
        
        if props.data_source == 'FILE':
            # File-based import UI
            box.prop(props, "filepath", text="")

            from ....core import importer
            if importer.is_graph_file(props.filepath):
                box.label(text="Native graph file detected (GEXF)", icon='NETWORK_DRIVE')
            else:
                box.prop(props, "csv_delimiter", text="Delimiter")
            
            row = box.row()
            row.scale_y = 1.2
            row.operator("scigraphs.load_columns", text="Load File", icon='IMPORT')
        
        elif props.data_source == 'DATABASE':
            # Database import UI
            self._draw_database_ui(box, props)
        
        elif props.data_source == 'SUITESPARSE':
            # SuiteSparse import UI
            self._draw_suitesparse_ui(box, props)

        elif props.data_source == 'REPRO':
            # Reproducible pipeline UI
            self._draw_repro_ui(context, box)
        
        # SuiteSparse and reproducible pipelines have their own streamlined flows.
        if props.data_source not in {'SUITESPARSE', 'REPRO'}:
            # Step 2: Define edges
            layout.separator()
            box = layout.box()
            box.label(text="Step 2: Define Graph Structure", icon='OUTLINER')
            
            col = box.column(align=True)
            col.prop(props, "source_column", text="Source Nodes")
            col.prop(props, "target_column", text="Target Nodes")
            
            box.separator()
            col = box.column()
            col.prop(props, "is_directed", text="Directed Graph", toggle=True)
            col.prop(props, "remove_self_loops", text="Remove Self-Loops")
            
            has_data = props.filepath if props.data_source == 'FILE' else props.sql_columns_cache
            
            if has_data:
                box.separator()
                box.prop(props, "weight_column", text="Weight Column")
            
            if len(props.available_csv_columns) > 0:
                box.separator()
                self._draw_attribute_checklist(box, props)
            
            # Step 3: Create graph
            layout.separator()
            box = layout.box()
            box.label(text="Step 3: Create Graph Object", icon='MESH_DATA')

            if props.data_source != 'SUITESPARSE':
                layout_box = box.box()
                layout_box.prop(props, "auto_layout_on_import")
                if props.auto_layout_on_import:
                    layout_box.prop(props, "layout_algorithm", text="Default Layout")
                    layout_box.prop(props, "layout_scale", text="Scale")
                    if props.use_geospatial:
                        layout_box.label(text="Skipped in geospatial mode", icon='INFO')

            row = box.row()
            row.scale_y = 1.5
            
            if props.data_source == 'FILE':
                row.operator("scigraphs.create_graph", text="Create Graph", icon='ADD')
            else:
                row.operator("scigraphs.create_graph_from_sql", text="Create Graph", icon='ADD')
        
        # Import Node Attributes (requires existing graph)
        obj = context.active_object
        if obj is not None and obj.type == 'MESH' and "num_nodes" in obj:
            layout.separator()
            self._draw_node_attribute_import(layout, props, obj)

        # Step 4: Setup visualization
        if props.data_source == 'REPRO':
            return

        layout.separator()
        box = layout.box()
        box.label(text="Step 4: Apply Visualization", icon='GEOMETRY_NODES')
        box.label(text="Add Geometry Nodes for rendering")
        row = box.row()
        row.scale_y = 1.3
        row.operator("scigraphs.setup_visualization", text="Setup Visual", icon='SHADING_RENDERED')
    
    def _draw_attribute_checklist(self, box, props):
        """Draw the column attribute checklist for selecting which columns to import."""
        try:
            source_idx = int(props.source_column)
        except (ValueError, TypeError):
            source_idx = -1
        try:
            target_idx = int(props.target_column)
        except (ValueError, TypeError):
            target_idx = -1
        
        attribute_columns = [
            (idx, item) for idx, item in enumerate(props.available_csv_columns)
            if idx != source_idx and idx != target_idx
        ]
        
        attr_box = box.box()
        
        header = attr_box.row()
        header.label(text="Import as Attributes", icon='SPREADSHEET')
        
        if not attribute_columns:
            info = attr_box.row()
            info.scale_y = 0.7
            info.label(text="No extra columns beyond Source/Target", icon='INFO')
            return
        
        toggle_row = header.row(align=True)
        toggle_row.scale_x = 0.6
        op_all = toggle_row.operator("scigraphs.toggle_all_attributes", text="All")
        op_all.enable = True
        op_none = toggle_row.operator("scigraphs.toggle_all_attributes", text="None")
        op_none.enable = False
        
        col = attr_box.column(align=True)
        for _idx, item in attribute_columns:
            row = col.row(align=True)
            row.prop(item, "import_as_attribute", text="")
            row.label(text=item.name)
            
            if item.column_type == "numeric":
                row.label(text="", icon='TRACKING')
            elif item.column_type == "datetime":
                row.label(text="", icon='TIME')
            else:
                row.label(text="", icon='FONT_DATA')
        
        selected_count = sum(1 for _, item in attribute_columns if item.import_as_attribute)
        footer = attr_box.row()
        footer.scale_y = 0.7
        footer.label(text=f"{selected_count} of {len(attribute_columns)} columns selected")
    
    def _draw_node_attribute_import(self, layout, props, obj):
        """Draw the section for importing external vertex-only attribute files."""
        box = layout.box()
        box.label(text="Import Node Attributes", icon='SPREADSHEET')
        
        info = box.row()
        info.scale_y = 0.7
        num_nodes = obj.get("num_nodes", 0)
        info.label(text=f"Active graph: {num_nodes} nodes", icon='INFO')
        
        col = box.column(align=True)
        col.prop(props, "node_attr_filepath", text="")
        
        row = col.row(align=True)
        row.prop(props, "node_attr_delimiter", text="Delimiter")
        row.prop(props, "node_attr_has_header", text="Header", toggle=True)
        
        row = col.row()
        row.scale_y = 1.3
        row.operator("scigraphs.import_node_attributes", text="Import Node Attributes", icon='IMPORT')
        
        hint = box.row()
        hint.scale_y = 0.6
        hint.label(text="Format: node_name <sep> value(s). Missing nodes get NaN.")

    def _draw_suitesparse_ui(self, box, props):
        """Draw the SuiteSparse Matrix Collection import UI."""
        col = box.column(align=True)
        
        # Matrix identifier input
        col.label(text="Matrix Identifier:", icon='URL')
        col.prop(props, "suitesparse_id", text="")
        
        # Help text
        info = col.box()
        info.scale_y = 0.7
        info.label(text="Format: Group/Name  (e.g. Grund/bayer09)")
        
        col.separator()
        
        # Graph mode
        col.label(text="Graph Representation:", icon='MESH_DATA')
        col.prop(props, "suitesparse_mode", text="")
        
        # Mode description
        desc = col.box()
        desc.scale_y = 0.7
        if props.suitesparse_mode == 'BIPARTITE':
            desc.label(text="Bipartite: rows + columns as separate nodes")
            desc.label(text="Preserves matrix structure (elongated layouts)")
        else:
            desc.label(text="Symmetric: standard adjacency graph")
            desc.label(text="Denser, rounder layouts")
        
        col.separator()
        
        # Giant component toggle
        col.prop(props, "suitesparse_giant_only")

        col.separator()
        layout_box = col.box()
        layout_box.prop(props, "auto_layout_on_import")
        if props.auto_layout_on_import:
            layout_box.prop(props, "layout_algorithm", text="Default Layout")
            layout_box.prop(props, "layout_scale", text="Scale")
        
        col.separator()
        
        # Download button
        row = col.row()
        row.scale_y = 1.5
        row.operator("scigraphs.download_suitesparse", text="Download & Create Graph", icon='IMPORT')
        
        # Browse button
        row = col.row()
        row.operator("scigraphs.browse_suitesparse", text="Browse Collection", icon='URL')
        
        # Status
        if props.suitesparse_status:
            status_box = col.box()
            status_box.scale_y = 0.7
            status_box.label(text=props.suitesparse_status, icon='INFO')

    def _draw_repro_ui(self, context, box):
        """Draw reproducible pipeline import controls inside the Data panel."""
        repro = context.scene.scigraphs_repro

        col = box.column(align=True)
        col.label(text="Pipeline File:", icon='FILE_CACHE')
        col.prop(repro, "pipeline_path", text="")

        row = col.row(align=True)
        row.scale_y = 1.2
        row.operator("scigraphs.validate_pipeline", text="Validate", icon='CHECKMARK')
        row.operator("scigraphs.run_pipeline", text="Run Pipeline", icon='PLAY')

        col.separator()

        template_box = col.box()
        template_box.label(text="Templates", icon='FILE_NEW')
        template_box.operator("scigraphs.export_pipeline_template", text="Export Template", icon='EXPORT')
        template_box.operator("scigraphs.export_current_repro_spec", text="Export Current Scene", icon='SCENE_DATA')

        col.separator()

        artifact_box = col.box()
        artifact_box.label(text="Artifacts", icon='FOLDER_REDIRECT')
        artifact_box.prop(repro, "artifacts_path", text="")
        artifact_box.operator("scigraphs.open_artifacts_folder", text="Open Folder", icon='FILEBROWSER')

        obj = context.active_object
        if obj is not None and "num_nodes" in obj:
            info = col.box()
            info.label(text="Current Graph", icon='OUTLINER_OB_MESH')
            info.label(text=f"Nodes: {obj['num_nodes']}")
            info.label(text=f"Edges: {obj.get('num_edges', 'N/A')}")
    
    def _draw_database_ui(self, box, props):
        """Draw the database import UI."""
        from ....preferences import get_preferences
        prefs = get_preferences()
        
        # Connection profile selector
        col = box.column(align=True)
        
        if prefs and prefs.db_profiles:
            row = col.row(align=True)
            row.prop(props, "db_profile_index", text="Connection")
            row.operator("scigraphs.open_db_preferences", text="", icon='PREFERENCES')
            
            # Show connection test button
            row = col.row(align=True)
            row.operator("scigraphs.test_sql_connection", text="Test", icon='PLUGIN')
            row.operator("scigraphs.open_db_preferences", text="Add New", icon='ADD')
        else:
            # No profiles configured
            info = col.box()
            info.scale_y = 0.7
            info.label(text="No database connections configured", icon='INFO')
            col.operator("scigraphs.open_db_preferences", text="Configure Databases", icon='PREFERENCES')
        
        col.separator()
        
        # SQL Query input
        col.label(text="SQL Query:", icon='TEXT')
        col.prop(props, "sql_query", text="")
        
        # Query actions
        row = col.row(align=True)
        row.scale_y = 1.2
        row.operator("scigraphs.load_sql_columns", text="Load Columns", icon='IMPORT')
        row.operator("scigraphs.preview_sql_query", text="Preview", icon='HIDE_OFF')
        
        # Query status
        if props.sql_query_status:
            status_box = col.box()
            status_box.scale_y = 0.7
            status_box.label(text=props.sql_query_status, icon='INFO')
        
        # Show loaded columns count
        if props.sql_columns_cache:
            columns = props.sql_columns_cache.split('|')
            info = col.box()
            info.scale_y = 0.7
            info.label(text=f"Loaded {len(columns)} columns")
        
        # Security notice
        sec_box = col.box()
        sec_box.scale_y = 0.6
        sec_box.label(text="Only SELECT queries allowed", icon='LOCKED')


class SCIGRAPHS_PT_data_geospatial(bpy.types.Panel):
    """Geospatial options for geographic data."""
    bl_label = "Geospatial Options"
    bl_parent_id = "SCIGRAPHS_PT_data"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        props = context.scene.scigraphs
        # Show if file is loaded or SQL columns are available
        if props.data_source == 'FILE':
            return bool(props.filepath)
        return bool(props.sql_columns_cache)

    def draw_header(self, context):
        props = context.scene.scigraphs
        self.layout.prop(props, "use_geospatial", text="")

    def draw(self, context):
        layout = self.layout
        props = context.scene.scigraphs
        
        layout.active = props.use_geospatial
        
        if props.use_geospatial:
            col = layout.column(align=True)
            col.label(text="Coordinate Source:", icon='EMPTY_AXIS')
            
            # Geocoding option (prominently displayed)
            row = col.row()
            row.prop(props, "geocode_columns", 
                     text="Auto-Geocode Source/Target as Countries/Cities")
            
            # Show info box if geocoding is enabled
            if props.geocode_columns:
                info_box = col.box()
                info_box.scale_y = 0.7
                info_box.label(text="Node names will be geocoded to coordinates", icon='INFO')
                info_box.label(text="(First run may take time, then cached)")
            
            # Coordinate columns (only if NOT geocoding)
            if not props.geocode_columns:
                col.separator()
                col.label(text="OR use explicit coordinate columns:")
                subcol = col.column(align=True)
                subcol.prop(props, "latitude_column", text="Latitude")
                subcol.prop(props, "longitude_column", text="Longitude")
            
            col.separator()
            
            subcol = col.column(align=True)
            subcol.prop(props, "show_globe", text="Show Earth Globe")
            if props.show_globe:
                subcol.prop(props, "globe_radius", text="Globe Radius")
                subcol.prop(props, "globe_subdivisions", text="Globe Quality")
                
                subcol.separator()
                subcol.label(text="Globe Appearance:", icon='SHADING_RENDERED')
                
                subcol.prop(props, "globe_theme_api", text="Globe Theme")
                
                if props.globe_theme_api != 'NONE':
                    theme_col = subcol.column(align=True)
                    theme_col.prop(props, "globe_texture_resolution", text="Texture Quality")
                    
                    theme_col.separator()
                    theme_col.label(text="PBR Settings:")
                    theme_col.prop(props, "globe_water_specular", text="Water Shine")
                    theme_col.prop(props, "globe_water_roughness", text="Water Roughness")
                    theme_col.prop(props, "globe_land_roughness", text="Land Roughness")
                    theme_col.prop(props, "globe_bump_strength", text="Surface Relief")
                    
                    info_box = subcol.box()
                    info_box.scale_y = 0.6
                    if props.globe_theme_api == 'NASA_BLUE_MARBLE':
                        info_box.label(text="NASA Blue Marble satellite imagery", icon='INFO')
                    elif props.globe_theme_api == 'NASA_VIIRS':
                        info_box.label(text="Earth at night - city lights", icon='LIGHT')
                    elif props.globe_theme_api == 'URBAN_DARK':
                        info_box.label(text="Dark theme with urban glow", icon='GHOST_ENABLED')
                    elif props.globe_theme_api == 'TOPOGRAPHIC_SHADED':
                        info_box.label(text="Elevation-based shading", icon='RNDCURVE')
                    elif props.globe_theme_api == 'DATA_OVERLAY':
                        info_box.label(text="Transparent for data viz", icon='MOD_MASK')
                    
                    subcol.separator()
                    row = subcol.row(align=True)
                    row.operator("scigraphs.download_globe_texture", text="Pre-download", icon='IMPORT')
                    row.operator("scigraphs.clear_texture_cache", text="Clear Cache", icon='TRASH')
                else:
                    subcol.prop(props, "globe_material", text="Material Style")
                    
                    if props.globe_material == 'WORLD_MAP':
                        map_col = subcol.column(align=True)
                        map_col.prop(props, "map_resolution", text="Map Detail")
                        map_col.prop(props, "map_feature_type", text="Map Type")
            
            col.separator()
            col.prop(props, "edge_style", text="Edge Style")


class SCIGRAPHS_PT_data_temporal(bpy.types.Panel):
    """Temporal data configuration."""
    bl_label = "Temporal Data"
    bl_parent_id = "SCIGRAPHS_PT_data"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        props = context.scene.scigraphs
        # Show if file is loaded or SQL columns are available
        if props.data_source == 'FILE':
            return bool(props.filepath)
        return bool(props.sql_columns_cache)

    def draw_header(self, context):
        props = context.scene.scigraphs
        self.layout.prop(props, "has_temporal_data", text="")

    def draw(self, context):
        layout = self.layout
        props = context.scene.scigraphs
        
        layout.active = props.has_temporal_data
        
        if props.has_temporal_data:
            col = layout.column(align=True)
            col.prop(props, "time_column", text="Time Column")
            col.prop(props, "time_aggregation", text="Aggregation")
            
            if props.time_aggregation == 'RANGE':
                subcol = col.column(align=True)
                subcol.prop(props, "time_range_start", text="From")
                subcol.prop(props, "time_range_end", text="To")
            
            layout.separator()
            box = layout.box()
            box.label(text="Temporal Analysis", icon='TIME')
            row = box.row()
            row.scale_y = 1.2
            row.operator("scigraphs.analyze_time_column", text="Analyze Time Column", icon='VIEWZOOM')
            
            if getattr(props, "temporal_analyzed", False):
                row = box.row()
                row.scale_y = 1.2
                row.operator("scigraphs.create_temporal_graphs", text="Create Temporal Graphs", icon='ADD')
            
            if getattr(props, "temporal_graph_loaded", False):
                layout.separator()
                box = layout.box()
                box.label(text="Playback", icon='PLAY')
                
                row = box.row(align=True)
                row.operator("scigraphs.temporal_first", text="", icon='REW')
                row.operator("scigraphs.temporal_previous", text="", icon='PLAY_REVERSE')
                row.operator("scigraphs.temporal_play", text="", icon='PLAY')
                row.operator("scigraphs.temporal_next", text="", icon='FF')
                row.operator("scigraphs.temporal_last", text="", icon='NEXT_KEYFRAME')
                
                row = box.row(align=True)
                row.operator("scigraphs.temporal_refresh", text="Refresh", icon='FILE_REFRESH')
                row.operator("scigraphs.temporal_clear", text="Clear", icon='X')


def register():
    bpy.utils.register_class(SCIGRAPHS_PT_data)
    bpy.utils.register_class(SCIGRAPHS_PT_data_geospatial)
    bpy.utils.register_class(SCIGRAPHS_PT_data_temporal)


def unregister():
    bpy.utils.unregister_class(SCIGRAPHS_PT_data_temporal)
    bpy.utils.unregister_class(SCIGRAPHS_PT_data_geospatial)
    bpy.utils.unregister_class(SCIGRAPHS_PT_data)

