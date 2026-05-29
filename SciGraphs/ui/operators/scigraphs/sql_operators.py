# SQL Import operators for SciGraphs
#
# Operators for loading graph data from SQL databases.

import bpy
from ....core import sql_importer, db_connector, geometry, graph
from ....preferences import get_preferences
from ...view_utils import focus_graph_in_top_view


class SCIGRAPHS_OT_LoadSQLColumns(bpy.types.Operator):
    """Execute SQL query to load column names."""
    bl_idname = "scigraphs.load_sql_columns"
    bl_label = "Load Columns from Query"
    bl_description = "Execute the SQL query to retrieve column names"
    
    def execute(self, context):
        props = context.scene.scigraphs
        
        # Get the selected database profile
        profile = self._get_profile(props)
        if profile is None:
            self.report({'WARNING'}, "No database profile selected. Configure in Preferences.")
            return {'CANCELLED'}
        
        # Check if query is provided
        if not props.sql_query or not props.sql_query.strip():
            self.report({'WARNING'}, "Please enter a SQL query")
            return {'CANCELLED'}
        
        # Get columns from query
        columns = sql_importer.get_columns_from_query(profile, props.sql_query)
        
        if not columns:
            props.sql_query_status = "Query returned no columns or failed"
            self.report({'ERROR'}, "Could not load columns from query")
            return {'CANCELLED'}
        
        # Store columns in cache (pipe-separated)
        props.sql_columns_cache = "|".join(columns)
        props.sql_query_status = f"Loaded {len(columns)} columns"
        
        self.report({'INFO'}, f"Loaded {len(columns)} columns from query")
        return {'FINISHED'}
    
    def _get_profile(self, props):
        """Get the selected database profile."""
        prefs = get_preferences()
        if not prefs or not prefs.db_profiles:
            return None
        
        try:
            idx = int(props.db_profile_index)
            if 0 <= idx < len(prefs.db_profiles):
                return prefs.db_profiles[idx]
        except (ValueError, TypeError):
            pass
        
        return None


class SCIGRAPHS_OT_PreviewSQLQuery(bpy.types.Operator):
    """Preview the results of the SQL query."""
    bl_idname = "scigraphs.preview_sql_query"
    bl_label = "Preview Query"
    bl_description = "Execute the query and preview the first few rows"
    
    def execute(self, context):
        props = context.scene.scigraphs
        
        profile = self._get_profile(props)
        if profile is None:
            self.report({'WARNING'}, "No database profile selected")
            return {'CANCELLED'}
        
        if not props.sql_query or not props.sql_query.strip():
            self.report({'WARNING'}, "Please enter a SQL query")
            return {'CANCELLED'}
        
        # Preview the query
        df, total_rows, error = sql_importer.preview_query(
            profile, props.sql_query, max_rows=10
        )
        
        if error:
            props.sql_query_status = f"Error: {error}"
            self.report({'ERROR'}, f"Query error: {error}")
            return {'CANCELLED'}
        
        if df is None or len(df) == 0:
            props.sql_query_status = "Query returned no results"
            self.report({'WARNING'}, "Query returned no results")
            return {'CANCELLED'}
        
        # Update status and cache columns
        props.sql_columns_cache = "|".join(df.columns)
        props.sql_row_count = total_rows
        props.sql_query_status = f"Preview: {len(df)} of {total_rows} rows, {len(df.columns)} columns"
        
        # Print preview to console
        print("\n=== SQL Query Preview ===")
        print(f"Columns: {list(df.columns)}")
        print(f"Total rows: {total_rows}")
        print(df.to_string())
        print("========================\n")
        
        self.report({'INFO'}, f"Query preview: {total_rows} rows, {len(df.columns)} columns")
        return {'FINISHED'}
    
    def _get_profile(self, props):
        """Get the selected database profile."""
        prefs = get_preferences()
        if not prefs or not prefs.db_profiles:
            return None
        
        try:
            idx = int(props.db_profile_index)
            if 0 <= idx < len(prefs.db_profiles):
                return prefs.db_profiles[idx]
        except (ValueError, TypeError):
            pass
        
        return None


class SCIGRAPHS_OT_CreateGraphFromSQL(bpy.types.Operator):
    """Create a graph from SQL query results."""
    bl_idname = "scigraphs.create_graph_from_sql"
    bl_label = "Create Graph from SQL"
    bl_description = "Execute the SQL query and create a graph visualization"
    
    def execute(self, context):
        props = context.scene.scigraphs
        
        profile = self._get_profile(props)
        if profile is None:
            self.report({'WARNING'}, "No database profile selected")
            return {'CANCELLED'}
        
        if not props.sql_query or not props.sql_query.strip():
            self.report({'WARNING'}, "Please enter a SQL query")
            return {'CANCELLED'}
        
        # Get column indices
        try:
            source_col = int(props.source_column)
            target_col = int(props.target_column)
        except (ValueError, TypeError):
            self.report({'ERROR'}, "Invalid column selection")
            return {'CANCELLED'}
        
        # Load graph data from SQL
        if props.use_geospatial:
            # Handle geospatial mode
            lat_col = None
            lon_col = None
            weight_col = None
            
            try:
                lat_col = int(props.latitude_column) if props.latitude_column else None
                lon_col = int(props.longitude_column) if props.longitude_column else None
                weight_col = int(props.weight_column) if props.weight_column else None
            except (ValueError, TypeError):
                pass
            
            graph_data = sql_importer.load_geospatial_graph_from_sql(
                profile,
                props.sql_query,
                source_col,
                target_col,
                lat_col=lat_col,
                lon_col=lon_col,
                weight_col=weight_col,
            )
        else:
            graph_data = sql_importer.load_graph_from_sql(
                profile,
                props.sql_query,
                source_col,
                target_col,
            )
        
        if graph_data is None:
            props.sql_query_status = "Failed to create graph"
            self.report({'ERROR'}, "Could not create graph from SQL query")
            return {'CANCELLED'}
        
        # Create the graph visualization
        graph_obj = geometry.create_graph_object(graph_data, is_directed=props.is_directed)

        if not props.use_geospatial:
            from .data_operators import SCIGRAPHS_AutoLayoutOnImport
            SCIGRAPHS_AutoLayoutOnImport.apply(context, graph_obj, self)
        
        focus_graph_in_top_view(context, graph_obj)
        props.sql_query_status = f"Graph created: {len(graph_data.nodes)} nodes, {len(graph_data.edges)} edges"
        self.report({'INFO'}, f"Graph created: {len(graph_data.nodes)} nodes, {len(graph_data.edges)} edges")
        
        return {'FINISHED'}
    
    def _get_profile(self, props):
        """Get the selected database profile."""
        prefs = get_preferences()
        if not prefs or not prefs.db_profiles:
            return None
        
        try:
            idx = int(props.db_profile_index)
            if 0 <= idx < len(prefs.db_profiles):
                return prefs.db_profiles[idx]
        except (ValueError, TypeError):
            pass
        
        return None


class SCIGRAPHS_OT_TestSQLConnection(bpy.types.Operator):
    """Test the selected database connection from the Data panel."""
    bl_idname = "scigraphs.test_sql_connection"
    bl_label = "Test Connection"
    bl_description = "Test if the database connection works"
    
    def execute(self, context):
        props = context.scene.scigraphs
        
        profile = self._get_profile(props)
        if profile is None:
            self.report({'WARNING'}, "No database profile selected")
            return {'CANCELLED'}
        
        success, message = db_connector.test_connection(profile)
        
        if success:
            props.sql_query_status = f"Connected: {message}"
            self.report({'INFO'}, f"Connection successful: {message}")
        else:
            props.sql_query_status = f"Failed: {message}"
            self.report({'ERROR'}, f"Connection failed: {message}")
        
        return {'FINISHED'}
    
    def _get_profile(self, props):
        """Get the selected database profile."""
        prefs = get_preferences()
        if not prefs or not prefs.db_profiles:
            return None
        
        try:
            idx = int(props.db_profile_index)
            if 0 <= idx < len(prefs.db_profiles):
                return prefs.db_profiles[idx]
        except (ValueError, TypeError):
            pass
        
        return None


class SCIGRAPHS_OT_OpenDBPreferences(bpy.types.Operator):
    """Open addon preferences to configure database connections."""
    bl_idname = "scigraphs.open_db_preferences"
    bl_label = "Configure Databases"
    bl_description = "Open addon preferences to add or edit database connections"
    
    def execute(self, context):
        bpy.ops.screen.userpref_show()
        bpy.context.preferences.active_section = 'ADDONS'
        return {'FINISHED'}


classes = [
    SCIGRAPHS_OT_LoadSQLColumns,
    SCIGRAPHS_OT_PreviewSQLQuery,
    SCIGRAPHS_OT_CreateGraphFromSQL,
    SCIGRAPHS_OT_TestSQLConnection,
    SCIGRAPHS_OT_OpenDBPreferences,
]


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)

