# Export and utility operators

import bpy
from ....core import export_utils, statistics
from ....core.mesh.mesh_utils import parse_graph_data_filtered, collect_mesh_attributes


class SCIGRAPHS_OT_ExportGraph(bpy.types.Operator):
    """Export graph to various formats."""
    bl_idname = "scigraphs.export_graph"
    bl_label = "Export Graph"
    bl_description = "Export graph data to selected format"
    
    def execute(self, context):
        props = context.scene.scigraphs
        obj = context.active_object
        
        if not obj or "num_nodes" not in obj:
            self.report({'ERROR'}, "No graph object selected")
            return {'CANCELLED'}
        
        if not props.export_filepath:
            self.report({'ERROR'}, "Please specify export file path")
            return {'CANCELLED'}
        
        graph_data = parse_graph_data_filtered(obj)
        
        node_attributes = None
        if props.export_include_attributes:
            node_attributes = collect_mesh_attributes(obj)
        
        export_format = props.export_format
        filepath = bpy.path.abspath(props.export_filepath)
        
        if export_format == 'GRAPHML' and not filepath.endswith('.graphml'):
            filepath += '.graphml'
        elif export_format == 'GEXF' and not filepath.endswith('.gexf'):
            filepath += '.gexf'
        elif export_format == 'JSON' and not filepath.endswith('.json'):
            filepath += '.json'
        elif export_format == 'CSV' and not filepath.endswith('.csv'):
            filepath += '.csv'
        
        success = False
        if export_format == 'GRAPHML':
            success = export_utils.export_to_graphml(graph_data, filepath, node_attributes)
        elif export_format == 'GEXF':
            success = export_utils.export_to_gexf(graph_data, filepath, node_attributes)
        elif export_format == 'JSON':
            success = export_utils.export_to_json(graph_data, filepath, node_attributes)
        elif export_format == 'CSV':
            success = export_utils.export_to_csv_edges(graph_data, filepath)
        
        if success:
            self.report({'INFO'}, f"Graph exported to {filepath}")
        else:
            self.report({'ERROR'}, "Export failed")
        
        return {'FINISHED' if success else 'CANCELLED'}


class SCIGRAPHS_OT_ExportPositions(bpy.types.Operator):
    """Export node positions to CSV."""
    bl_idname = "scigraphs.export_positions"
    bl_label = "Export Positions"
    bl_description = "Export node positions to CSV file"
    
    def execute(self, context):
        props = context.scene.scigraphs
        obj = context.active_object
        
        if not obj or "num_nodes" not in obj:
            self.report({'ERROR'}, "No graph object selected")
            return {'CANCELLED'}
        
        if not props.export_filepath:
            self.report({'ERROR'}, "Please specify export file path")
            return {'CANCELLED'}
        
        filepath = bpy.path.abspath(props.export_filepath)
        if not filepath.endswith('.csv'):
            filepath += '_positions.csv'
        
        success = export_utils.export_positions(obj, filepath)
        
        if success:
            self.report({'INFO'}, f"Positions exported to {filepath}")
        else:
            self.report({'ERROR'}, "Export failed")
        
        return {'FINISHED' if success else 'CANCELLED'}


class SCIGRAPHS_OT_GenerateStatisticsReport(bpy.types.Operator):
    """Generate comprehensive statistics report."""
    bl_idname = "scigraphs.generate_statistics_report"
    bl_label = "Generate Statistics Report"
    bl_description = "Generate text report with all graph statistics"
    
    def execute(self, context):
        props = context.scene.scigraphs
        obj = context.active_object
        
        if not obj or "num_nodes" not in obj:
            self.report({'ERROR'}, "No graph object selected")
            return {'CANCELLED'}
        
        graph_data = parse_graph_data_filtered(obj)
        
        self.report({'INFO'}, "Calculating statistics...")
        stats = statistics.calculate_all_statistics(graph_data)
        
        if props.report_include_powerlaw:
            deg_dist = stats['degree_distribution']
            if 'degrees' in deg_dist:
                power_law = statistics.power_law_fit(deg_dist['degrees'])
                stats['power_law'] = power_law
        
        if props.export_filepath:
            filepath = bpy.path.abspath(props.export_filepath)
            if not filepath.endswith('.txt'):
                filepath += '_report.txt'
        else:
            filepath = bpy.path.abspath("//graph_statistics_report.txt")
        
        success = export_utils.export_statistics_report(stats, filepath)
        
        if success:
            self.report({'INFO'}, f"Statistics report generated: {filepath}")
        else:
            self.report({'ERROR'}, "Report generation failed")
        
        return {'FINISHED' if success else 'CANCELLED'}


class SCIGRAPHS_OT_ExportGEXF(bpy.types.Operator):
    """Export graph to GEXF format."""
    bl_idname = "scigraphs.export_gexf"
    bl_label = "Export to GEXF"
    bl_description = "Export graph to GEXF format"
    
    def execute(self, context):
        props = context.scene.scigraphs
        obj = context.active_object
        
        if not obj or "num_nodes" not in obj:
            self.report({'ERROR'}, "No graph object selected")
            return {'CANCELLED'}
        
        if not props.export_filepath:
            self.report({'ERROR'}, "Please specify export file path")
            return {'CANCELLED'}
        
        graph_data = parse_graph_data_filtered(obj)
        node_attributes = collect_mesh_attributes(obj) if props.export_include_attributes else None
        
        filepath = bpy.path.abspath(props.export_filepath)
        if not filepath.endswith('.gexf'):
            filepath += '.gexf'
        
        success = export_utils.export_to_gexf(graph_data, filepath, node_attributes)
        
        if success:
            self.report({'INFO'}, f"Graph exported to {filepath}")
        else:
            self.report({'ERROR'}, "Export failed")
        
        return {'FINISHED' if success else 'CANCELLED'}


class SCIGRAPHS_OT_ExportGraphML(bpy.types.Operator):
    """Export graph to GraphML format."""
    bl_idname = "scigraphs.export_graphml"
    bl_label = "Export to GraphML"
    bl_description = "Export graph to GraphML format"
    
    def execute(self, context):
        props = context.scene.scigraphs
        obj = context.active_object
        
        if not obj or "num_nodes" not in obj:
            self.report({'ERROR'}, "No graph object selected")
            return {'CANCELLED'}
        
        if not props.export_filepath:
            self.report({'ERROR'}, "Please specify export file path")
            return {'CANCELLED'}
        
        graph_data = parse_graph_data_filtered(obj)
        node_attributes = collect_mesh_attributes(obj) if props.export_include_attributes else None
        
        filepath = bpy.path.abspath(props.export_filepath)
        if not filepath.endswith('.graphml'):
            filepath += '.graphml'
        
        success = export_utils.export_to_graphml(graph_data, filepath, node_attributes)
        
        if success:
            self.report({'INFO'}, f"Graph exported to {filepath}")
        else:
            self.report({'ERROR'}, "Export failed")
        
        return {'FINISHED' if success else 'CANCELLED'}


class SCIGRAPHS_OT_ExportPajek(bpy.types.Operator):
    """Export graph to Pajek NET format."""
    bl_idname = "scigraphs.export_pajek"
    bl_label = "Export to Pajek"
    bl_description = "Export graph to Pajek NET format"
    
    def execute(self, context):
        props = context.scene.scigraphs
        obj = context.active_object
        
        if not obj or "num_nodes" not in obj:
            self.report({'ERROR'}, "No graph object selected")
            return {'CANCELLED'}
        
        if not props.export_filepath:
            self.report({'ERROR'}, "Please specify export file path")
            return {'CANCELLED'}
        
        graph_data = parse_graph_data_filtered(obj)
        
        filepath = bpy.path.abspath(props.export_filepath)
        if not filepath.endswith('.net'):
            filepath += '.net'
        
        success = export_utils.export_to_pajek(graph_data, filepath)
        
        if success:
            self.report({'INFO'}, f"Graph exported to {filepath}")
        else:
            self.report({'ERROR'}, "Export failed")
        
        return {'FINISHED' if success else 'CANCELLED'}


class SCIGRAPHS_OT_ExportJSON(bpy.types.Operator):
    """Export graph to JSON format."""
    bl_idname = "scigraphs.export_json"
    bl_label = "Export to JSON"
    bl_description = "Export graph to JSON format"
    
    def execute(self, context):
        props = context.scene.scigraphs
        obj = context.active_object
        
        if not obj or "num_nodes" not in obj:
            self.report({'ERROR'}, "No graph object selected")
            return {'CANCELLED'}
        
        if not props.export_filepath:
            self.report({'ERROR'}, "Please specify export file path")
            return {'CANCELLED'}
        
        graph_data = parse_graph_data_filtered(obj)
        node_attributes = collect_mesh_attributes(obj) if props.export_include_attributes else None
        
        filepath = bpy.path.abspath(props.export_filepath)
        if not filepath.endswith('.json'):
            filepath += '.json'
        
        success = export_utils.export_to_json(graph_data, filepath, node_attributes)
        
        if success:
            self.report({'INFO'}, f"Graph exported to {filepath}")
        else:
            self.report({'ERROR'}, "Export failed")
        
        return {'FINISHED' if success else 'CANCELLED'}


class SCIGRAPHS_OT_CalculateGlobalStatistics(bpy.types.Operator):
    """Calculate and display global graph statistics."""
    bl_idname = "scigraphs.calculate_global_statistics"
    bl_label = "Calculate Global Statistics"
    bl_description = "Calculate global graph metrics and store in object properties"
    
    def execute(self, context):
        obj = context.active_object
        
        if not obj or "num_nodes" not in obj:
            self.report({'ERROR'}, "No graph object selected")
            return {'CANCELLED'}
        
        graph_data = parse_graph_data_filtered(obj)
        
        self.report({'INFO'}, "Calculating statistics...")
        stats = statistics.calculate_all_statistics(graph_data)
        
        obj["stat_density"] = stats['density']
        obj["stat_global_clustering"] = stats['global_clustering']
        obj["stat_diameter"] = stats['diameter']
        obj["stat_avg_path_length"] = stats['average_path_length']
        obj["stat_assortativity"] = stats['assortativity']
        
        deg_dist = stats['degree_distribution']
        obj["stat_degree_mean"] = deg_dist['mean']
        obj["stat_degree_median"] = deg_dist['median']
        obj["stat_degree_std"] = deg_dist['std']
        obj["stat_degree_min"] = deg_dist['min']
        obj["stat_degree_max"] = deg_dist['max']
        
        self.report({'INFO'}, 
            f"Statistics calculated: density={stats['density']:.3f}, "
            f"clustering={stats['global_clustering']:.3f}, "
            f"diameter={stats['diameter']}")
        
        return {'FINISHED'}


def register():
    bpy.utils.register_class(SCIGRAPHS_OT_ExportGraph)
    bpy.utils.register_class(SCIGRAPHS_OT_ExportGEXF)
    bpy.utils.register_class(SCIGRAPHS_OT_ExportGraphML)
    bpy.utils.register_class(SCIGRAPHS_OT_ExportPajek)
    bpy.utils.register_class(SCIGRAPHS_OT_ExportJSON)
    bpy.utils.register_class(SCIGRAPHS_OT_ExportPositions)
    bpy.utils.register_class(SCIGRAPHS_OT_GenerateStatisticsReport)
    bpy.utils.register_class(SCIGRAPHS_OT_CalculateGlobalStatistics)


def unregister():
    bpy.utils.unregister_class(SCIGRAPHS_OT_CalculateGlobalStatistics)
    bpy.utils.unregister_class(SCIGRAPHS_OT_GenerateStatisticsReport)
    bpy.utils.unregister_class(SCIGRAPHS_OT_ExportPositions)
    bpy.utils.unregister_class(SCIGRAPHS_OT_ExportJSON)
    bpy.utils.unregister_class(SCIGRAPHS_OT_ExportPajek)
    bpy.utils.unregister_class(SCIGRAPHS_OT_ExportGraphML)
    bpy.utils.unregister_class(SCIGRAPHS_OT_ExportGEXF)
    bpy.utils.unregister_class(SCIGRAPHS_OT_ExportGraph)
