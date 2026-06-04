"""
Proximity Graphs UI panel for City2Graph.

Provides interface for generating single-layer and multi-layer proximity graphs
from OSM feature objects.
"""

import bpy


class SCIGRAPHS_PT_c2g_proximity(bpy.types.Panel):
    """Main Proximity Graphs panel."""
    bl_label = "Proximity Graphs"
    bl_parent_id = "SCIGRAPHS_PT_city2graph_main"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'City2Graph'
    bl_options = {'DEFAULT_CLOSED'}
    
    def draw(self, context):
        layout = self.layout
        props = context.scene.city2graph
        
        layout.use_property_split = True
        layout.use_property_decorate = False
        
        box = layout.box()
        box.label(text="Spatial Network Generation", icon='OUTLINER_OB_MESH')
        
        info_box = box.box()
        info_box.scale_y = 0.7
        col = info_box.column(align=True)
        col.label(text="Generate graphs based on spatial proximity")
        col.label(text="Supports multiple graph types and distance metrics")


class SCIGRAPHS_PT_c2g_proximity_single(bpy.types.Panel):
    """Single-layer proximity graphs panel."""
    bl_label = "Single-Layer Graphs"
    bl_parent_id = "SCIGRAPHS_PT_c2g_proximity"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'City2Graph'
    
    def draw(self, context):
        layout = self.layout
        props = context.scene.city2graph
        
        layout.use_property_split = True
        layout.use_property_decorate = False
        
        box = layout.box()
        box.label(text="Configuration", icon='SETTINGS')
        
        col = box.column(align=True)
        col.prop(props, "prox_feature_object", text="Features")
        
        feature_obj = props.prox_feature_object
        if feature_obj and (feature_obj.get("is_osm_features") or feature_obj.get("is_city2graph")):
            feature_count = feature_obj.get("feature_count", 0)
            if not feature_count and feature_obj.type == 'MESH' and feature_obj.data:
                mesh = feature_obj.data
                feature_count = (
                    len(mesh.polygons) or len(mesh.edges) or len(mesh.vertices)
                )
            info_row = col.row()
            info_row.scale_y = 0.8
            info_row.label(text=f"  {feature_count} features", icon='CHECKMARK')
        elif feature_obj:
            warning_row = col.row()
            warning_row.scale_y = 0.8
            warning_row.alert = True
            warning_row.label(text="  Not a valid feature object", icon='ERROR')
        
        layout.separator()
        
        box = layout.box()
        box.label(text="Graph Type", icon='MESH_DATA')
        
        col = box.column(align=True)
        col.prop(props, "prox_graph_type", text="Type")
        
        layout.separator()
        
        box = layout.box()
        box.label(text="Distance Metric", icon='DRIVER_DISTANCE')
        
        col = box.column(align=True)
        col.prop(props, "prox_distance_metric", text="Metric")
        
        if props.prox_distance_metric == 'NETWORK':
            col.separator()
            col.prop(props, "prox_network_object", text="Network")
            
            network_obj = props.prox_network_object
            if network_obj and network_obj.get("is_osmnx"):
                info_row = col.row()
                info_row.scale_y = 0.8
                info_row.label(text="  OSMnx network", icon='CHECKMARK')
            elif network_obj:
                warning_row = col.row()
                warning_row.scale_y = 0.8
                warning_row.alert = True
                warning_row.label(text="  Not an OSMnx network", icon='ERROR')
        
        layout.separator()
        
        box = layout.box()
        box.label(text="Parameters", icon='PREFERENCES')
        
        col = box.column(align=True)
        
        graph_type = props.prox_graph_type
        
        if graph_type == 'KNN':
            col.prop(props, "prox_knn_k")
        elif graph_type == 'FIXED_RADIUS':
            col.prop(props, "prox_radius")
        elif graph_type == 'WAXMAN':
            col.prop(props, "prox_waxman_beta")
            col.prop(props, "prox_waxman_r0")
            col.prop(props, "prox_waxman_seed")
        elif graph_type == 'CONTIGUITY':
            col.prop(props, "prox_contiguity_type")
            # Note: predicate parameter not used by city2graph.contiguity_graph()
            # col.prop(props, "prox_contiguity_predicate")
        
        layout.separator()
        
        req_box = layout.box()
        req_box.scale_y = 0.7
        req_box.label(text="Requirements:", icon='INFO')
        req_box.label(text="• Select a feature object above (OSM or city2graph)")
        if props.prox_distance_metric == 'NETWORK':
            req_box.label(text="• Select OSMnx street network")
        
        layout.separator()
        
        row = layout.row()
        row.scale_y = 1.3
        row.operator("scigraphs.generate_proximity_graph", icon='PLAY', text="Generate Graph")


class SCIGRAPHS_PT_c2g_proximity_multi(bpy.types.Panel):
    """Multi-layer proximity graphs panel."""
    bl_label = "Multi-Layer Graphs"
    bl_parent_id = "SCIGRAPHS_PT_c2g_proximity"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'City2Graph'
    bl_options = {'DEFAULT_CLOSED'}
    
    def draw(self, context):
        layout = self.layout
        props = context.scene.city2graph
        
        layout.use_property_split = True
        layout.use_property_decorate = False
        
        box = layout.box()
        box.label(text="Feature Layers", icon='OUTLINER_OB_GROUP_INSTANCE')
        
        info_box = box.box()
        info_box.scale_y = 0.7
        info_box.label(text="Connect different feature types (e.g., restaurants → hospitals)")
        
        box.separator()
        
        col = box.column(align=True)
        col.prop(props, "prox_layer1_object", text="Layer 1")
        
        layer1_obj = props.prox_layer1_object
        if layer1_obj and (layer1_obj.get("is_osm_features") or layer1_obj.get("is_city2graph")):
            count = layer1_obj.get("feature_count", 0)
            info_row = col.row()
            info_row.scale_y = 0.8
            info_row.label(text=f"  {count} features", icon='CHECKMARK')
        
        col.separator()
        col.prop(props, "prox_layer2_object", text="Layer 2")
        
        layer2_obj = props.prox_layer2_object
        if layer2_obj and (layer2_obj.get("is_osm_features") or layer2_obj.get("is_city2graph")):
            count = layer2_obj.get("feature_count", 0)
            info_row = col.row()
            info_row.scale_y = 0.8
            info_row.label(text=f"  {count} features", icon='CHECKMARK')
        
        col.separator()
        col.prop(props, "prox_layer3_object", text="Layer 3 (Optional)")
        
        layer3_obj = props.prox_layer3_object
        if layer3_obj and (layer3_obj.get("is_osm_features") or layer3_obj.get("is_city2graph")):
            count = layer3_obj.get("feature_count", 0)
            info_row = col.row()
            info_row.scale_y = 0.8
            info_row.label(text=f"  {count} features", icon='CHECKMARK')
        
        layout.separator()
        
        box = layout.box()
        box.label(text="Connection Method", icon='LINKED')
        
        col = box.column(align=True)
        col.prop(props, "prox_multilayer_method", text="Method")
        
        if props.prox_multilayer_method == 'KNN':
            col.prop(props, "prox_multilayer_k")
        else:
            col.prop(props, "prox_multilayer_radius")
        
        col.separator()
        col.prop(props, "prox_distance_metric", text="Metric")
        
        if props.prox_distance_metric == 'NETWORK':
            col.separator()
            col.prop(props, "prox_network_object", text="Network")
        
        layout.separator()
        
        req_box = layout.box()
        req_box.scale_y = 0.7
        req_box.label(text="Requirements:", icon='INFO')
        req_box.label(text="• Select at least 2 feature layers")
        req_box.label(text="• Features will be connected across layers")
        req_box.label(text="• Node attributes created for coloring")
        
        layout.separator()
        
        row = layout.row()
        row.scale_y = 1.3
        row.operator("scigraphs.generate_multilayer_graph", icon='PLAY', text="Generate Multi-Layer Graph")


class SCIGRAPHS_PT_c2g_proximity_group(bpy.types.Panel):
    """Group nodes (polygon-to-point) graphs panel."""
    bl_label = "Group Nodes (Polygon → Point)"
    bl_parent_id = "SCIGRAPHS_PT_c2g_proximity"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'City2Graph'
    bl_options = {'DEFAULT_CLOSED'}
    
    def draw(self, context):
        layout = self.layout
        props = context.scene.city2graph
        
        layout.use_property_split = True
        layout.use_property_decorate = False
        
        box = layout.box()
        box.label(text="Spatial Containment", icon='MESH_GRID')
        
        info_box = box.box()
        info_box.scale_y = 0.7
        info_box.label(text="Connect polygon zones to points they contain")
        info_box.label(text="Example: Districts → Buildings, Zones → Amenities")
        
        box.separator()
        
        col = box.column(align=True)
        col.prop(props, "prox_polygons_object", text="Polygons")
        
        polygons_obj = props.prox_polygons_object
        if polygons_obj and (polygons_obj.get("is_osm_features") or polygons_obj.get("is_city2graph")):
            info_row = col.row()
            info_row.scale_y = 0.8
            info_row.label(text=f"  {polygons_obj.get('feature_count', 0)} features", icon='CHECKMARK')
        
        col.separator()
        col.prop(props, "prox_points_object", text="Points")
        
        points_obj = props.prox_points_object
        if points_obj and (points_obj.get("is_osm_features") or points_obj.get("is_city2graph")):
            info_row = col.row()
            info_row.scale_y = 0.8
            info_row.label(text=f"  {points_obj.get('feature_count', 0)} features", icon='CHECKMARK')
        
        layout.separator()
        
        box = layout.box()
        box.label(text="Containment Rule", icon='SHADING_BBOX')
        
        col = box.column(align=True)
        col.prop(props, "prox_group_predicate", text="Predicate")
        col.prop(props, "prox_distance_metric", text="Metric")
        
        if props.prox_distance_metric == 'NETWORK':
            col.separator()
            col.prop(props, "prox_network_object", text="Network")
        
        layout.separator()
        
        req_box = layout.box()
        req_box.scale_y = 0.7
        req_box.label(text="Requirements:", icon='INFO')
        req_box.label(text="• Select polygon and point feature objects")
        req_box.label(text="• Points will be connected to containing polygons")
        
        layout.separator()
        
        row = layout.row()
        row.scale_y = 1.3
        row.operator("scigraphs.generate_group_nodes_graph", icon='PLAY', text="Generate Group Nodes")


class SCIGRAPHS_PT_c2g_proximity_advanced(bpy.types.Panel):
    """Advanced proximity graph parameters panel."""
    bl_label = "Advanced Settings"
    bl_parent_id = "SCIGRAPHS_PT_c2g_proximity"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'City2Graph'
    bl_options = {'DEFAULT_CLOSED'}
    
    def draw(self, context):
        layout = self.layout
        props = context.scene.city2graph
        
        layout.use_property_split = True
        layout.use_property_decorate = False
        
        box = layout.box()
        box.label(text="Point Processing", icon='MESH_DATA')
        
        info_box = box.box()
        info_box.scale_y = 0.7
        info_box.label(text="Control duplicate point handling")
        info_box.label(text="Useful for OSM ways/relations imported as vertices")
        
        box.separator()
        
        col = box.column(align=True)
        col.prop(props, "prox_deduplicate")
        
        if props.prox_deduplicate:
            col.prop(props, "prox_dedup_tolerance")
            
            info_row = col.row()
            info_row.scale_y = 0.8
            info_row.label(text=f"  Points within {props.prox_dedup_tolerance}m will merge", icon='INFO')


class SCIGRAPHS_PT_c2g_proximity_viz(bpy.types.Panel):
    """Proximity graph visualization options panel."""
    bl_label = "Visualization"
    bl_parent_id = "SCIGRAPHS_PT_c2g_proximity"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'City2Graph'
    bl_options = {'DEFAULT_CLOSED'}
    
    def draw(self, context):
        layout = self.layout
        props = context.scene.city2graph
        
        layout.use_property_split = True
        layout.use_property_decorate = False
        
        box = layout.box()
        box.label(text="Edge Display", icon='CURVE_DATA')
        
        col = box.column(align=True)
        col.prop(props, "prox_curve_thickness")
        col.prop(props, "prox_visualize_limit")
        
        layout.separator()
        
        box = layout.box()
        box.label(text="Setup & Coloring", icon='COLOR')
        
        info_box = box.box()
        info_box.scale_y = 0.7
        info_box.label(text="Generated graphs are native SciGraphs meshes.")
        info_box.label(text="1. Setup Visualization to instance nodes/edges.")
        info_box.label(text="2. Open Coloring to colour by attribute.")
        
        col = box.column(align=True)
        col.scale_y = 1.2
        col.operator("scigraphs.setup_visualization", icon='MOD_HUE_SATURATION', text="Setup Visualization")
        col.operator("scigraphs.color_show_toolbar", icon='COLOR', text="Open Coloring")


classes = [
    SCIGRAPHS_PT_c2g_proximity,
    SCIGRAPHS_PT_c2g_proximity_single,
    SCIGRAPHS_PT_c2g_proximity_multi,
    SCIGRAPHS_PT_c2g_proximity_group,
    SCIGRAPHS_PT_c2g_proximity_advanced,
    SCIGRAPHS_PT_c2g_proximity_viz,
]


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)

