"""
Metapath analysis UI panel for heterogeneous graph analysis.
"""

import bpy


class SCIGRAPHS_PT_c2g_metapaths(bpy.types.Panel):
    """Metapath Analysis panel."""
    bl_label = "Metapath Analysis"
    bl_parent_id = "SCIGRAPHS_PT_city2graph_main"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'City2Graph'
    bl_options = {'DEFAULT_CLOSED'}
    
    def draw(self, context):
        layout = self.layout
        props = context.scene.city2graph
        obj = context.active_object
        
        layout.use_property_split = True
        layout.use_property_decorate = False
        
        box = layout.box()
        box.label(text="Heterogeneous Graph Analysis", icon='OUTLINER_OB_CURVE')
        
        info_box = box.box()
        info_box.scale_y = 0.7
        col = info_box.column(align=True)
        col.label(text="Connects amenities via street network")
        col.label(text="Uses City2Graph metapath algorithm")
        
        layout.separator()
        
        box = layout.box()
        box.label(text="Configuration", icon='SETTINGS')
        
        col = box.column(align=True)
        col.prop(props, "metapath_amenities_object", text="Amenities")
        
        amenities_obj = props.metapath_amenities_object
        if amenities_obj and (amenities_obj.get("is_osm_features") or amenities_obj.get("is_city2graph")):
            feature_count = amenities_obj.get("feature_count", 0)
            info_row = col.row()
            info_row.scale_y = 0.8
            info_row.label(text=f"  {feature_count} features", icon='CHECKMARK')
        elif amenities_obj:
            warning_row = col.row()
            warning_row.scale_y = 0.8
            warning_row.alert = True
            warning_row.label(text="  Not a valid features object", icon='ERROR')
        
        layout.separator()
        
        box = layout.box()
        box.label(text="Complete Analysis", icon='AUTO')
        
        col = box.column(align=True)
        col.prop(props, "metapath_hops")
        col.prop(props, "metapath_k_neighbors")
        col.prop(props, "metapath_amenity_limit")
        
        box.separator()
        
        col = box.column(align=True)
        col.prop(props, "metapath_visualize_limit")
        col.prop(props, "metapath_curve_thickness")
        
        box.separator()
        
        req_box = box.box()
        req_box.scale_y = 0.7
        req_box.label(text="Requirements:", icon='INFO')
        req_box.label(text="• Select OSMnx street network in 3D view")
        req_box.label(text="• Choose amenities object above")
        
        box.separator()
        
        row = box.row()
        row.scale_y = 1.3
        row.operator("scigraphs.compute_metapaths_wizard", 
                     icon='PLAY', text="Run Complete Analysis")
        
        layout.separator()
        
        # --- Weighted Metapaths ---
        box = layout.box()
        box.label(text="Weighted Metapaths (Dijkstra)", icon='FORCE_CURVE')
        
        info_box = box.box()
        info_box.scale_y = 0.7
        info_box.label(text="Connect nodes reachable within a cost threshold")
        info_box.label(text="Requires bridges computed first")
        
        col = box.column(align=True)
        col.prop(props, "metapath_weight_attr")
        col.prop(props, "metapath_weight_threshold")
        col.prop(props, "metapath_weight_min_threshold")
        col.prop(props, "metapath_endpoint_type")
        
        box.separator()
        row = box.row()
        row.scale_y = 1.2
        row.operator("scigraphs.compute_metapaths_by_weight", icon='PLAY', text="Compute Weighted Metapaths")
        
        layout.separator()
        
        box = layout.box()
        box.label(text="Advanced: Step-by-Step", icon='PROPERTIES')
        
        help_box = box.box()
        help_box.scale_y = 0.7
        help_box.label(text="For advanced users:", icon='INFO')
        help_box.label(text="Run operators individually for debugging")
        
        box.separator()
        
        col = box.column(align=True)
        col.label(text="1. Create Street Dual Graph:")
        col.operator("scigraphs.create_street_dual_graph", icon='MESH_DATA')
        
        col.separator()
        col.label(text="2. Bridge Amenities:")
        col.operator("scigraphs.bridge_amenities", icon='STICKY_UVS_LOC')
        
        col.separator()
        col.label(text="3. Compute Metapaths:")
        col.operator("scigraphs.compute_metapaths", icon='CURVE_DATA')
        
        if obj:
            layout.separator()
            info_box = layout.box()
            info_box.scale_y = 0.8
            
            if obj.get("is_street_dual"):
                num_nodes = obj.get('num_nodes', 0)
                num_edges = obj.get('num_edges', 0)
                info_box.label(text="Selected: Street Dual Graph", icon='MESH_DATA')
                info_box.label(text=f"Segments: {num_nodes:,} | Connections: {num_edges:,}")
                
                if obj.get("has_metapath_bridges"):
                    num_bridges = obj.get('num_bridges', 0)
                    info_box.label(text=f"Bridges: {num_bridges} amenity connections")
                    info_box.label(text="Ready to compute metapaths")
            
            elif obj.get("is_metapath_result"):
                raw_count = obj.get("num_metapaths_raw", 0)
                unique_count = obj.get("num_metapaths_unique", 0)
                hops = obj.get("metapath_hops", "?")
                num_viz = obj.get("num_visualized", 0)
                avg_mult = obj.get("avg_multiplicity", 0)
                min_mult = obj.get("min_multiplicity", 0)
                max_mult = obj.get("max_multiplicity", 0)
                
                info_box.label(text="Selected: Metapath Result", icon='CURVE_DATA')
                info_box.label(text=f"Total: {raw_count:,} | Unique: {unique_count:,} ({hops}-hop)")
                
                if avg_mult > 0:
                    info_box.label(text=f"Multiplicity: {min_mult}-{max_mult} (avg {avg_mult:.1f}x)")
                
                info_box.label(text=f"Visualized: {num_viz:,} curves")
                
                # Convert to mesh button
                info_box.separator()
                info_box.operator("scigraphs.convert_metapaths_to_mesh", 
                                 icon='MESH_DATA', 
                                 text="Convert to Mesh (View in Spreadsheet)")
                
                # Hint about accessing multiplicity data
                hint_box = info_box.box()
                hint_box.scale_y = 0.6
                hint_box.label(text="Multiplicity stored in custom properties", icon='INFO')
                hint_box.label(text="Convert to mesh to view in Spreadsheet Editor")
            
            elif obj.get("is_metapath_mesh"):
                raw_count = obj.get("num_metapaths_raw", 0)
                unique_count = obj.get("num_metapaths_unique", 0)
                hops = obj.get("metapath_hops", "?")
                
                info_box.label(text="Selected: Metapath Mesh", icon='MESH_DATA')
                info_box.label(text=f"Total: {raw_count:,} | Unique: {unique_count:,} ({hops}-hop)")
                
                # Check if attribute exists
                if obj.data and obj.data.attributes.get("multiplicity"):
                    info_box.label(text="Attribute 'multiplicity' available", icon='CHECKMARK')
                    
                    hint_box = info_box.box()
                    hint_box.scale_y = 0.6
                    hint_box.label(text="View in Spreadsheet Editor:", icon='INFO')
                    hint_box.label(text="Set domain to 'Edge' to see multiplicity")


classes = [
    SCIGRAPHS_PT_c2g_metapaths,
]


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)

