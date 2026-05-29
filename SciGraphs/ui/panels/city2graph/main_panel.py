import bpy


class SCIGRAPHS_PT_city2graph_main(bpy.types.Panel):
    """City2Graph main panel."""
    bl_label = "City2Graph"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'City2Graph'
    
    def draw(self, context):
        from ... import gizmos
        gizmos.set_active_toolbar(context, 'CITY2GRAPH')

        layout = self.layout
        obj = context.active_object
        
        box = layout.box()
        box.label(text="Urban Analytics & GeoAI", icon='WORLD')
        
        if obj and (obj.get("is_city2graph") or obj.get("is_osmnx")):
            info = box.box()
            info.scale_y = 0.8
            info.label(text=f"Object: {obj.name}", icon='MESH_DATA')
            
            if obj.get("feature_count"):
                info.label(text=f"Features: {obj.get('feature_count')}")
            if obj.get("num_nodes"):
                nodes = obj.get("num_nodes", 0)
                edges = obj.get("num_edges", 0)
                info.label(text=f"Nodes: {nodes:,} | Edges: {edges:,}")
            
            if obj.get("is_osmnx"):
                info.label(text="Type: OSMnx Street Network", icon='FORCE_CURVE')
                info.label(text="Can be used with City2Graph operations")
            elif obj.get("is_tessellation"):
                info.label(text="Type: Urban Tessellation", icon='MESH_GRID')
            elif obj.get("is_morphological_graph"):
                info.label(text="Type: Morphological Graph", icon='OUTLINER_OB_CURVE')
            elif obj.get("is_travel_graph"):
                info.label(text="Type: Travel Graph", icon='ANIM')
        else:
            info = box.box()
            info.scale_y = 0.8
            info.label(text="No City2Graph or OSMnx object selected", icon='INFO')
            info.label(text="Use panels below to import data")


def register():
    bpy.utils.register_class(SCIGRAPHS_PT_city2graph_main)


def unregister():
    bpy.utils.unregister_class(SCIGRAPHS_PT_city2graph_main)

