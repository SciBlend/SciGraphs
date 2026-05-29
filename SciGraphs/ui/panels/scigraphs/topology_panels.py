# Topological analysis panels for SciGraphs addon
# Provides UI for planarity, genus, face detection, and surface embedding

import bpy


class SCIGRAPHS_PT_topology(bpy.types.Panel):
    """Main topology analysis panel."""
    bl_label = "Topological Analysis"
    bl_parent_id = "SCIGRAPHS_PT_analysis"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_options = {'DEFAULT_CLOSED'}
    
    def draw(self, context):
        layout = self.layout
        props = context.scene.scigraphs
        obj = context.active_object
        
        if not obj or "num_nodes" not in obj:
            box = layout.box()
            box.label(text="No graph loaded", icon='ERROR')
            box.label(text="Create a graph first in Data panel")
            return
        
        # Mode selector
        layout.prop(props, "topology_analysis_mode", expand=True)


class SCIGRAPHS_PT_topology_surface(bpy.types.Panel):
    """Surface embedding analysis (planarity, genus, faces)."""
    bl_label = "Surface Embedding"
    bl_parent_id = "SCIGRAPHS_PT_topology"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_options = {'DEFAULT_CLOSED'}
    
    @classmethod
    def poll(cls, context):
        props = context.scene.scigraphs
        return props.topology_analysis_mode == 'SURFACE'
    
    def draw(self, context):
        layout = self.layout
        props = context.scene.scigraphs
        obj = context.active_object
        
        if not obj or "num_nodes" not in obj:
            return
        
        layout.use_property_split = True
        layout.use_property_decorate = False
        
        # Surface type selector
        box = layout.box()
        box.label(text="Embedding Surface", icon='SURFACE_NSURFACE')
        box.prop(props, "topology_surface_type", text="")
        
        # Planarity check
        layout.separator()
        box = layout.box()
        box.label(text="Planarity Analysis", icon='MESH_PLANE')
        
        row = box.row()
        row.scale_y = 1.3
        row.operator("scigraphs.check_planarity", text="Check Planarity", icon='VIEWZOOM')
        
        # Show planarity result if available
        if "topo_is_planar" in obj:
            info = box.box()
            info.scale_y = 0.8
            
            is_planar = obj.get("topo_is_planar", False)
            if is_planar:
                info.label(text="Graph is PLANAR", icon='CHECKMARK')
            else:
                info.label(text="Graph is NON-PLANAR", icon='X')
                k_type = obj.get("topo_kuratowski_type", "")
                if k_type:
                    info.label(text=f"Contains {k_type} subdivision")
        
        # Genus calculation
        layout.separator()
        box = layout.box()
        box.label(text="Genus Computation", icon='MESH_TORUS')
        
        row = box.row()
        row.scale_y = 1.3
        row.operator("scigraphs.calculate_genus", text="Calculate Genus", icon='PLAY')
        
        # Show genus result
        if "topo_genus_lower_bound" in obj:
            info = box.box()
            info.scale_y = 0.8
            
            is_planar = obj.get("topo_is_planar", False)
            genus_bound = obj.get("topo_genus_lower_bound", 0)
            
            if is_planar:
                info.label(text="Genus = 0 (planar)")
            else:
                info.label(text=f"Genus >= {genus_bound}")
            
            # Euler characteristic
            if "topo_euler_V" in obj:
                V = obj.get("topo_euler_V", 0)
                E = obj.get("topo_euler_E", 0)
                info.label(text=f"V={V}, E={E}")
                
                if "topo_euler_F" in obj:
                    F = obj.get("topo_euler_F", 0)
                    chi = obj.get("topo_euler_chi", 0)
                    info.label(text=f"F={F}, chi={chi}")
        
        # Face computation (only for planar graphs)
        if obj.get("topo_is_planar", False):
            layout.separator()
            box = layout.box()
            box.label(text="Face Detection", icon='FACE_MAPS')
            
            box.prop(props, "topology_show_faces")
            
            row = box.row()
            row.scale_y = 1.2
            row.operator("scigraphs.compute_faces", text="Compute Faces", icon='PLAY')
            
            if "topo_num_faces" in obj:
                info = box.box()
                info.scale_y = 0.8
                num_faces = obj.get("topo_num_faces", 0)
                info.label(text=f"Found {num_faces} faces")
                info.label(text="Attribute: face_id", icon='NODE')
            
            # Dual Graph section (Chapter 2.6 - Mohar & Thomassen)
            layout.separator()
            box = layout.box()
            box.label(text="Geometric Dual Graph (G*)", icon='MOD_MESHDEFORM')
            
            dual_info = box.box()
            dual_info.scale_y = 0.7
            dual_info.label(text="Creates dual graph where:")
            dual_info.label(text="  - Vertices = face centroids")
            dual_info.label(text="  - Edges = adjacent faces")
            
            row = box.row()
            row.scale_y = 1.2
            row.operator("scigraphs.create_dual_graph", text="Create Dual Graph", icon='OUTLINER_OB_MESH')
            
            # Show dual graph controls if exists
            dual_name = obj.get("topo_dual_child", "")
            if dual_name and dual_name in bpy.data.objects:
                dual_box = box.box()
                dual_box.scale_y = 0.8
                dual_box.label(text=f"Dual: {dual_name}", icon='CHECKMARK')
                
                row = dual_box.row(align=True)
                dual_obj = bpy.data.objects[dual_name]
                icon = 'HIDE_OFF' if not dual_obj.hide_viewport else 'HIDE_ON'
                row.operator("scigraphs.toggle_dual_graph", text="Toggle", icon=icon)
                row.operator("scigraphs.remove_dual_graph", text="Remove", icon='X')
        
        # Surface visualization
        layout.separator()
        box = layout.box()
        box.label(text="Embedding Layout", icon='RESTRICT_VIEW_OFF')
        
        # Operator
        row = box.row()
        row.scale_y = 1.2
        row.operator("scigraphs.visualize_surface", text="Compute Embedding", icon='MOD_WARP')
        
        # Info about visualization - explain what happens
        viz_info = box.box()
        viz_info.scale_y = 0.7
        if props.topology_surface_type == 'PLANE':
            viz_info.label(text="Chrobak-Payne algorithm", icon='CHECKMARK')
            viz_info.label(text="Edges will NOT cross!")
            viz_info.label(text="Creates: Plane surface mesh")
        
        # Validate crossings
        box.separator()
        row = box.row()
        row.operator("scigraphs.validate_crossings", text="Validate Crossings", icon='ERROR')
        
        if "topo_has_crossings" in obj:
            cross_info = box.box()
            cross_info.scale_y = 0.8
            has_crossings = obj.get("topo_has_crossings", False)
            num_crossings = obj.get("topo_num_crossings", 0)
            if has_crossings:
                cross_info.label(text=f"{num_crossings} edge crossing(s) found!", icon='ERROR')
            else:
                cross_info.label(text="No edge crossings!", icon='CHECKMARK')
        
        # Show surface info if exists
        surface_name = obj.get("topo_surface_child", "")
        if surface_name and surface_name in bpy.data.objects:
            layout.separator()
            surf_box = layout.box()
            surf_box.label(text="Generated Surface", icon='MESH_UVSPHERE')
            
            col = surf_box.column(align=True)
            col.scale_y = 0.8
            col.label(text=f"Mesh: {surface_name}")
            
            layout_type = obj.get("topo_layout_type", "unknown")
            col.label(text=f"Type: {layout_type}")
            
            if "num_curve_verts" in obj:
                curve_verts = obj.get("num_curve_verts", 0)
                total_verts = obj.get("num_mesh_verts", 0)
                num_nodes = obj.get("num_nodes", 0)
                col.label(text=f"Total verts: {total_verts}")
                col.label(text=f"  Nodes (is_intersection=1): {num_nodes}")
                col.label(text=f"  Curve points (is_intersection=0): {curve_verts}")
            
            # Toggle and remove buttons
            row = surf_box.row(align=True)
            surface_obj = bpy.data.objects[surface_name]
            icon = 'HIDE_OFF' if not surface_obj.hide_viewport else 'HIDE_ON'
            row.operator("scigraphs.toggle_topo_surface", text="Toggle", icon=icon)
            row.operator("scigraphs.remove_topo_surface", text="Remove", icon='X')


class SCIGRAPHS_PT_topology_results(bpy.types.Panel):
    """Summary of all topology results."""
    bl_label = "Results Summary"
    bl_parent_id = "SCIGRAPHS_PT_topology"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_options = {'DEFAULT_CLOSED'}
    
    @classmethod
    def poll(cls, context):
        obj = context.active_object
        # Only show if some analysis has been done
        return obj and any(key.startswith("topo_") for key in obj.keys())
    
    def draw(self, context):
        layout = self.layout
        obj = context.active_object
        
        if not obj:
            return
        
        layout.use_property_split = False
        
        box = layout.box()
        box.label(text="Topology Properties", icon='INFO')
        
        col = box.column(align=True)
        col.scale_y = 0.8
        
        # Surface embedding results
        if "topo_is_planar" in obj:
            is_planar = obj.get("topo_is_planar", False)
            col.label(text=f"Planar: {'Yes' if is_planar else 'No'}")
        
        if "topo_genus_lower_bound" in obj:
            genus = obj.get("topo_genus_lower_bound", 0)
            col.label(text=f"Genus bound: {genus}")
        
        if "topo_num_faces" in obj:
            faces = obj.get("topo_num_faces", 0)
            col.label(text=f"Faces: {faces}")
        
        col.separator()
        
        # Spatial results
        if "topo_num_cycles" in obj:
            cycles = obj.get("topo_num_cycles", 0)
            col.label(text=f"Cycles: {cycles}")
        
        if "topo_num_linked_pairs" in obj:
            linked = obj.get("topo_num_linked_pairs", 0)
            col.label(text=f"Linked pairs: {linked}")
        
        if "topo_intrinsically_linked" in obj:
            intrinsic = obj.get("topo_intrinsically_linked", False)
            col.label(text=f"Intrinsically linked: {'Yes' if intrinsic else 'No'}")
        
        # Euler characteristic
        if "topo_euler_V" in obj and "topo_euler_E" in obj:
            V = obj.get("topo_euler_V", 0)
            E = obj.get("topo_euler_E", 0)
            F = obj.get("topo_euler_F", None)
            
            col.separator()
            col.label(text="Euler Characteristic:")
            col.label(text=f"  V = {V}")
            col.label(text=f"  E = {E}")
            if F is not None:
                chi = obj.get("topo_euler_chi", V - E + F)
                col.label(text=f"  F = {F}")
                col.label(text=f"  chi = {chi}")


def register():
    bpy.utils.register_class(SCIGRAPHS_PT_topology)
    bpy.utils.register_class(SCIGRAPHS_PT_topology_surface)
    bpy.utils.register_class(SCIGRAPHS_PT_topology_results)


def unregister():
    bpy.utils.unregister_class(SCIGRAPHS_PT_topology_results)
    bpy.utils.unregister_class(SCIGRAPHS_PT_topology_surface)
    bpy.utils.unregister_class(SCIGRAPHS_PT_topology)

